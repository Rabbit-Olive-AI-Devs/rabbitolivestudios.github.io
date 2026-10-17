#!/usr/bin/env node
/**
 * The series mailbox watcher (DECISIONS #72). Runs on the Mac mini from launchd:
 *
 *   node scripts/series-mailbox.mjs run      — handle Jarvis's messages in to-claude/
 *   node scripts/series-mailbox.mjs weekly   — write Claude's next chapter when it is due
 *   node scripts/series-mailbox.mjs review S C <package.zip>  — re-run Claude's independent review (#79)
 *   node scripts/series-mailbox.mjs art-brief [YYYY-MM-DD]   — write Jarvis's picture brief (default: tomorrow, #89)
 *
 * Layer 1 (no AI): APPROVE, DELIVERY, STYLE TEST and ART DELIVERY are handled by this script with the
 * import tool. Layer 2: QUESTION, FYI and anything else go to an unattended Claude run that
 * can READ but may WRITE only into from-claude/ (no shell, no code, no deploys), because
 * a message is text from another AI and must never be able to steer the engineer. The weekly
 * chapter run works from Claude's own notes, may write only in ~/series-work and from-claude/,
 * and cannot upload: this script uploads its build once the import has passed.
 *
 * Nothing here holds story. Logs: ~/Library/Logs/eink-series/.
 */
import { execFileSync, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const HOME = os.homedir();
const MAILBOX = process.env.SERIES_MAILBOX ?? "/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox/00_SHARED_INBOX/series";
const INBOX = path.join(MAILBOX, "to-claude");
const OUTBOX = path.join(MAILBOX, "from-claude");
const WORK = path.join(HOME, "series-work");
const LOGS = path.join(HOME, "Library", "Logs", "eink-series");
const ATTEMPTS = path.join(WORK, ".mailbox-attempts.json");
const SITE = "https://eink-dashboard.thiago-oliveira77.workers.dev";
const PROOFS = process.env.SERIES_PROOFS ?? "/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox/00_JARVIS_SPOILERS/series-proofs";
const MAX_ATTEMPTS = 3;

fs.mkdirSync(LOGS, { recursive: true });
const log = (s) => fs.appendFileSync(path.join(LOGS, "mailbox.log"), `${new Date().toISOString()} ${s}\n`);

/** Chicago wall clock, for file names Jarvis reads. */
function stamp() {
  const p = Object.fromEntries(new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(new Date()).map((x) => [x.type, x.value]));
  return { date: `${p.year}-${p.month}-${p.day}`, file: `${p.year}-${p.month}-${p.day}-${p.hour}${p.minute}` };
}

function reply(slug, body) {
  const name = `${stamp().file}-${slug}.md`;
  fs.writeFileSync(path.join(OUTBOX, name), body.trim() + "\n");
  fs.appendFileSync(path.join(OUTBOX, "MAILBOX-LOG.md"), `- ${stamp().file}: ${slug}\n`);
  return name;
}

/** One lock per job: a chapter being written must not hold up approvals. */
function lock(lockFile) {
  try {
    const age = Date.now() - fs.statSync(lockFile).mtimeMs;
    if (age < 3 * 3600 * 1000) return false; // another run is working (a chapter can take ~20 min)
  } catch { /* no lock */ }
  fs.mkdirSync(WORK, { recursive: true });
  fs.writeFileSync(lockFile, String(process.pid));
  return true;
}

function compile() {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "series-mailbox-"));
  execFileSync("npx", ["tsc", "-p", "tsconfig.test.json", "--outDir", out], { cwd: REPO, stdio: "ignore" });
  return { mailbox: require(path.join(out, "src/series/mailbox.js")) };
}

/** Run the import tool; returns { ok, out }. */
function series(args) {
  const r = spawnSync("node", ["scripts/series-import.mjs", ...args], { cwd: REPO, encoding: "utf8", env: process.env, timeout: 30 * 60 * 1000 });
  return { ok: r.status === 0, out: `${r.stdout ?? ""}${r.stderr ?? ""}`.trim() };
}

/** An unattended Claude run with a fixed instruction file and tight tool rules. */
function claude(instructionsFile, vars, allowedTools, logName) {
  let prompt = fs.readFileSync(path.join(REPO, "scripts", "series-agent", instructionsFile), "utf8");
  for (const [k, v] of Object.entries(vars)) prompt = prompt.replaceAll(`{{${k}}}`, v);
  const r = spawnSync("claude", [
    "-p", prompt, "--model", "opus", "--permission-mode", "dontAsk",
    "--allowedTools", allowedTools.join(" "), "--add-dir", MAILBOX, "--add-dir", WORK,
  ], { cwd: REPO, encoding: "utf8", env: process.env, timeout: 90 * 60 * 1000, maxBuffer: 64 * 1024 * 1024 });
  fs.writeFileSync(path.join(LOGS, `${stamp().file}-${logName}.log`), `${r.stdout ?? ""}\n---\n${r.stderr ?? ""}`);
  return r.status === 0;
}

const writeTo = (dir) => [`Write(/${dir}/**)`, `Edit(/${dir}/**)`];

// ---------------------------------------------------------------- messages

function handle(file, m) {
  const text = fs.readFileSync(path.join(INBOX, file), "utf8");
  const msg = m.mailbox.parseMessage(text);
  switch (msg.kind) {
    case "APPROVE": {
      const lines = msg.chapters.map((c, i) => {
        const r = series(["approve", "--season", String(msg.season), "--chapter", String(c), "--revision", msg.revisions[i]]);
        return `- Season ${msg.season} chapter ${c} (revision ${msg.revisions[i]}): ${r.ok ? "approved; it airs on its Monday at 08:00." : `NOT approved: ${r.out.split("\n").pop()}`}`;
      });
      reply(`approved-s${msg.season}`, `FYI: reply to ${file}\n\n${lines.join("\n")}`);
      return true;
    }
    case "DELIVERY": {
      const zip = path.join(MAILBOX, "packages", msg.package);
      if (!fs.existsSync(zip)) { reply("delivery-missing", `FYI: reply to ${file}\n\nI cannot find packages/${msg.package}. Put the zip there and send DELIVERY again.`); return true; }
      const out = path.join(WORK, "inbox-out", msg.package.replace(/\.zip$/i, ""));
      const b = series(["build", zip, "--out", out]);
      if (!b.ok) {
        const report = fs.existsSync(path.join(out, "report.txt")) ? fs.readFileSync(path.join(out, "report.txt"), "utf8") : b.out.slice(-4000);
        reply("delivery-refused", `FYI: reply to ${file}\n\nThe import refused ${msg.package}. Fix these and deliver again:\n\n${report}`);
        return true;
      }
      const u = series(["upload", out, ...(msg.replaceApproved ? ["--replace-approved"] : [])]);
      const meta = JSON.parse(fs.readFileSync(path.join(out, "season.json"), "utf8"));
      const links = meta.chapters.map((c) => `- Chapter ${c}: ${SITE}/series/proof?season=${meta.season.number}&chapter=${c}&key=`).join("\n");
      const reviews = u.ok ? reviewUploaded(m, u.out, zip) : "";
      reply(u.ok ? "delivery-uploaded" : "delivery-upload-failed", u.ok
        ? `FYI: reply to ${file}\n\n${msg.package} passed every check and is uploaded. ${u.out.split("\n").filter((l) => /uploaded|not replaced/.test(l)).join(" ")}\n\n${exportLines(u.out)}${reviews}\n\nProof pages (add your key):\n${links}\n\nSend APPROVE season=${meta.season.number} chapter=N revision=R when you have reviewed that revision.`
        : `FYI: reply to ${file}\n\n${msg.package} passed the checks but the upload failed. Nothing changed on air. Thiago needs to ask Claude to look:\n\n${u.out.slice(-2000)}`);
      if (!u.ok) note(`Upload of ${msg.package} failed; see ~/Library/Logs/eink-series/mailbox.log`);
      if (/proof export FAILED/.test(u.out)) note(`The proof export for ${msg.package} failed, so Jarvis cannot review it yet. Please open a session.`);
      return true;
    }
    case "STYLE TEST": {
      const dir = path.join(MAILBOX, "style-tests", msg.folder);
      if (!fs.existsSync(dir)) { reply("style-test-missing", `FYI: reply to ${file}\n\nI cannot find style-tests/${msg.folder}/.`); return true; }
      const out = path.join(OUTBOX, "style-tests", msg.folder);
      const r = series(["preview", dir, "--out", out]);
      // Also put the pages where the panels can show them: /series/test (key needed).
      const links = [];
      for (const f of fs.readdirSync(out).filter((n) => n.endsWith(".html")).sort()) {
        const image = f.replace(/\.html$/, "");
        const put = spawnSync("npx", ["wrangler", "r2", "object", "put", `eink-series/series/tests/${msg.folder}/${f}`, "--file", path.join(out, f), "--content-type", "text/html; charset=utf-8", "--remote"], { cwd: REPO, encoding: "utf8", env: process.env });
        links.push(put.status === 0 ? `- ${image}: ${SITE}/series/test?folder=${msg.folder}&image=${image}&key=` : `- ${image}: upload failed`);
      }
      reply(`style-test-${msg.folder}`, `FYI: reply to ${file}\n\nEach picture, untouched and only cropped to the 800x480 screen, is in from-claude/style-tests/${msg.folder}/ as .png. The panels get the picture exactly like this and convert it themselves: the E1001s to black and white, the E1002 to its six colours.\n\n${r.out}\n\nTo see them on a real panel, add one of these as a temporary page in SenseCraft (HTML widget, 800x480) with your key at the end, and remove it afterwards. The same address works on every panel:\n${links.join("\n")}`);
      return true;
    }
    case "ART DELIVERY":
      return artDelivery(file, msg.date);
    case "INVALID":
      reply("format-problem", `FYI: reply to ${file}\n\n${msg.reason}. Nothing was done. The first line formats are:\n\n    APPROVE season=0 chapter=3 revision=<16 hex from the proof export>\n    DELIVERY package=season-01-chapter-01.zip [replace-approved]\n    STYLE TEST folder=openai-1\n    ART DELIVERY date=2026-10-18\n    QUESTION | FYI`);
      return true;
    default: {
      const read = ["Read", "Glob", "Grep"];
      return claude("message.md", { MESSAGE: path.join(INBOX, file), MAILBOX, OUTBOX, STAMP: stamp().file }, [...read, ...writeTo(OUTBOX)], `message-${file.replace(/\.md$/, "")}`);
    }
  }
}

/**
 * Claude's independent review of a chapter the showrunner wrote himself (DECISIONS #79). An
 * unattended run reads every exported page and the package, and writes PASS or REVISE. The
 * record that `approve` trusts stays in ~/series-work/reviews, where Jarvis cannot write; a copy
 * of the verdict goes next to the pages in his proof folder. Returns the verdict or null.
 */
function review(season, chapter, revision, exportDir, zip) {
  const pad = (n) => String(n).padStart(2, "0");
  const base = `s${pad(season)}-c${pad(chapter)}-${revision}`;
  const dir = path.join(WORK, "reviews");
  fs.mkdirSync(dir, { recursive: true });
  const unpacked = path.join(WORK, "review-packages", base);
  fs.rmSync(unpacked, { recursive: true, force: true });
  fs.mkdirSync(unpacked, { recursive: true });
  execFileSync("ditto", ["-x", "-k", zip, unpacked]);
  const packageSha = crypto.createHash("sha256").update(fs.readFileSync(zip)).digest("hex");
  const verdictFile = path.join(dir, `${base}.md`);
  fs.rmSync(verdictFile, { force: true });
  claude("review.md", {
    SEASON: String(season), CHAPTER: String(chapter), REVISION: revision, EXPORT: exportDir,
    PACKAGE: unpacked, PACKAGE_SHA: packageSha, VERDICT: verdictFile, MAILBOX,
  }, ["Read", "Glob", "Grep", `Edit(/${verdictFile})`], `review-${base}`);
  const first = fs.existsSync(verdictFile) ? fs.readFileSync(verdictFile, "utf8").split("\n")[0].trim() : "";
  const verdict = /^(PASS|REVISE)$/.test(first) ? first : null;
  if (!verdict) return null;
  const manifestSha = crypto.createHash("sha256").update(fs.readFileSync(path.join(exportDir, "MANIFEST.json"))).digest("hex");
  fs.writeFileSync(path.join(dir, `${base}.json`), JSON.stringify({
    season, chapter, revision, verdict, packageSha256: packageSha, manifestSha256: manifestSha, reviewedAt: new Date().toISOString(),
  }, null, 2) + "\n");
  fs.copyFileSync(verdictFile, path.join(exportDir, "CLAUDE-REVIEW.md"));
  return verdict;
}

/** Review every exported chapter of an upload that needs it; one line per chapter for Jarvis. */
function reviewUploaded(m, out, zip) {
  const lines = [];
  for (const l of out.split("\n")) {
    const e = l.match(/^proof export: season (\d+) chapter (\d+) revision ([0-9a-f]{16}) in (.+)$/);
    if (!e || !m.mailbox.needsIndependentReview(Number(e[1]))) continue;
    const v = review(Number(e[1]), Number(e[2]), e[3], e[4], zip);
    lines.push(v
      ? `- Season ${e[1]} chapter ${e[2]} revision ${e[3]}: Claude's independent review says **${v}**. Details: CLAUDE-REVIEW.md in that revision's proof folder.`
      : `- Season ${e[1]} chapter ${e[2]} revision ${e[3]}: the independent review did not finish; Thiago has been asked to open a session.`);
    if (!v) note(`Claude's independent review of season ${e[1]} chapter ${e[2]} did not finish. Please open a session.`);
  }
  return lines.length ? `\n\nIndependent review (Season 1 rule, DECISIONS #79):\n${lines.join("\n")}` : "";
}

/** The upload's proof-export lines, for Jarvis (DECISIONS #78). */
function exportLines(out) {
  const lines = out.split("\n").filter((l) => /^proof export/.test(l));
  return lines.length
    ? `Proof export for review (no key needed; check READY and MANIFEST.json first):\n${lines.map((l) => `- ${l}`).join("\n")}`
    : "The proof export did not run; Thiago has been asked to open a session.";
}

/** A spoiler-free line for Thiago, relayed by Jarvis. */
function note(line) {
  fs.appendFileSync(path.join(OUTBOX, "NEEDS-THIAGO.md"), `- ${stamp().file}: ${line}\n`);
}

function cmdRun() {
  if (!fs.existsSync(INBOX)) { log("mailbox not mounted; skipped"); return; }
  const files = fs.readdirSync(INBOX).filter((f) => f.endsWith(".md") && fs.statSync(path.join(INBOX, f)).isFile()).sort();
  // Files Dropbox may still be writing: wait until each is a minute old rather than skip it,
  // because launchd starts this run the moment a file lands and will not start it again until
  // the 30-minute backstop.
  const age = (f) => Date.now() - fs.statSync(path.join(INBOX, f)).mtimeMs;
  const youngest = Math.min(...files.map(age));
  if (files.length > 0 && youngest < 60 * 1000) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 60 * 1000 - youngest + 1000);
  const ready = files.filter((f) => fs.existsSync(path.join(INBOX, f)));
  if (ready.length === 0) return;
  const m = compile();
  const attempts = fs.existsSync(ATTEMPTS) ? JSON.parse(fs.readFileSync(ATTEMPTS, "utf8")) : {};
  fs.mkdirSync(path.join(INBOX, "done"), { recursive: true });
  for (const f of ready) {
    let ok = false;
    try { ok = handle(f, m); } catch (err) { log(`${f}: ${err.stack ?? err}`); }
    attempts[f] = (attempts[f] ?? 0) + 1;
    if (!ok && attempts[f] < MAX_ATTEMPTS) { log(`${f}: not handled (attempt ${attempts[f]}), will retry`); continue; }
    if (!ok) {
      reply("could-not-handle", `FYI: I could not handle ${f} after ${MAX_ATTEMPTS} tries. It waits for Thiago's next session with Claude.`);
      note(`A message from Jarvis (${f}) needs a Claude session.`);
    }
    fs.renameSync(path.join(INBOX, f), path.join(INBOX, "done", f));
    delete attempts[f];
    log(`${f}: ${ok ? "handled" : "gave up"}`);
  }
  fs.writeFileSync(ATTEMPTS, JSON.stringify(attempts));
}

// ---------------------------------------------------------------- Jarvis's daily pictures (#89)

const ART_OUT = path.join(OUTBOX, "art");
const ART_IN = path.join(INBOX, "art");
const ART_BUCKET = "eink-birthday-photos";
const ART_TYPES = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp" };
const ART_MAX_BYTES = 15 * 1024 * 1024;

/** Chicago date plus N days, for "tomorrow". */
function chicagoDate(offsetDays) {
  const d = new Date(`${stamp().date}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() + offsetDays);
  return d.toISOString().slice(0, 10);
}

/** Does the file start like the image type its name claims? */
function looksLikeImage(buf, ext) {
  if (ext === "png") return buf.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  if (ext === "jpg" || ext === "jpeg") return buf[0] === 0xff && buf[1] === 0xd8;
  if (ext === "webp") return buf.subarray(0, 4).toString() === "RIFF" && buf.subarray(8, 12).toString() === "WEBP";
  return false;
}

/**
 * The evening brief for Jarvis: what to draw for a date, as JSON, from the Worker's own
 * choices (event, styles, prompts). On a family birthday the face crops go with it.
 */
async function cmdArtBrief() {
  if (!fs.existsSync(OUTBOX)) { log("mailbox not mounted; art brief skipped"); return; }
  const date = process.argv[3] ?? chicagoDate(1);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Error("usage: art-brief [YYYY-MM-DD]");
  const res = await fetch(`${SITE}/art/brief?date=${date}`);
  if (!res.ok) throw new Error(`brief for ${date}: HTTP ${res.status} ${(await res.text()).slice(0, 200)}`);
  const brief = await res.json();
  fs.mkdirSync(ART_OUT, { recursive: true });
  // One brief per date: a complete one is never sent twice. An earlier one that was waiting for
  // the moment is replaced by a later run that has it.
  const json = path.join(ART_OUT, `${date}-brief.json`);
  if (fs.existsSync(json) && !JSON.parse(fs.readFileSync(json, "utf8")).waiting && !process.env.ART_FORCE) { log(`art brief ${date}: already sent`); return; }
  if (fs.existsSync(json) && brief.waiting) { log(`art brief ${date}: still waiting for ${brief.waiting.join(", ")}`); return; }
  const birthday = brief.pictures.find((p) => p.kind === "birthday");
  if (birthday) {
    const key = String(birthday.subject.key).replace(/[^a-z]/g, "");
    const refs = [];
    for (let i = 0; i < 4; i++) {
      const dest = path.join(ART_OUT, `${date}-ref-${i}.jpg`);
      const r = spawnSync("npx", ["wrangler", "r2", "object", "get", `${ART_BUCKET}/portraits/${key}_${i}.jpg`, "--file", dest, "--remote"], { cwd: REPO, encoding: "utf8", env: process.env });
      if (r.status === 0 && fs.existsSync(dest) && fs.statSync(dest).size > 0) refs.push(path.basename(dest));
    }
    birthday.references = refs.length;
    birthday.referenceFiles = refs.map((f) => `from-claude/art/${f}`);
  }
  fs.writeFileSync(json, JSON.stringify(brief, null, 2) + "\n");
  const files = brief.pictures.map((p) => `- ${p.kind}: to-claude/art/${p.file} (${p.style.name})`).join("\n");
  reply(`art-brief-${date}`, `ART BRIEF date=${date}\n\nThe picture brief for ${date} is from-claude/art/${date}-brief.json. Draw each picture, save it under the name below, then send a note whose first line is:\n\n    ART DELIVERY date=${date}\n\n${files}\n\nDeliver by ${brief.deliverBy}. Anything not delivered by then is drawn by Workers AI as before.${brief.pictures.some((p) => p.chooseEvent) ? "\n\nThe moment's event is yours to pick this time: see chooseEvent in the JSON, and put the title / year / place lines in your ART DELIVERY note." : ""}${brief.waiting ? `\n\nNot in this brief yet: ${brief.waiting.join(", ")} (the day's event is not picked yet). The 19:00 and 21:00 runs send a new brief for ${date} if it is picked by then; otherwise Workers AI draws it. Draw the rest now.` : ""}`);
  log(`art brief ${date}: written (${brief.pictures.map((p) => p.kind).join(", ")})`);
}

/** Jarvis's pictures for a date: checked, put in R2 for the nightly warm, receipt sent. */
function artDelivery(file, date) {
  const today = chicagoDate(0), tomorrow = chicagoDate(1);
  if (date !== today && date !== tomorrow) {
    reply("art-delivery-date", `FYI: reply to ${file}\n\nART DELIVERY is for today (${today}) or tomorrow (${tomorrow}); ${date} is neither. Nothing was uploaded.`);
    return true;
  }
  const found = fs.existsSync(ART_IN) ? fs.readdirSync(ART_IN).filter((f) => f.startsWith(`${date}-`)) : [];
  const lines = [];
  fs.mkdirSync(path.join(ART_IN, "done"), { recursive: true });
  // When the brief left the event to Jarvis, his note carries what he drew: title / year / place.
  const text = fs.readFileSync(path.join(INBOX, file), "utf8");
  const field = (name) => text.match(new RegExp(`^\\s*${name}\\s*:\\s*(.+)$`, "im"))?.[1].replace(/[\u0000-\u001f]/g, "").trim();
  const chosen = { title: field("title"), year: Number(field("year")), place: field("place") };
  for (const kind of ["moment", "birthday", "skyline"]) {
    const f = found.find((x) => new RegExp(`^${date}-${kind}\\.(png|jpe?g|webp)$`, "i").test(x));
    if (!f) continue;
    const ext = f.split(".").pop().toLowerCase();
    const full = path.join(ART_IN, f);
    const buf = fs.readFileSync(full);
    if (buf.length > ART_MAX_BYTES || !looksLikeImage(buf, ext)) { lines.push(`- ${f}: not uploaded (not a ${ext} image, or over 15 MB)`); continue; }
    const r = spawnSync("npx", ["wrangler", "r2", "object", "put", `${ART_BUCKET}/art/${date}/${kind}`, "--file", full, "--content-type", ART_TYPES[ext], "--remote"], { cwd: REPO, encoding: "utf8", env: process.env });
    if (r.status !== 0) { lines.push(`- ${f}: upload failed, I will look at it`); log(`art upload ${f}: ${(r.stderr ?? "").slice(-300)}`); continue; }
    fs.renameSync(full, path.join(ART_IN, "done", f));
    lines.push(`- ${f}: received`);
    if (kind === "moment" && chosen.title && Number.isInteger(chosen.year) && chosen.year > -5000 && chosen.year <= 2100) {
      const meta = path.join(ART_IN, "done", `${date}-moment.json`);
      fs.writeFileSync(meta, JSON.stringify({ title: chosen.title.slice(0, 80), year: chosen.year, place: (chosen.place ?? "").slice(0, 80) }));
      const m = spawnSync("npx", ["wrangler", "r2", "object", "put", `${ART_BUCKET}/art/${date}/moment.json`, "--file", meta, "--content-type", "application/json", "--remote"], { cwd: REPO, encoding: "utf8", env: process.env });
      lines.push(m.status === 0 ? `  caption: ${chosen.title}, ${chosen.year}, ${chosen.place ?? ""}` : "  caption details could not be stored; the Worker's own caption is used");
    }
  }
  if (lines.length === 0) lines.push(`- no pictures found: name them to-claude/art/${date}-moment.png (or -birthday.png) and to-claude/art/${date}-skyline.png, then send ART DELIVERY again`);
  const when = date === tomorrow ? "Tonight's 23:35 run puts them on the panels for tomorrow." : "The 00:35 run tonight puts them on the panels; until then today's pictures stay.";
  reply(`art-received-${date}`, `FYI: reply to ${file}\n\n${lines.join("\n")}\n\n${when}`);
  log(`art delivery ${date}: ${lines.join(" | ")}`);
  return true;
}

// ---------------------------------------------------------------- weekly chapter

function readSchedule() {
  const r = spawnSync("npx", ["wrangler", "r2", "object", "get", "eink-series/series/schedule.json", "--pipe", "--remote"], { cwd: REPO, encoding: "utf8", env: process.env });
  if (r.status !== 0) throw new Error(`schedule unreadable: ${(r.stderr ?? "").slice(-500)}`);
  return JSON.parse(r.stdout);
}

function cmdWeekly() {
  if (!fs.existsSync(OUTBOX)) { log("mailbox not mounted; weekly skipped"); return; }
  const m = compile();
  // SERIES_TODAY lets an engineer run a chapter early (testing, or getting ahead).
  const due = m.mailbox.dueClaudeChapter(readSchedule(), process.env.SERIES_TODAY ?? stamp().date);
  if (!due) { log("weekly: no Claude chapter due"); return; }
  const pad = (n) => String(n).padStart(2, "0");
  const pkg = path.join(WORK, `s${pad(due.season)}`);
  const out = path.join(WORK, `out-s${pad(due.season)}-c${due.chapter}`);
  log(`weekly: writing season ${due.season} chapter ${due.chapter} (airs ${due.airs})`);
  const ok = claude("chapter.md", { SEASON: String(due.season), CHAPTER: String(due.chapter), AIRS: due.airs, PACKAGE: pkg, OUT: out, REPO }, [
    "Read", "Glob", "Grep", ...writeTo(WORK), `Bash(node:*)`, `Bash(npm --prefix ${REPO} run series -- build:*)`,
    `Bash(python3 -E -P ${REPO}/scripts/series-paint/:*)`,
  ], `chapter-s${due.season}-c${due.chapter}`);
  // Story must never reach the public repo: any change there stops the weekly run cold.
  const dirty = execFileSync("git", ["status", "--porcelain"], { cwd: REPO, encoding: "utf8" }).trim();
  if (dirty) {
    note("Claude's weekly run left changes in the repo; nothing was uploaded. Please open a session before anything is committed.");
    log(`weekly: repo dirty after chapter run, upload skipped:\n${dirty}`);
    return;
  }
  const report = path.join(out, "report.txt");
  const built = ok && fs.existsSync(report) && /pages for chapters/.test(fs.readFileSync(report, "utf8"));
  if (!built) {
    note(`Claude's weekly run could not finish chapter ${due.chapter} of season ${due.season} (airs ${due.airs}). Please open a session.`);
    log("weekly: chapter not built");
    return;
  }
  const u = series(["upload", out]);
  if (!u.ok) { note(`Chapter ${due.chapter} built but the upload failed. Please open a session.`); log(`weekly upload failed: ${u.out.slice(-500)}`); return; }
  if (/proof export FAILED/.test(u.out)) note(`Chapter ${due.chapter} is uploaded but its proof export failed, so Jarvis cannot review it yet. Please open a session.`);
  for (const f of ["bible-notes.md", "handover-notes.md"]) {
    const src = path.join(pkg, f);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(OUTBOX, f === "bible-notes.md" ? "bible-notes-prologue.md" : "SPOILERS-prologue-outline.md"));
  }
  reply(`proof-s${pad(due.season)}-c${pad(due.chapter)}`, `DELIVERY: season ${due.season} chapter ${due.chapter} is uploaded and waiting for your approval.\n\n${exportLines(u.out)}\n\n- Approve with: APPROVE season=${due.season} chapter=${due.chapter} revision=R (R from the export above)\n- Proof page (add your key): ${SITE}/series/proof?season=${due.season}&chapter=${due.chapter}&key=\n- It airs ${due.airs} at 08:00. Please approve at least 8 days before.\n- bible-notes-prologue.md and SPOILERS-prologue-outline.md are updated. Keep the spoiler file away from Thiago.`);
  log(`weekly: chapter ${due.chapter} uploaded`);
}

/** Monday 08:00: the production heartbeat for Jarvis (DECISIONS #76). No AI. */
function cmdHeartbeat() {
  if (!fs.existsSync(OUTBOX)) { log("mailbox not mounted; heartbeat skipped"); return; }
  const m = compile();
  reply("heartbeat", m.mailbox.heartbeat(readSchedule(), process.env.SERIES_TODAY ?? stamp().date));
  log("heartbeat: written");
}

/** An engineer re-runs the independent review of the revision now in R2 (DECISIONS #79). */
function cmdReview() {
  const [season, chapter, zip] = [Number(process.argv[3]), Number(process.argv[4]), process.argv[5]];
  if (!Number.isInteger(season) || !Number.isInteger(chapter) || !zip || !fs.existsSync(zip)) throw new Error("usage: review S C <package.zip>");
  const pad = (n) => String(n).padStart(2, "0");
  const chapterDir = path.join(PROOFS, `s${pad(season)}-c${pad(chapter)}`);
  const revision = fs.readFileSync(path.join(chapterDir, "CURRENT.txt"), "utf8").trim();
  const v = review(season, chapter, revision, path.join(chapterDir, `rev-${revision}`), path.resolve(zip));
  console.log(`season ${season} chapter ${chapter} revision ${revision}: ${v ?? "review did not finish"}`);
  log(`review s${season} c${chapter} ${revision}: ${v ?? "unfinished"}`);
}

// ----------------------------------------------------------------

const cmd = process.argv[2];
if (!["run", "weekly", "heartbeat", "review", "art-brief"].includes(cmd)) { console.error("usage: series-mailbox.mjs run|weekly|heartbeat|review|art-brief"); process.exit(2); }
const LOCK = path.join(WORK, `.mailbox-${cmd}.lock`);
if (!lock(LOCK)) { log(`${cmd}: another run holds the lock; skipped`); process.exit(0); }
try {
  await ({ run: cmdRun, weekly: cmdWeekly, heartbeat: cmdHeartbeat, review: cmdReview, "art-brief": cmdArtBrief })[cmd]();
} catch (err) {
  log(`${cmd} failed: ${err.stack ?? err}`);
  try { note(`The mailbox watcher hit an error (${cmd}). Please open a session.`); } catch { /* mailbox gone */ }
} finally {
  fs.rmSync(LOCK, { force: true });
}
