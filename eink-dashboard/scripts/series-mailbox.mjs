#!/usr/bin/env node
/**
 * The series mailbox watcher (DECISIONS #72). Runs on the Mac mini from launchd:
 *
 *   node scripts/series-mailbox.mjs run      — handle Jarvis's messages in to-claude/
 *   node scripts/series-mailbox.mjs weekly   — write Claude's next chapter when it is due
 *
 * Layer 1 (no AI): APPROVE, DELIVERY and STYLE TEST are handled by this script with the
 * import tool. Layer 2: QUESTION, FYI and anything else go to an unattended Claude run that
 * can READ but may WRITE only into from-claude/ (no shell, no code, no deploys), because
 * a message is text from another AI and must never be able to steer the engineer. The weekly
 * chapter run works from Claude's own notes, may write only in ~/series-work and from-claude/,
 * and cannot upload: this script uploads its build once the import has passed.
 *
 * Nothing here holds story. Logs: ~/Library/Logs/eink-series/.
 */
import { execFileSync, spawnSync } from "node:child_process";
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
      const lines = msg.chapters.map((c) => {
        const r = series(["approve", "--season", String(msg.season), "--chapter", String(c)]);
        return `- Season ${msg.season} chapter ${c}: ${r.ok ? "approved; it airs on its Monday at 08:00." : `NOT approved: ${r.out.split("\n").pop()}`}`;
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
      reply(u.ok ? "delivery-uploaded" : "delivery-upload-failed", u.ok
        ? `FYI: reply to ${file}\n\n${msg.package} passed every check and is uploaded. ${u.out.split("\n").filter((l) => /uploaded|not replaced/.test(l)).join(" ")}\n\nProof pages (add your key):\n${links}\n\nSend APPROVE season=${meta.season.number} chapter=N when you are happy.`
        : `FYI: reply to ${file}\n\n${msg.package} passed the checks but the upload failed. Nothing changed on air. Thiago needs to ask Claude to look:\n\n${u.out.slice(-2000)}`);
      if (!u.ok) note(`Upload of ${msg.package} failed; see ~/Library/Logs/eink-series/mailbox.log`);
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
    case "INVALID":
      reply("format-problem", `FYI: reply to ${file}\n\n${msg.reason}. Nothing was done. The first line formats are:\n\n    APPROVE season=0 chapter=3\n    DELIVERY package=season-01-chapter-01.zip [replace-approved]\n    STYLE TEST folder=openai-1\n    QUESTION | FYI`);
      return true;
    default: {
      const read = ["Read", "Glob", "Grep"];
      return claude("message.md", { MESSAGE: path.join(INBOX, file), MAILBOX, OUTBOX, STAMP: stamp().file }, [...read, ...writeTo(OUTBOX)], `message-${file.replace(/\.md$/, "")}`);
    }
  }
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
  for (const f of ["bible-notes.md", "handover-notes.md"]) {
    const src = path.join(pkg, f);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(OUTBOX, f === "bible-notes.md" ? "bible-notes-prologue.md" : "SPOILERS-prologue-outline.md"));
  }
  reply(`proof-s${pad(due.season)}-c${pad(due.chapter)}`, `DELIVERY: season ${due.season} chapter ${due.chapter} is uploaded and waiting for your approval.\n\n- Proof page (add your key): ${SITE}/series/proof?season=${due.season}&chapter=${due.chapter}&key=\n- It airs ${due.airs} at 08:00. Please approve at least 8 days before.\n- bible-notes-prologue.md and SPOILERS-prologue-outline.md are updated. Keep the spoiler file away from Thiago.`);
  log(`weekly: chapter ${due.chapter} uploaded`);
}

/** Monday 08:00: the production heartbeat for Jarvis (DECISIONS #76). No AI. */
function cmdHeartbeat() {
  if (!fs.existsSync(OUTBOX)) { log("mailbox not mounted; heartbeat skipped"); return; }
  const m = compile();
  reply("heartbeat", m.mailbox.heartbeat(readSchedule(), process.env.SERIES_TODAY ?? stamp().date));
  log("heartbeat: written");
}

// ----------------------------------------------------------------

const cmd = process.argv[2];
if (!["run", "weekly", "heartbeat"].includes(cmd)) { console.error("usage: series-mailbox.mjs run|weekly|heartbeat"); process.exit(2); }
const LOCK = path.join(WORK, `.mailbox-${cmd}.lock`);
if (!lock(LOCK)) { log(`${cmd}: another run holds the lock; skipped`); process.exit(0); }
try {
  ({ run: cmdRun, weekly: cmdWeekly, heartbeat: cmdHeartbeat })[cmd]();
} catch (err) {
  log(`${cmd} failed: ${err.stack ?? err}`);
  try { note(`The mailbox watcher hit an error (${cmd}). Please open a session.`); } catch { /* mailbox gone */ }
} finally {
  fs.rmSync(LOCK, { force: true });
}
