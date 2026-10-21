#!/usr/bin/env node
/**
 * The desk serial's jobs on the Mac mini (docs/superpowers/specs/2026-10-21-desk-serial-design.md):
 *
 *   node scripts/serial-mailbox.mjs run            — build, review and publish beats as Jarvis's pictures land (every 2 min)
 *   node scripts/serial-mailbox.mjs weekly         — write the days still missing up to the Sunday of the week 12+ days ahead (daily 09:00)
 *   node scripts/serial-mailbox.mjs check <day.json>  — fit, glyph and no-repeat check of one day file (the writer runs it)
 *
 * The story lives in ~/desk-serial-work/ and Jarvis's spoiler folder, never in the repo, the shared
 * mailbox's open notes or the logs: everything printed or logged here is counts, ids and reasons.
 */

import { execFileSync, spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { buildBeat, compile, put } from "./serial-build.mjs";

const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const HOME = os.homedir();
const WORK = process.env.SERIAL_WORK ?? path.join(HOME, "desk-serial-work");
const DROPBOX = "/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox";
const MAILBOX = path.join(DROPBOX, "00_SHARED_INBOX", "series");
const INBOX = path.join(MAILBOX, "to-claude", "serial");
const OUTBOX = path.join(MAILBOX, "from-claude");
const SPOILERS = path.join(DROPBOX, "00_JARVIS_SPOILERS", "desk-serial");
const STATE = path.join(WORK, "state.json");
const LOGS = path.join(HOME, "Library", "Logs", "eink-series");
const KV_NAMESPACE = "de97776d35af4df08b13fd2158acebdc";
const ACCOUNT = "f22a506dedde3bb3837157cd47d5fe5c";
const REVIEW_TRIES = 3;

const log = (s) => {
  fs.mkdirSync(LOGS, { recursive: true });
  fs.appendFileSync(path.join(LOGS, "serial.log"), `${new Date().toISOString()} ${s}\n`);
};

function chicago(now = new Date()) {
  const p = Object.fromEntries(new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(now).map((x) => [x.type, x.value]));
  return { date: `${p.year}-${p.month}-${p.day}`, stamp: `${p.year}-${p.month}-${p.day}-${p.hour}${p.minute}` };
}

const shift = (date, n) => {
  const d = new Date(`${date}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
};

/** Every day file the writer has made, by date. */
export function dayFiles(work = WORK) {
  const out = {};
  for (const season of fs.existsSync(work) ? fs.readdirSync(work).filter((d) => /^s\d+$/.test(d)) : []) {
    const dir = path.join(work, season, "days");
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir).filter((f) => /^\d{4}-\d{2}-\d{2}\.json$/.test(f))) out[f.slice(0, 10)] = path.join(dir, f);
  }
  return out;
}

/**
 * The text a beat airs with. A beat marked `live: "match"` has a `text_done` variant for after the
 * day's Flamengo match ends, with {{us}}, {{them}} and {{opponent}} filled from the board.
 */
export function beatText(beat, match) {
  if (beat.live === "match" && match?.done && Array.isArray(beat.text_done)) {
    const fill = (l) => l.replaceAll("{{us}}", String(match.us)).replaceAll("{{them}}", String(match.them)).replaceAll("{{opponent}}", match.opponent);
    return { text: beat.text_done.map(fill), variant: `done:${match.us}-${match.them}` };
  }
  return { text: beat.text, variant: "default" };
}

/** Whether a beat needs (re)building: a new picture, new live text, or a review still owed. */
export function needsBuild(prev, sha, variant) {
  if (!prev || prev.sha !== sha) return true;
  if (prev.verdict === "PASS") return prev.variant !== variant;
  return prev.verdict === "UNREVIEWED" && (prev.tries ?? 0) < REVIEW_TRIES;
}

/** Every string literal in the engine's story bank, to keep the serial from repeating it. */
function engineLines() {
  const src = fs.readFileSync(path.join(REPO, "src", "castaway-story.ts"), "utf8");
  return new Set([...src.matchAll(/"((?:[^"\\]|\\.){12,})"/g)].map((m) => m[1].trim().toLowerCase()));
}

/** Problems with one day file: fit, glyphs, and lines repeated from other days or the engine. */
export function checkDay(lib, day, otherLines, engine) {
  const problems = [];
  const beats = day.beats.slice().sort((a, b) => a.start.localeCompare(b.start));
  const seen = new Set();
  const widest = { us: "99", them: "99", opponent: "WWW" };
  for (const [i, b] of beats.entries()) {
    const log = beats.slice(0, i).slice(-2).reverse().map((e) => ({ time: e.start, text: e.log }));
    const texts = [b.text, ...(b.live === "match" ? [beatText(b, { done: true, ...widest }).text] : [])];
    for (const t of texts) {
      const misfit = lib.page.storyTextFits(b.title, t.join(" "), log);
      if (misfit) problems.push(`${b.id}: ${misfit}`);
    }
    for (const ch of [b.title, b.log, ...b.text, ...(b.text_done ?? [])].join("")) {
      if (ch !== " " && !lib.fonts.FACES.plex13r.glyphs[ch]) problems.push(`${b.id}: no glyph for character code ${ch.charCodeAt(0)}`);
    }
    if (!b.scene?.focus || !(b.scene.focus.fx >= 0 && b.scene.focus.fx <= 1 && b.scene.focus.fy >= 0 && b.scene.focus.fy <= 1)) problems.push(`${b.id}: focus missing or outside 0..1`);
    for (const line of [b.text.join(" "), b.log]) {
      const k = line.trim().toLowerCase();
      if (otherLines.has(k) || engine.has(k) || seen.has(k)) problems.push(`${b.id}: a line repeats an earlier one`);
      seen.add(k);
    }
  }
  return problems;
}

function linesOfOtherDays(date) {
  const lines = new Set();
  for (const [d, f] of Object.entries(dayFiles())) {
    if (d === date) continue;
    for (const b of JSON.parse(fs.readFileSync(f, "utf8")).beats) for (const l of [b.text.join(" "), b.log]) lines.add(l.trim().toLowerCase());
  }
  return lines;
}

function kvGet(key) {
  const run = () => execFileSync("npx", ["wrangler", "kv", "key", "get", "--remote", "--namespace-id", KV_NAMESPACE, key],
    { cwd: REPO, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], env: { ...process.env, CLOUDFLARE_ACCOUNT_ID: ACCOUNT } });
  try { return run(); } catch {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10_000); // a fresh wrangler token is sometimes refused for a moment
    try { return run(); } catch { return null; }
  }
}

/** Claude's independent look at a built beat (design §6.6): PASS, REVISE with a reason, or null. */
function review(beat, pagePng, picture) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "serial-review-"));
  fs.writeFileSync(path.join(dir, "beat.json"), JSON.stringify(beat, null, 2));
  let prompt = fs.readFileSync(path.join(REPO, "scripts", "series-agent", "serial-review.md"), "utf8");
  for (const [k, v] of Object.entries({ BEAT: path.join(dir, "beat.json"), PAGE: pagePng, PICTURE: picture })) prompt = prompt.replaceAll(`{{${k}}}`, v);
  const r = spawnSync("claude", ["-p", prompt, "--model", "opus", "--permission-mode", "dontAsk", "--allowedTools", "Read", "--add-dir", dir, "--add-dir", path.dirname(picture), "--add-dir", path.dirname(pagePng)],
    { cwd: dir, encoding: "utf8", timeout: 10 * 60 * 1000, maxBuffer: 16 * 1024 * 1024 });
  fs.rmSync(dir, { recursive: true, force: true });
  const m = /VERDICT:\s*(PASS|REVISE)\s*:?\s*(.*)/.exec(r.stdout ?? "");
  return m ? { verdict: m[1], reason: m[2].trim() } : null;
}

function reply(slug, body) {
  fs.writeFileSync(path.join(OUTBOX, `${chicago().stamp}-${slug}.md`), body.trim() + "\n");
}

async function cmdRun() {
  if (!fs.existsSync(INBOX)) { log("mailbox not mounted; skipped"); return; }
  const today = chicago().date;
  const files = dayFiles();
  const dates = Object.keys(files).filter((d) => d >= shift(today, -1) && d <= shift(today, 14)).sort();
  const state = fs.existsSync(STATE) ? JSON.parse(fs.readFileSync(STATE, "utf8")) : {};
  const days = Object.fromEntries(dates.map((d) => [d, JSON.parse(fs.readFileSync(files[d], "utf8"))]));
  const sha = (f) => crypto.createHash("sha256").update(fs.readFileSync(f)).digest("hex");
  const picture = (d, b) => path.join(INBOX, d, `${b.id}.png`);

  // Only today and tomorrow can have a match result worth waiting for.
  const live = dates.some((d) => d <= shift(today, 1) && days[d].beats.some((b) => b.live === "match"));
  const board = live ? kvGet("flamengo:v3:state") : null;
  const work = [];
  for (const d of dates) {
    for (const b of days[d].beats) if (fs.existsSync(picture(d, b))) work.push({ d, b, sha: sha(picture(d, b)) });
  }
  // A day not yet published (today or tomorrow) goes up even with no beats, so the alert sees what is planned.
  const unpublished = dates.filter((d) => d <= shift(today, 1) && state[`day:${d}`] === undefined);
  // On a match day the live text can change at any run, so the build runs every time.
  const anything = live || unpublished.length > 0 || work.some((w) => needsBuild(state[w.b.id], w.sha, "default"));
  if (!anything) return;

  const lib = compile();
  const pages = path.join(WORK, "pages");
  fs.mkdirSync(pages, { recursive: true });
  const dirtyDays = new Set(unpublished);
  const published = [];
  for (const { d, b, sha: s } of work) {
    const match = board ? lib.castaway.matchOn(JSON.parse(board), d) : null;
    const { text, variant } = beatText(b, match);
    const prev = state[b.id];
    if (!needsBuild(prev, s, variant)) continue;
    const beats = days[d].beats.slice().sort((x, y) => x.start.localeCompare(y.start));
    const earlier = beats.slice(0, beats.findIndex((x) => x.id === b.id));
    const r = await buildBeat(lib, days[d], { ...b, text }, earlier, new Uint8Array(fs.readFileSync(picture(d, b))));
    if (r.error) {
      if (prev?.sha !== s) {
        log(`${b.id}: not built (${r.error})`);
        fs.mkdirSync(path.join(SPOILERS, d), { recursive: true });
        fs.writeFileSync(path.join(SPOILERS, d, `review-${b.id}.md`), `Beat ${b.id} could not be built: ${r.error}. Please send a new picture with the same name.\n`);
        reply(`serial-revise-${b.id}`, `SERIAL REVISE date=${d} beat=${b.id}\n\nThis beat's picture could not be used (${r.error}). Please replace ${b.id}.png in to-claude/serial/${d}/.`);
      }
      state[b.id] = { sha: s, verdict: "ERROR", reason: r.error };
      continue;
    }
    const html = path.join(pages, `${b.id}.html`);
    fs.writeFileSync(html, r.html);
    const png = path.join(pages, `${b.id}.png`);
    fs.writeFileSync(png, Buffer.from(r.html.split("base64,")[1].split('"')[0], "base64"));
    // A new live text on an approved picture needs no second review: only the score changed.
    let verdict = prev?.sha === s && prev.verdict === "PASS" ? { verdict: "PASS", reason: "live text" } : review(b, png, picture(d, b));
    if (!verdict) {
      const tries = (prev?.sha === s ? prev.tries ?? 0 : 0) + 1;
      state[b.id] = { sha: s, verdict: "UNREVIEWED", tries, variant };
      log(`${b.id}: review did not finish (try ${tries})`);
      if (tries >= REVIEW_TRIES) fs.appendFileSync(path.join(OUTBOX, "NEEDS-THIAGO.md"), `- ${chicago().stamp}: the desk serial's review could not run for beat ${b.id}; it is not airing. Please open a session.\n`);
      continue;
    }
    state[b.id] = { sha: s, verdict: verdict.verdict, variant, reason: verdict.verdict === "PASS" ? undefined : verdict.reason };
    log(`${b.id}: built, ${verdict.verdict}`);
    if (verdict.verdict === "PASS") {
      put(lib.serial.serialPageKey(b.id), html, "text/html; charset=utf-8");
      dirtyDays.add(d);
      if (prev?.sha !== s) published.push(`${b.id} (airs ${d} ${b.start})`);
    } else {
      fs.mkdirSync(path.join(SPOILERS, d), { recursive: true });
      fs.writeFileSync(path.join(SPOILERS, d, `review-${b.id}.md`), `Claude's review of beat ${b.id}: REVISE.\n\n${verdict.reason}\n\nPlease send a new picture with the same name.\n`);
      reply(`serial-revise-${b.id}`, `SERIAL REVISE date=${d} beat=${b.id}\n\nClaude's review asks for a new picture for this beat; the reason is in your spoiler folder (desk-serial/${d}/review-${b.id}.md). Replace ${b.id}.png in to-claude/serial/${d}/. Until then the previous beat keeps showing.`);
      dirtyDays.add(d);
    }
  }
  for (const d of dirtyDays) {
    const passed = days[d].beats.filter((b) => state[b.id]?.verdict === "PASS").map((b) => ({ id: b.id, start: b.start, key: lib.serial.serialPageKey(b.id) }));
    const body = { date: d, day: days[d].day, planned: days[d].beats.length, beats: passed, ...(days[d].same_day ? { same_day: true } : {}), ...(days[d].season_end ? { season_end: true } : {}) };
    const file = path.join(WORK, "uploaded", `${d}.json`);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, JSON.stringify(body));
    put(lib.serial.serialDayKey(d), file, "application/json");
    state[`day:${d}`] = passed.length;
    log(`${d}: ${passed.length} of ${body.planned} beats published`);
  }
  fs.writeFileSync(STATE, JSON.stringify(state, null, 1));
  if (published.length) reply(`serial-received-${published.length}`, `SERIAL RECEIVED\n\nBuilt, reviewed and published:\n${published.map((p) => `- ${p}`).join("\n")}`);
}

/** The last date the writer should have covered: the Sunday of the week starting 12 or more days ahead. */
export function writeThrough(today) {
  const start = new Date(`${shift(today, 12)}T12:00:00Z`);
  const toMonday = (8 - start.getUTCDay()) % 7; // 0 when it is already Monday
  return shift(start.toISOString().slice(0, 10), toMonday + 6);
}

function cmdWeekly() {
  if (!fs.existsSync(OUTBOX)) { log("mailbox not mounted; weekly skipped"); return; }
  const today = chicago().date, through = writeThrough(today);
  const have = dayFiles();
  const missing = [];
  for (let d = shift(today, 1); d <= through; d = shift(d, 1)) if (!have[d]) missing.push(d);
  if (missing.length === 0) { log(`weekly: written through ${through}`); return; }
  log(`weekly: writing ${missing.length} days (${missing[0]}..${missing.at(-1)})`);
  let prompt = fs.readFileSync(path.join(REPO, "scripts", "series-agent", "serial-writer.md"), "utf8");
  for (const [k, v] of Object.entries({ FROM: missing[0], THROUGH: missing.at(-1), TODAY: today, WORK, SPOILERS, OUTBOX, INBOX, MAILBOX, REPO })) prompt = prompt.replaceAll(`{{${k}}}`, v);
  const r = spawnSync("claude", ["-p", prompt, "--model", "opus", "--permission-mode", "dontAsk",
    "--allowedTools", ["Read", "Glob", "Grep", "WebSearch", "WebFetch", `Write(/${WORK}/**)`, `Edit(/${WORK}/**)`, `Write(/${SPOILERS}/**)`, `Write(/${OUTBOX}/**)`,
      `Bash(node ${REPO}/scripts/serial-mailbox.mjs check:*)`, `Bash(mkdir:*)`].join(" "),
    "--add-dir", WORK, "--add-dir", SPOILERS, "--add-dir", MAILBOX],
  { cwd: REPO, encoding: "utf8", timeout: 90 * 60 * 1000, maxBuffer: 64 * 1024 * 1024 });
  fs.writeFileSync(path.join(LOGS, `${chicago().stamp}-serial-weekly.log`), `${r.stdout ?? ""}\n---\n${r.stderr ?? ""}`);
  const dirty = execFileSync("git", ["status", "--porcelain"], { cwd: REPO, encoding: "utf8" }).trim();
  if (dirty) {
    fs.appendFileSync(path.join(OUTBOX, "NEEDS-THIAGO.md"), `- ${chicago().stamp}: the desk serial's writer left changes in the repo. Please open a session before anything is committed.\n`);
    log("weekly: repo dirty after the writer ran");
  }
  const still = missing.filter((d) => !dayFiles()[d]);
  log(`weekly: ${missing.length - still.length} of ${missing.length} days written`);
  if (still.length) fs.appendFileSync(path.join(OUTBOX, "NEEDS-THIAGO.md"), `- ${chicago().stamp}: the desk serial's writer could not write ${still.length} day(s) from ${still[0]}. Please open a session.\n`);
}

function cmdCheck() {
  const file = process.argv[3];
  const day = JSON.parse(fs.readFileSync(file, "utf8"));
  const problems = checkDay(compile(), day, linesOfOtherDays(day.date), engineLines());
  console.log(problems.length ? problems.join("\n") : `${day.date}: ${day.beats.length} beats OK`);
  process.exit(problems.length ? 1 : 0);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const cmd = process.argv[2];
  const run = { run: cmdRun, weekly: cmdWeekly, check: cmdCheck }[cmd];
  if (!run) { console.error("usage: serial-mailbox.mjs run | weekly | check <day.json>"); process.exit(2); }
  const lockFile = path.join(WORK, `.serial-${cmd}.lock`);
  if (cmd !== "check") {
    try { if (Date.now() - fs.statSync(lockFile).mtimeMs < 2 * 3600 * 1000) process.exit(0); } catch { /* no lock */ }
    fs.mkdirSync(WORK, { recursive: true });
    fs.writeFileSync(lockFile, String(process.pid));
  }
  try { await run(); } catch (err) { log(`${cmd} failed: ${err.stack ?? err}`); process.exitCode = 1; } finally { if (cmd !== "check") fs.rmSync(lockFile, { force: true }); }
}
