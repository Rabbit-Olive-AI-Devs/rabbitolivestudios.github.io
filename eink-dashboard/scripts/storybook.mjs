#!/usr/bin/env node
/**
 * Keeps one frame per story moment of the desk story page for a story book (DECISIONS #95).
 * Runs on the Mac mini:
 *
 *   node scripts/storybook.mjs live                 — if a new story moment has started, fetch the
 *                                                     page once and save its picture
 *   node scripts/storybook.mjs backfill [FROM] [TO] — render past moments on this Mac with the
 *                                                     Worker's code, at 10-minute steps (Chicago
 *                                                     dates, default 2026-10-17 to now)
 *   node scripts/storybook.mjs epub                 — rebuild the ebook (scripts/storybook-epub.py)
 *   node scripts/storybook.mjs annotate [DATE...]   — add the moment key to every row of the
 *                                                     saved days' indexes (backup: index.jsonl.bak)
 *
 * Which moment is showing comes from the Worker's own story engine, compiled here as
 * `npm run test:utils` does, with the cached context that moves moments (sunrise, sunset, a match
 * day) read from production KV: reads only, never the Worker. So the live job checks every 10
 * minutes for free and fetches the page only when a new moment starts. If the engine cannot say,
 * it fetches once and saves only when the picture below the top bar changed.
 *
 * Frames go to Dropbox, `<BOOK>/<YYYY-MM-DD>/<HHMM>.png` (Chicago time), with one `index.jsonl`
 * per day: {time, at, day, file, sha256, bytes, source, moment, moment_start}. `moment` is an
 * opaque key (date and the moment's start time), `moment_start` its start (ISO). A failed fetch
 * or the text fallback page is logged and skipped, never saved. No AI, no KV writes.
 *
 * Nothing here holds story, and the log says only times, sizes and hashes.
 * Log: ~/Library/Logs/eink-series/storybook.log. Started by launchd through materialize-exec.py
 * (DECISIONS #72), since Dropbox may have made the index online-only.
 */
import { execFileSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const KV_NAMESPACE = "de97776d35af4df08b13fd2158acebdc";
const ACCOUNT = "f22a506dedde3bb3837157cd47d5fe5c";

const DROPBOX = process.env.STORYBOOK_DROPBOX ?? "/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox";
const BOOK = path.join(DROPBOX, "00_STORYBOOK_SURPRISE_do_not_open");
const PAGE = process.env.STORYBOOK_URL ?? "https://eink-dashboard.thiago-oliveira77.workers.dev/desk/castaway";
const LOGS = path.join(os.homedir(), "Library", "Logs", "eink-series");
export const EPOCH = "2026-10-17"; // the page's day 1 (CASTAWAY_EPOCH in src/castaway.ts)
const STEP_MS = 10 * 60 * 1000;   // the desk panel's refresh interval
const TOP_BAR_BOTTOM = 36;        // rows above this are the shared top bar, with its clock

function log(s) {
  fs.mkdirSync(LOGS, { recursive: true });
  fs.appendFileSync(path.join(LOGS, "storybook.log"), `${new Date().toISOString()} ${s}\n`);
}

/** The PNG inlined in a page, or null (the text fallback page has none). */
export function extractPng(html) {
  const m = /src="data:image\/png;base64,([A-Za-z0-9+/=]+)"/.exec(html);
  if (!m) return null;
  const png = Buffer.from(m[1], "base64");
  const SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  return png.length > 8 && SIG.every((b, i) => png[i] === b) ? png : null;
}

/** Chicago date, HH:MM and the file stem HHMM for an instant. */
export function chicagoStamp(at) {
  const p = Object.fromEntries(new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(at).map((x) => [x.type, x.value]));
  const hh = String(Number(p.hour) % 24).padStart(2, "0");
  return { date: `${p.year}-${p.month}-${p.day}`, time: `${hh}:${p.minute}`, stem: `${hh}${p.minute}` };
}

/** The page's day number for a Chicago date (same arithmetic as dayNumber in src/castaway.ts). */
export function dayNumber(date) {
  const ms = (d) => Date.UTC(Number(d.slice(0, 4)), Number(d.slice(5, 7)) - 1, Number(d.slice(8, 10)));
  return Math.round((ms(date) - ms(EPOCH)) / 86_400_000) + 1;
}

/** A free file name in a day folder: HHMM.png, or HHMM-2.png when the hour repeats (DST). */
export function frameName(stem, taken) {
  if (!taken.has(`${stem}.png`)) return `${stem}.png`;
  for (let n = 2; ; n++) if (!taken.has(`${stem}-${n}.png`)) return `${stem}-${n}.png`;
}

/** Instants from FROM 00:00 to TO 23:50 (Chicago dates), every 10 minutes, none after `until`. */
export function backfillInstants(from, to, until) {
  const start = chicagoMidnight(from), end = Math.min(chicagoMidnight(nextDate(to)), until.getTime() + 1);
  const out = [];
  for (let t = start; t < end; t += STEP_MS) out.push(new Date(t));
  return out;
}

function nextDate(d) {
  const t = new Date(Date.UTC(Number(d.slice(0, 4)), Number(d.slice(5, 7)) - 1, Number(d.slice(8, 10)) + 1));
  return t.toISOString().slice(0, 10);
}

/** The UTC instant of 00:00 in Chicago on a date (CST or CDT). */
function chicagoMidnight(date) {
  for (const offsetH of [5, 6]) {
    const t = Date.UTC(Number(date.slice(0, 4)), Number(date.slice(5, 7)) - 1, Number(date.slice(8, 10)), offsetH);
    const s = chicagoStamp(new Date(t));
    if (s.date === date && s.stem === "0000") return t;
  }
  throw new Error(`no Chicago midnight for ${date}`);
}

async function fetchPage(url) {
  for (let attempt = 1; ; attempt++) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(30_000), headers: { "user-agent": "eink-storybook" } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.text();
    } catch (err) {
      if (attempt >= 2) throw err;
      log(`fetch failed (${err.message}), retrying`);
      await new Promise((r) => setTimeout(r, 10_000));
    }
  }
}

function readIndex(dir) {
  try {
    return fs.readFileSync(path.join(dir, "index.jsonl"), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
  } catch (err) {
    if (err.code === "ENOENT") return [];
    throw err;
  }
}

/** The live page right now, or null (logged) when the fetch fails or it is the text fallback. */
async function fetchLive(at) {
  let html;
  try {
    html = await fetchPage(PAGE);
  } catch (err) {
    log(`live ${at.toISOString()}: fetch failed (${err.message}), skipped`);
    return null;
  }
  const png = extractPng(html);
  if (!png) log(`live ${at.toISOString()}: no picture in the page (fallback), skipped`);
  return png;
}

/** The rows below the top bar of a 1-bit or 8-bit PNG as written by src/png.ts, or null. */
export function belowTopBar(png) {
  let pos = 8, ihdr = null;
  const idat = [];
  while (pos + 8 <= png.length) {
    const len = png.readUInt32BE(pos), type = png.toString("latin1", pos + 4, pos + 8);
    const data = png.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") ihdr = data;
    if (type === "IDAT") idat.push(data);
    pos += 12 + len;
  }
  if (!ihdr || ihdr[12] !== 0) return null;   // interlaced: not what the page writes
  const width = ihdr.readUInt32BE(0), height = ihdr.readUInt32BE(4);
  const channels = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[ihdr[9]] ?? 0;
  const stride = 1 + Math.ceil((width * ihdr[8] * channels) / 8);
  let raw;
  try { raw = zlib.inflateSync(Buffer.concat(idat)); } catch { return null; }
  if (!channels || raw.length < stride * height) return null;
  // src/png.ts writes every row unfiltered; any other filter would need unfiltering to compare.
  for (let y = TOP_BAR_BOTTOM; y < height; y++) if (raw[y * stride] !== 0) return null;
  return raw.subarray(TOP_BAR_BOTTOM * stride, stride * height);
}

/** Whether the picture below the top bar differs (or cannot be compared). */
export function pictureChanged(a, b) {
  const x = a && belowTopBar(a), y = b && belowTopBar(b);
  return !x || !y || !x.equals(y);
}

/**
 * The Worker's story engine and page drawing, compiled on this Mac (like `npm run test:utils`), with
 * the cached context the page reads (weather, planet, Flamengo board) from production KV as it is
 * now: reads only, once per key, never the Worker. A backfill renders here because an earlier
 * version fetched `?at=` from the live page and its burst used up the free plan's CPU allowance,
 * which briefly failed every desk page with 1102. `context(at)` throws when a key that moves moments
 * (weather, Flamengo board) could not be read; the planet only changes words, so it may be missing.
 */
function engine() {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "storybook-"));
  let castaway, page, hud, serial;
  try {
    execFileSync("npx", ["tsc", "-p", "tsconfig.test.json", "--outDir", out], { cwd: REPO, stdio: "ignore" });
    castaway = require(path.join(out, "src/castaway.js"));
    page = require(path.join(out, "src/hud/castaway-page.js"));
    hud = require(path.join(out, "src/hud/page.js"));
    serial = require(path.join(out, "src/serial.js"));
  } finally {
    fs.rmSync(out, { recursive: true, force: true });
  }
  const kv = new Map(), failed = new Set();
  const read = (key) => execFileSync("npx", ["wrangler", "kv", "key", "get", "--remote", "--namespace-id", KV_NAMESPACE, key],
    { cwd: REPO, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, CLOUDFLARE_ACCOUNT_ID: ACCOUNT } });
  const env = { CACHE: { get: async (key, type) => {
    if (!kv.has(key)) {
      let raw = null;
      try {
        raw = read(key);
      } catch (err) {
        // Wrangler refreshes its hourly sign-in token on the first call after it expires, and the
        // new token is sometimes refused with a 401 for a moment; a retry 10 s later gets through.
        log(`kv get ${key} failed, retrying: ${String(err.stderr ?? "").replace(/\x1b\[[0-9;]*m/g, "").match(/\d{3}: \w+/)?.[0] ?? "error"}`);
        await new Promise((r) => setTimeout(r, 10_000));
        try { raw = read(key); } catch { failed.add(key); }
      }
      kv.set(key, raw);
    }
    const raw = kv.get(key);
    return raw && type === "json" ? JSON.parse(raw) : raw;
  } } };
  const context = async (at, strict) => {
    const ctx = await castaway.loadCastawayContext(env, at);
    const missing = [...failed].filter((k) => !k.startsWith("planet:"));
    if (strict && missing.length) throw new Error(`${missing.length} cache read(s) failed`);
    return ctx;
  };
  return {
    /** The moment showing at `at`: {key, start}. A serial beat, when one is published, else the engine's. */
    moment: async (at, strict = true) => serialMoment(serial, castaway, at) ?? castaway.castawayMoment(at, await context(at, strict)),
    /** The page's picture at `at` and its moment, or null (logged) when drawing fails. */
    render: async (at) => {
      try {
        const ctx = await context(at, false);
        const png = Buffer.from(await hud.canvasToPng(page.drawCastawayPage(castaway.castawayView(at, ctx))));
        return { png, moment: castaway.castawayMoment(at, ctx) };
      } catch (err) {
        log(`backfill ${at.toISOString()}: render failed (${err.message}), skipped`);
        return { png: null, moment: null };
      }
    },
  };
}

/**
 * The serial beat showing at `at`, from the Mac mini's copy of each published day file
 * (scripts/serial-mailbox.mjs writes ~/desk-serial-work/uploaded/<date>.json): {key, start} or null.
 */
export function serialMoment(serial, castaway, at, uploaded = path.join(os.homedir(), "desk-serial-work", "uploaded")) {
  const { date, min } = castaway.chicagoParts(at);
  const read = (d) => { try { return JSON.parse(fs.readFileSync(path.join(uploaded, `${d}.json`), "utf8")); } catch { return null; } };
  const prev = new Date(at.getTime() - 24 * 3600 * 1000);
  const beat = serial.currentBeat(read(date), read(castaway.chicagoParts(prev).date), min);
  if (!beat) return null;
  const day = beat.id.slice(0, 10);
  // The beat's Chicago wall time as an instant: read the wall clock Chicago shows at that UTC time,
  // and shift by the difference (the zone offset, which is the same an hour either side of the start).
  const guess = Date.parse(`${day}T${beat.start}:00Z`);
  const p = Object.fromEntries(new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(new Date(guess)).map((x) => [x.type, x.value]));
  const wall = Date.UTC(Number(p.year), Number(p.month) - 1, Number(p.day), Number(p.hour), Number(p.minute));
  return { key: `serial:${beat.id}`, start: new Date(guess + (guess - wall)) };
}

/** Index rows with the moment key of each row's time added (`momentAt(at)` gives {key, start}). */
export async function annotateRows(rows, momentAt) {
  const out = [];
  for (const r of rows) {
    const m = await momentAt(new Date(r.at));
    out.push({ ...r, moment: m.key, moment_start: m.start.toISOString() });
  }
  return out;
}

/** Whether the live job should fetch: no frame yet for this moment (or the moment is unknown). */
export function needsFrame(index, moment) {
  return !moment || !index.some((e) => e.moment === moment.key);
}

/**
 * Save one frame for instant `at` unless that day already holds it (same SHA-256, same moment, or
 * for a backfill a frame already kept at that minute). Returns "saved" | "same" | "skip".
 */
function keep(at, source, png, moment) {
  if (!png) return "skip";
  const { date, time, stem } = chicagoStamp(at);
  const dir = path.join(BOOK, date);
  fs.mkdirSync(dir, { recursive: true });
  const index = readIndex(dir);
  const sha256 = crypto.createHash("sha256").update(png).digest("hex");
  if (!needsFrame(index, moment) || index.some((e) => e.sha256 === sha256 || (source === "backfill" && e.time === time))) return "same";
  const file = frameName(stem, new Set(index.map((e) => e.file)));
  fs.writeFileSync(path.join(dir, file), png);
  const entry = { time, at: at.toISOString(), day: dayNumber(date), file, sha256, bytes: png.length, source };
  if (moment) Object.assign(entry, { moment: moment.key, moment_start: moment.start.toISOString() });
  fs.appendFileSync(path.join(dir, "index.jsonl"), JSON.stringify(entry) + "\n");
  return "saved";
}

async function main() {
  const [cmd, ...args] = process.argv.slice(2);
  if (!fs.existsSync(DROPBOX)) {
    log(`Dropbox folder missing (${DROPBOX}); nothing saved`);
    process.exit(1);
  }
  if (cmd === "live") {
    const at = new Date();
    const { date, time } = chicagoStamp(at);
    let moment = null;
    try {
      moment = await engine().moment(at);
    } catch (err) {
      log(`live ${date} ${time}: moment unknown (${err.message}), comparing pictures instead`);
    }
    const dir = path.join(BOOK, date);
    const index = readIndex(dir);
    if (!needsFrame(index, moment)) {
      log(`live ${date} ${time}: same moment, skipped`);
      return;
    }
    const png = await fetchLive(at);
    const last = index.slice().sort((a, b) => a.at.localeCompare(b.at)).at(-1);
    const unchanged = png && !moment && last && !pictureChanged(fs.readFileSync(path.join(dir, last.file)), png);
    const r = unchanged ? "same" : keep(at, "live", png, moment);
    log(`live ${date} ${time}: ${r === "same" ? "no change" : r === "saved" && moment ? "new moment, saved" : r}`);
    process.exit(r === "skip" ? 1 : 0);
  }
  if (cmd === "epub") {
    // Python started from here, not by launchd: macOS grants this job's Node access to the Dropbox
    // volume, and a child inherits it; the system Python started directly is refused.
    execFileSync("/usr/bin/python3", ["-I", path.join(REPO, "scripts", "storybook-epub.py")], { stdio: "inherit" });
    return;
  }
  if (cmd === "annotate") {
    const days = args.length ? args : fs.readdirSync(BOOK).filter((d) => /^\d{4}-\d{2}-\d{2}$/.test(d)).sort();
    const { moment } = engine();
    for (const d of days) {
      const file = path.join(BOOK, d, "index.jsonl");
      const rows = readIndex(path.join(BOOK, d));
      if (!rows.length) continue;
      fs.copyFileSync(file, `${file}.bak`);
      const out = await annotateRows(rows, (at) => moment(at));
      fs.writeFileSync(file, out.map((r) => JSON.stringify(r)).join("\n") + "\n");
      const line = `annotate ${d}: ${out.length} rows, ${new Set(out.map((r) => r.moment)).size} moments`;
      log(line);
      console.log(line);
    }
    return;
  }
  if (cmd === "backfill") {
    const now = new Date();
    const from = args[0] ?? EPOCH, to = args[1] ?? chicagoStamp(now).date;
    const { render } = engine();
    const counts = {};
    for (const at of backfillInstants(from, to, now)) {
      const { png, moment } = await render(at);
      const r = keep(at, "backfill", png, moment);
      const d = chicagoStamp(at).date;
      counts[d] ??= { saved: 0, same: 0, skip: 0 };
      counts[d][r]++;
    }
    for (const [d, c] of Object.entries(counts)) {
      const line = `backfill ${d}: ${c.saved} saved, ${c.same} already kept, ${c.skip} skipped`;
      log(line);
      console.log(line);
    }
    return;
  }
  console.error("usage: storybook.mjs live | backfill [FROM] [TO] | epub | annotate [DATE...]");
  process.exit(2);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => { log(`failed: ${err.message}`); process.exit(1); });
}
