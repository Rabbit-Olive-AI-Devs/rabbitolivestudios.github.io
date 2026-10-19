#!/usr/bin/env node
/**
 * Keeps every new frame of the desk story page for a story book (DECISIONS #95). Runs on the
 * Mac mini:
 *
 *   node scripts/storybook.mjs live                 — fetch the page now; save its picture if new
 *   node scripts/storybook.mjs backfill [FROM] [TO] — render past moments on this Mac with the
 *                                                     Worker's code, at 10-minute steps (Chicago
 *                                                     dates, default 2026-10-17 to now)
 *
 * Frames go to Dropbox, `<BOOK>/<YYYY-MM-DD>/<HHMM>.png` (Chicago time), with one append-only
 * `index.jsonl` per day: {time, at, day, file, sha256, bytes, source}. A frame is saved only when
 * its SHA-256 is not already in that day's index. A failed fetch or the text fallback page is
 * logged and skipped, never saved. No AI, no KV writes: the page itself only reads the cache.
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

/**
 * Renders past moments on this Mac with the Worker's own code (compiled like `npm run test:utils`),
 * so a backfill costs the Worker nothing: an earlier version fetched `?at=` from the live page and
 * its burst used up the free plan's CPU allowance, which briefly failed every desk page with 1102.
 * The cached context the page reads (weather, planet, Flamengo board) comes from production KV as
 * it is now: reads only, once per key.
 */
function localRenderer() {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "storybook-"));
  execFileSync("npx", ["tsc", "-p", "tsconfig.test.json", "--outDir", out], { cwd: REPO, stdio: "ignore" });
  const { loadCastawayContext, castawayView } = require(path.join(out, "src/castaway.js"));
  const { drawCastawayPage } = require(path.join(out, "src/hud/castaway-page.js"));
  const { canvasToPng } = require(path.join(out, "src/hud/page.js"));
  const kv = new Map();
  const env = { CACHE: { get: async (key, type) => {
    if (!kv.has(key)) {
      let raw = null;
      try {
        raw = execFileSync("npx", ["wrangler", "kv", "key", "get", "--remote", "--namespace-id", KV_NAMESPACE, key],
          { cwd: REPO, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], env: { ...process.env, CLOUDFLARE_ACCOUNT_ID: ACCOUNT } });
      } catch { /* missing key: the page draws without it */ }
      kv.set(key, raw);
    }
    const raw = kv.get(key);
    return raw && type === "json" ? JSON.parse(raw) : raw;
  } } };
  return async (at) => {
    try {
      return Buffer.from(await canvasToPng(drawCastawayPage(castawayView(at, await loadCastawayContext(env, at)))));
    } catch (err) {
      log(`backfill ${at.toISOString()}: render failed (${err.message}), skipped`);
      return null;
    }
  };
}

/**
 * Save one frame for instant `at` unless that day already holds it (same SHA-256, or for a
 * backfill a frame already kept at that minute). Returns "saved" | "same" | "skip".
 */
async function keep(at, source, png) {
  if (!png) return "skip";
  const { date, time, stem } = chicagoStamp(at);
  const dir = path.join(BOOK, date);
  fs.mkdirSync(dir, { recursive: true });
  const index = readIndex(dir);
  const sha256 = crypto.createHash("sha256").update(png).digest("hex");
  if (index.some((e) => e.sha256 === sha256 || (source === "backfill" && e.time === time))) return "same";
  const file = frameName(stem, new Set(index.map((e) => e.file)));
  fs.writeFileSync(path.join(dir, file), png);
  const entry = { time, at: at.toISOString(), day: dayNumber(date), file, sha256, bytes: png.length, source };
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
    const r = await keep(at, "live", await fetchLive(at));
    log(`live ${chicagoStamp(at).date} ${chicagoStamp(at).time}: ${r === "same" ? "no change" : r}`);
    process.exit(r === "skip" ? 1 : 0);
  }
  if (cmd === "backfill") {
    const now = new Date();
    const from = args[0] ?? EPOCH, to = args[1] ?? chicagoStamp(now).date;
    const render = localRenderer();
    const counts = {};
    for (const at of backfillInstants(from, to, now)) {
      const r = await keep(at, "backfill", await render(at));
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
  console.error("usage: storybook.mjs live | backfill [FROM] [TO]");
  process.exit(2);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => { log(`failed: ${err.message}`); process.exit(1); });
}
