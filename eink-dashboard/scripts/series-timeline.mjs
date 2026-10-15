#!/usr/bin/env node
/**
 * What the panels will show, refresh by refresh (DECISIONS #80): walks the live schedule from
 * R2 through the real clock code every 15 minutes and prints each change, then checks that
 * every page it needs exists in R2. Page NAMES only: it never opens a page, so it is safe to
 * run in front of the owner.
 *
 *   CLOUDFLARE_ACCOUNT_ID=… node scripts/series-timeline.mjs [--from YYYY-MM-DD] [--to YYYY-MM-DD] [--approve 1,2]
 *
 * --approve pretends those chapters are approved, to see what airs once they are.
 */
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const arg = (n, f) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] : f; };
const r2 = (key) => spawnSync("npx", ["wrangler", "r2", "object", "get", `eink-series/${key}`, "--pipe", "--remote"], { cwd: REPO, encoding: "buffer" });

if (!process.env.CLOUDFLARE_ACCOUNT_ID) { console.error("set CLOUDFLARE_ACCOUNT_ID inline"); process.exit(2); }
const out = fs.mkdtempSync(path.join(os.tmpdir(), "series-timeline-"));
execFileSync("npx", ["tsc", "-p", "tsconfig.test.json", "--outDir", out], { cwd: REPO, stdio: "ignore" });
const clock = require(path.join(out, "src/series/clock.js"));
const keys = require(path.join(out, "src/series/keys.js"));

const got = r2("series/schedule.json");
if (got.status !== 0) { console.error("could not read series/schedule.json"); process.exit(1); }
const schedule = JSON.parse(got.stdout.toString());
const extra = (arg("--approve", "") || "").split(",").filter(Boolean).map(Number);
for (const s of schedule.seasons) {
  s.approved = [...new Set([...s.approved, ...extra.filter((c) => s.pending.includes(c))])].sort((a, b) => a - b);
  s.pending = s.pending.filter((c) => !s.approved.includes(c));
}

/** What /desk/series serves at `when` (mirrors handleSeries, not a preview). */
function serves(when) {
  const slot = clock.resolveSlot(clock.seriesSlot(when, schedule), schedule, false);
  const next = slot.kind === "between" ? schedule.seasons.find((s) => s.number === slot.next) : undefined;
  const days = next ? clock.daysUntil(next.premiere, when) : -1;
  if (next && (days > 0 || (days === 0 && clock.chicagoParts(when).hour < 8))) return `countdown card: ${days === 0 ? "TODAY" : `${days} day${days === 1 ? "" : "s"}`} (drawn live)`;
  return keys.slotKey(slot) ?? "text page: not started";
}

const today = clock.chicagoParts(new Date()).date;
const from = Date.parse(`${arg("--from", today)}T12:00:00Z`) - 7 * 3600e3;
const to = Date.parse(`${arg("--to", new Date(from + 28 * 86400e3).toISOString().slice(0, 10))}T23:00:00Z`);
const fmt = new Intl.DateTimeFormat("en-US", { timeZone: "America/Chicago", weekday: "short", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
console.log(`schedule: ${schedule.seasons.map((s) => `season ${s.number} approved [${s.approved}] pending [${s.pending}]`).join("; ")}${extra.length ? ` (pretending ${extra} approved)` : ""}\n`);
const needed = new Set();
let last = null;
for (let t = from; t <= to; t += 15 * 60e3) {
  const v = serves(new Date(t));
  if (v.startsWith("series/")) needed.add(v);
  if (v !== last) console.log(`${fmt.format(new Date(t)).padEnd(24)} ${v}`);
  last = v;
}
const missing = [...needed].filter((k) => r2(k).status !== 0);
console.log(`\n${needed.size} stored pages needed; ${missing.length ? `MISSING in R2: ${missing.join(", ")}` : "all present in R2"}`);
process.exit(missing.length ? 1 : 0);
