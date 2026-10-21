#!/usr/bin/env node
/**
 * The desk serial's offline build (docs/superpowers/specs/2026-10-21-desk-serial-design.md):
 * one day file plus Jarvis's paintings → one stored page per beat in the private series bucket.
 *
 *   node scripts/serial-build.mjs <day.json> <pictures-dir> [--upload]
 *
 * The day file is {date, day, beats: [{id, start, title, text: string[], log, scene: {focus: {fx, fy}}}]}
 * and lives outside the repo (~/desk-serial-work/): it is the story, a surprise. A beat builds when
 * <pictures-dir>/<id>.png exists, is at least 800x315 and not blank, and its text fits; the others
 * are left out of the uploaded day file, so the previous beat holds until the picture lands. Prints
 * counts and reasons only, never text. Draws with the Worker's own code, compiled here.
 */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const BUCKET = "eink-series";
const ACCOUNT = "f22a506dedde3bb3837157cd47d5fe5c";

/** The Worker's drawing code, compiled to a temp folder (like `npm run test:utils`). */
function compile() {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "serial-build-"));
  execFileSync("npx", ["tsc", "-p", "tsconfig.test.json", "--outDir", out], { cwd: REPO, stdio: "ignore" });
  return load(out);
}

/** The modules the build uses, from a compiled output folder. */
export function load(out) {
  const from = (p) => require(path.join(out, p));
  return {
    page: from("src/hud/castaway-page.js"), hud: from("src/hud/page.js"), art: from("src/series/art.js"),
    decode: from("src/png-decode.js"), png: from("src/png.js"), serial: from("src/serial.js"), title: from("src/hud/castaway-page.js").CASTAWAY_TITLE,
  };
}

/** One beat's page HTML, or {error}. `earlier` are the day's previous beats, oldest first. */
export async function buildBeat(lib, day, beat, earlier, bytes) {
  const log = earlier.slice(-2).reverse().map((b) => ({ time: b.start, text: b.log }));
  const text = beat.text.join(" ");
  const misfit = lib.page.storyTextFits(beat.title, text, log);
  if (misfit) return { error: misfit };
  if (!(bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47)) return { error: "not a PNG" };
  let img;
  try { img = await lib.decode.decodePNG(bytes); } catch (err) { return { error: `unreadable PNG (${err.message})` }; }
  const { w, h } = lib.page.SERIAL_PICTURE;
  if (img.width < w || img.height < h) return { error: `picture ${img.width}x${img.height} is smaller than ${w}x${h}` };
  let rgb = img.rgb;
  if (!rgb) { rgb = new Uint8Array(img.width * img.height * 3); img.gray.forEach((v, i) => rgb.fill(v, i * 3, i * 3 + 3)); }
  const full = { w: img.width, h: img.height, rgb: lib.art.flattenAlpha(rgb, img.alpha) };
  if (lib.art.isNearlyBlank(full)) return { error: "picture is blank" };
  const f = beat.scene?.focus ?? { fx: 0.5, fy: 0.5 };
  const { image } = lib.art.cropToFill(full, w, h, [f.fx * img.width, f.fy * img.height]);
  const canvas = lib.page.drawSerialPage({ day: day.day, time: beat.start, chapter: beat.title, text, log }, image);
  return { html: lib.hud.hudPageHTML(lib.title, lib.png.pngToBase64(await lib.hud.canvasToPng(canvas))) };
}

function put(key, file, type) {
  const args = ["wrangler", "r2", "object", "put", `${BUCKET}/${key}`, "--remote", "--file", file, "--content-type", type];
  const env = { ...process.env, CLOUDFLARE_ACCOUNT_ID: ACCOUNT };
  try {
    execFileSync("npx", args, { cwd: REPO, stdio: "ignore", env });
  } catch {
    console.log(`upload ${key} failed, retrying in 10 s`);
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10_000);
    execFileSync("npx", args, { cwd: REPO, stdio: "ignore", env });
  }
}

async function main() {
  const [dayFile, picDir, flag] = process.argv.slice(2);
  if (!dayFile || !picDir) {
    console.error("usage: serial-build.mjs <day.json> <pictures-dir> [--upload]");
    process.exit(2);
  }
  const day = JSON.parse(fs.readFileSync(dayFile, "utf8"));
  const beats = day.beats.slice().sort((a, b) => a.start.localeCompare(b.start));
  const lib = compile();
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "serial-pages-"));
  const built = [];
  for (const [i, beat] of beats.entries()) {
    const pic = path.join(picDir, `${beat.id}.png`);
    if (!fs.existsSync(pic)) { console.log(`${beat.id}: waiting for its picture`); continue; }
    const r = await buildBeat(lib, day, beat, beats.slice(0, i), new Uint8Array(fs.readFileSync(pic)));
    if (r.error) { console.log(`${beat.id}: not built (${r.error})`); continue; }
    const file = path.join(out, `${beat.id}.html`);
    fs.writeFileSync(file, r.html);
    built.push({ id: beat.id, start: beat.start, key: lib.serial.serialPageKey(beat.id), file });
    console.log(`${beat.id}: built (${Math.round(r.html.length / 1024)} KB)`);
  }
  if (flag === "--upload") {
    for (const b of built) put(b.key, b.file, "text/html; charset=utf-8");
    const schedule = path.join(out, "day.json");
    fs.writeFileSync(schedule, JSON.stringify({ date: day.date, day: day.day, beats: built.map(({ id, start, key }) => ({ id, start, key })) }));
    put(lib.serial.serialDayKey(day.date), schedule, "application/json");
    console.log(`${day.date}: uploaded ${built.length} of ${beats.length} beats`);
  } else {
    console.log(`${day.date}: built ${built.length} of ${beats.length} beats in ${out} (not uploaded)`);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => { console.error(`serial-build failed: ${err.message}`); process.exit(1); });
}
