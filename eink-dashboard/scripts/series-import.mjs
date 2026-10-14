#!/usr/bin/env node
/**
 * Offline import for the series (DECISIONS #71). Runs on the Mac mini, never in
 * the Worker. Packages and output stay OUTSIDE the repo (~/series-work/): the repo
 * is public and they are spoilers.
 *
 *   npm run series -- build <package.zip|dir> [--chapter N] [--out DIR]
 *   npm run series -- upload <out-dir>
 *   npm run series -- approve --season S --chapter C
 *   npm run series -- templates
 *
 * Needs macOS `sips` and `ditto`, and CLOUDFLARE_ACCOUNT_ID for upload/approve.
 */
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const BUCKET = "eink-series";
const SCHEDULE_KEY = "series/schedule.json";

function compile() {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "series-build-"));
  execFileSync("npx", ["tsc", "-p", "tsconfig.test.json", "--outDir", out], { cwd: REPO, stdio: "inherit" });
  const load = (p) => require(path.join(out, p));
  return {
    pkgv: load("src/series/package.js"), build: load("src/series/build.js"), art: load("src/series/art.js"),
    tpl: load("src/series/templates.js"), types: load("src/series/types.js"), dec: load("src/png-decode.js"),
    hud: { canvas: load("src/hud/canvas.js"), type: load("src/hud/type.js"), parts: load("src/hud/parts.js") },
    png: load("src/png.js"),
  };
}

function arg(name, fallback = null) {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
}

function unpack(src) {
  if (fs.statSync(src).isDirectory()) return src;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "series-pkg-"));
  execFileSync("ditto", ["-x", "-k", src, dir]);
  const json = execFileSync("find", [dir, "-name", "season.json", "-not", "-path", "*/__MACOSX/*"]).toString().trim().split("\n")[0];
  if (!json) throw new Error("no season.json in the zip");
  return path.dirname(json);
}

async function cmdBuild() {
  const src = process.argv[3];
  const out = arg("--out", path.join(os.homedir(), "series-work", "out"));
  const only = arg("--chapter") ? Number(arg("--chapter")) : null;
  const m = compile();
  const dir = unpack(src);
  const raw = JSON.parse(fs.readFileSync(path.join(dir, "season.json"), "utf8"));
  const artDir = path.join(dir, "art");
  const artFiles = new Set(fs.existsSync(artDir) ? fs.readdirSync(artDir) : []);
  const bannedFile = path.join(dir, "banned.txt");
  const extra = fs.existsSync(bannedFile) ? fs.readFileSync(bannedFile, "utf8").split("\n").map((s) => s.trim()).filter(Boolean) : [];
  const { pkg, problems } = m.pkgv.validatePackage(raw, artFiles, extra);
  if (!pkg) return report(out, problems);

  // Every referenced picture → PNG with sips (reads PNG, JPEG, WebP, HEIC) → decoded RGB.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "series-art-"));
  const decoded = new Map();
  const chapters = only ? [only] : pkg.chapters.map((c) => c.number);
  for (const c of chapters) {
    for (const sc of pkg.chapters[c - 1].scenes) for (const p of sc.panels) {
      if (p.art === "hold" || decoded.has(p.art)) continue;
      const file = path.join(tmp, `${decoded.size}.png`);
      try {
        execFileSync("sips", ["-s", "format", "png", path.join(artDir, p.art), "--out", file], { stdio: "ignore" });
      } catch { decoded.set(p.art, null); continue; }
      const d = await m.dec.decodePNG(new Uint8Array(fs.readFileSync(file)));
      const rgb = d.rgb ?? Uint8Array.from({ length: d.width * d.height * 3 }, (_, i) => d.gray[Math.floor(i / 3)]);
      decoded.set(p.art, { w: d.width, h: d.height, rgb: m.art.flattenAlpha(rgb, d.alpha) });
    }
  }
  const all = [...problems];
  const files = new Map();
  for (const c of chapters) {
    const r = await m.build.buildChapter(pkg, c, (n) => decoded.get(n) ?? null);
    all.push(...r.problems);
    for (const [k, v] of r.files) files.set(k, v);
  }
  for (const [k, v] of await m.build.buildCards(pkg.season)) files.set(k, v);
  if (all.length) return report(out, all);
  fs.rmSync(out, { recursive: true, force: true });
  for (const [k, v] of files) {
    fs.mkdirSync(path.dirname(path.join(out, k)), { recursive: true });
    fs.writeFileSync(path.join(out, k), v);
  }
  fs.writeFileSync(path.join(out, "season.json"), JSON.stringify({ season: pkg.season, chapters: chapters }, null, 2));
  report(out, [], `${files.size} pages for chapters ${chapters.join(", ")} written to ${out}`);
}

function report(out, problems, ok) {
  fs.mkdirSync(out, { recursive: true });
  const text = problems.length ? `REFUSED: ${problems.length} problems\n` + problems.map((p) => `  - ${p}`).join("\n") : ok;
  fs.writeFileSync(path.join(out, "report.txt"), text + "\n");
  console.log(text);
  if (problems.length) process.exit(1);
}

function wrangler(args, opts = {}) {
  if (!process.env.CLOUDFLARE_ACCOUNT_ID) throw new Error("set CLOUDFLARE_ACCOUNT_ID inline");
  return execFileSync("npx", ["wrangler", ...args, "--remote"], { cwd: REPO, ...opts });
}

function readSchedule() {
  try { return JSON.parse(wrangler(["r2", "object", "get", `${BUCKET}/${SCHEDULE_KEY}`, "--pipe"], { stdio: ["ignore", "pipe", "ignore"] }).toString()); }
  catch { return { seasons: [] }; }
}

function writeSchedule(s) {
  const f = path.join(os.tmpdir(), "series-schedule.json");
  fs.writeFileSync(f, JSON.stringify(s, null, 2));
  wrangler(["r2", "object", "put", `${BUCKET}/${SCHEDULE_KEY}`, "--file", f, "--content-type", "application/json"], { stdio: "inherit" });
}

async function cmdUpload() {
  const out = process.argv[3];
  const meta = JSON.parse(fs.readFileSync(path.join(out, "season.json"), "utf8"));
  const keys = execFileSync("find", ["series", "-name", "*.png"], { cwd: out }).toString().trim().split("\n").filter(Boolean);
  let next = 0, failed = 0;
  const worker = async () => {
    while (next < keys.length) {
      const k = keys[next++];
      await new Promise((res) => {
        const p = spawn("npx", ["wrangler", "r2", "object", "put", `${BUCKET}/${k}`, "--file", path.join(out, k), "--content-type", "image/png", "--remote"], { cwd: REPO, stdio: "ignore" });
        p.on("exit", (code) => { if (code !== 0) { failed++; console.error(`failed: ${k}`); } res(); });
      });
    }
  };
  await Promise.all(Array.from({ length: 6 }, worker));
  if (failed) { console.error(`${failed} uploads failed; schedule not changed. Re-run upload.`); process.exit(1); }
  const s = readSchedule();
  const m = compile();
  let entry = s.seasons.find((x) => x.number === meta.season.number);
  if (!entry) { entry = { ...meta.season, chapters: m.types.plannedChapters(meta.season.number), approved: [], pending: [] }; s.seasons.push(entry); }
  Object.assign(entry, meta.season, { chapters: m.types.plannedChapters(meta.season.number) });
  for (const c of meta.chapters) if (!entry.approved.includes(c) && !entry.pending.includes(c)) entry.pending.push(c);
  writeSchedule(s);
  console.log(`uploaded ${keys.length} pages; chapters ${meta.chapters.join(", ")} pending approval`);
}

function cmdApprove() {
  const season = Number(arg("--season")), chapter = Number(arg("--chapter"));
  const s = readSchedule();
  const e = s.seasons.find((x) => x.number === season);
  if (!e || !(e.pending.includes(chapter) || e.approved.includes(chapter))) { console.error("that chapter is not uploaded"); process.exit(1); }
  e.pending = e.pending.filter((c) => c !== chapter);
  if (!e.approved.includes(chapter)) e.approved.push(chapter);
  e.approved.sort((a, b) => a - b);
  writeSchedule(s);
  console.log(`season ${season} chapter ${chapter} approved`);
}

async function cmdTemplates() {
  const m = compile();
  const { createCanvas, strokeRect } = m.hud.canvas;
  const { drawType } = m.hud.type;
  const { hatch } = m.hud.parts;
  const dir = path.join(REPO, "docs", "series", "templates");
  fs.mkdirSync(dir, { recursive: true });
  for (const t of Object.values(m.tpl.TEMPLATES)) {
    const c = createCanvas();
    t.panels.forEach((p, i) => {
      strokeRect(c, p.x, p.y, p.w, p.h, 2);
      const calm = Math.round(p.h * m.tpl.CALM_FRACTION);
      if (t.id === "splash") hatch(c, p.x + 16, p.y + p.h - 112, 360, 96, 6);
      else if (t.balloonsPerPanel > 0 || t.panelCaptions) hatch(c, p.x + 2, p.y + 2, p.w - 4, calm, 6);
      drawType(c, "plex13s", p.x + p.w / 2, p.y + p.h / 2, `PANEL ${i + 1}  ${p.w} x ${p.h}`, { align: "center" });
    });
    if (t.strip) { hatch(c, 0, m.tpl.STRIP_TOP, 800, 80, 6); drawType(c, "plex13s", 400, 446, "NARRATION AND CREDITS (OURS)", { align: "center" }); }
    if (t.column) { hatch(c, t.column.x, t.column.y, t.column.w, t.column.h, 6); }
    drawType(c, "plex15s", 400, 26, `TEMPLATE ${t.id.toUpperCase()}: HATCHED = KEEP CALM`, { align: "center" });
    fs.writeFileSync(path.join(dir, `${t.id}.png`), await m.png.encodePNG1Bit(c.px, 800, 480));
  }
  console.log(`template guides written to ${dir}`);
}

const cmd = process.argv[2];
const run = { build: cmdBuild, upload: cmdUpload, approve: cmdApprove, templates: cmdTemplates }[cmd];
if (!run) { console.error("usage: series-import.mjs build|upload|approve|templates"); process.exit(2); }
await run();
