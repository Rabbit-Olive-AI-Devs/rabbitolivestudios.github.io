#!/usr/bin/env node
/**
 * Offline import for the series (DECISIONS #71). Runs on the Mac mini, never in
 * the Worker. Packages and output stay OUTSIDE the repo (~/series-work/): the repo
 * is public and they are spoilers.
 *
 *   npm run series -- build <package.zip|dir> [--chapter N] [--out DIR]
 *   npm run series -- upload <out-dir>
 *   npm run series -- approve --season S --chapter C --revision R
 *   npm run series -- export --season S --chapter C [--to DIR]   (the proof export for Jarvis, from R2)
 *   npm run series -- templates
 *   npm run series -- preview <picture|folder> --out DIR   (a writer's style test on all three panels)
 *
 * Needs macOS `sips` and `ditto`, and CLOUDFLARE_ACCOUNT_ID for upload/approve.
 */
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { einkPaths } from "./eink-paths.mjs";

const require = createRequire(import.meta.url);
const REPO = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const BUCKET = "eink-series";
const SCHEDULE_KEY = "series/schedule.json";
// Jarvis's spoiler review folder (DECISIONS #78): outside the shared series mailbox.
const PROOFS = process.env.SERIES_PROOFS ?? path.join(einkPaths().dropbox, "00_JARVIS_SPOILERS", "series-proofs");

function compile() {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), "series-build-"));
  execFileSync("npx", ["tsc", "-p", "tsconfig.test.json", "--outDir", out], { cwd: REPO, stdio: "inherit" });
  const load = (p) => require(path.join(out, p));
  return {
    pkgv: load("src/series/package.js"), build: load("src/series/build.js"), art: load("src/series/art.js"),
    tpl: load("src/series/templates.js"), types: load("src/series/types.js"), dec: load("src/png-decode.js"),
    schedule: load("src/series/schedule.js"), page: load("src/hud/page.js"), proof: load("src/series/proof.js"),
    hud: { canvas: load("src/hud/canvas.js"), type: load("src/hud/type.js"), parts: load("src/hud/parts.js") },
    png: load("src/png.js"), mailbox: load("src/series/mailbox.js"),
  };
}

function arg(name, fallback = null) {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : fallback;
}

/** Build output is a spoiler: never inside the (public) repo, and only ever a folder we made. */
function safeOut(dir) {
  const abs = path.resolve(dir);
  const repoRoot = path.resolve(REPO, "..");
  if (abs === repoRoot || abs.startsWith(repoRoot + path.sep)) throw new Error(`--out ${abs} is inside the repo; use ~/series-work/…`);
  if ([os.homedir(), "/", process.cwd()].includes(abs)) throw new Error(`--out ${abs} would be emptied; give a dedicated folder`);
  return abs;
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
  const out = safeOut(arg("--out", path.join(os.homedir(), "series-work", "out")));
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
      try {
        const d = await m.dec.decodePNG(new Uint8Array(fs.readFileSync(file)));
        const rgb = d.rgb ?? Uint8Array.from({ length: d.width * d.height * 3 }, (_, i) => d.gray[Math.floor(i / 3)]);
        decoded.set(p.art, { w: d.width, h: d.height, rgb: m.art.flattenAlpha(rgb, d.alpha) });
      } catch (err) {
        console.error(`${p.art}: ${err.message}`);
        decoded.set(p.art, null); // reported per scene as "could not be read"
      }
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
  fs.rmSync(path.join(out, "series"), { recursive: true, force: true });
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

/**
 * Only a definite "no such object" means an empty schedule. Any other failure (network,
 * expired login, bad JSON) aborts: writing back a guessed schedule would wipe approvals.
 */
function readSchedule(m) {
  let text;
  try {
    text = wrangler(["r2", "object", "get", `${BUCKET}/${SCHEDULE_KEY}`, "--pipe"], { stdio: ["ignore", "pipe", "pipe"] }).toString();
  } catch (err) {
    const msg = `${err.stderr ?? ""}${err.stdout ?? ""}${err.message}`;
    if (/does not exist|NoSuchKey|not found/i.test(msg)) return { seasons: [] };
    throw new Error(`could not read ${SCHEDULE_KEY}; nothing changed:\n${msg}`);
  }
  const parsed = JSON.parse(text);
  if (!m.schedule.isSchedule(parsed)) throw new Error(`${SCHEDULE_KEY} has the wrong shape; nothing changed`);
  const backups = path.join(os.homedir(), "series-work", "schedule-backups");
  fs.mkdirSync(backups, { recursive: true });
  fs.writeFileSync(path.join(backups, `${new Date().toISOString().replace(/[:.]/g, "-")}.json`), text);
  return parsed;
}

function writeSchedule(s) {
  const f = path.join(os.tmpdir(), "series-schedule.json");
  fs.writeFileSync(f, JSON.stringify(s, null, 2));
  wrangler(["r2", "object", "put", `${BUCKET}/${SCHEDULE_KEY}`, "--file", f, "--content-type", "application/json"], { stdio: "inherit" });
}

async function cmdUpload() {
  const out = process.argv[3];
  const meta = JSON.parse(fs.readFileSync(path.join(out, "season.json"), "utf8"));
  const m = compile();
  const before = readSchedule(m);
  const plan = m.schedule.mergeUpload(before, meta.season, meta.chapters, process.argv.includes("--replace-approved"));
  if (plan.skipped.length) console.log(`chapters ${plan.skipped.join(", ")} are approved and on air: not replaced (re-run with --replace-approved once the showrunner approved the fix)`);
  const pad = (n) => String(n).padStart(2, "0");
  const skip = new Set(plan.skipped.map((c) => `series/s${pad(meta.season.number)}/c${pad(c)}/`));
  const keys = execFileSync("find", ["series", "-name", "*.html"], { cwd: out }).toString().trim().split("\n").filter(Boolean)
    .filter((k) => ![...skip].some((prefix) => k.startsWith(prefix)));
  const put = (k) => new Promise((res) => {
    const p = spawn("npx", ["wrangler", "r2", "object", "put", `${BUCKET}/${k}`, "--file", path.join(out, k), "--content-type", "text/html; charset=utf-8", "--remote"], { cwd: REPO, stdio: "ignore" });
    p.on("exit", (code) => res(code === 0));
  });
  const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
  let next = 0, failed = 0;
  const worker = async () => {
    while (next < keys.length) {
      const k = keys[next++];
      // A put fails now and then on the network; try each page three times before giving up.
      let ok = false;
      for (let attempt = 1; attempt <= 3 && !ok; attempt++) {
        ok = await put(k);
        if (!ok && attempt < 3) { console.error(`retrying (${attempt}/2): ${k}`); await sleep(attempt * 3000); }
      }
      if (!ok) { failed++; console.error(`failed: ${k}`); }
    }
  };
  await Promise.all(Array.from({ length: 6 }, worker));
  if (failed) { console.error(`${failed} uploads failed; schedule not changed. Re-run upload.`); process.exit(1); }
  writeSchedule(plan.schedule);
  console.log(`uploaded ${keys.length} pages; chapters ${plan.upload.join(", ")} uploaded (new ones pending approval)`);
  // Jarvis reviews what R2 now holds, not this build folder; a failed export leaves the upload standing.
  for (const c of plan.upload) {
    try {
      const e = await exportChapter(m, meta.season.number, c, PROOFS);
      console.log(`proof export: season ${meta.season.number} chapter ${c} revision ${e.revision} in ${e.dir}`);
    } catch (err) {
      console.error(`proof export FAILED: season ${meta.season.number} chapter ${c}: ${err.message}`);
    }
  }
}

/** One R2 object as bytes, tried three times. */
function r2Get(key) {
  return new Promise((resolve, reject) => {
    const tryOnce = (attempt) => {
      const p = spawn("npx", ["wrangler", "r2", "object", "get", `${BUCKET}/${key}`, "--pipe", "--remote"], { cwd: REPO, stdio: ["ignore", "pipe", "pipe"] });
      const chunks = [], err = [];
      p.stdout.on("data", (d) => chunks.push(d));
      p.stderr.on("data", (d) => err.push(d));
      p.on("exit", (code) => {
        if (code === 0) return resolve(Buffer.concat(chunks));
        const msg = Buffer.concat(err).toString();
        if (/does not exist|NoSuchKey|not found/i.test(msg)) return reject(new Error(`missing in R2: ${key}`));
        if (attempt < 3) return setTimeout(() => tryOnce(attempt + 1), attempt * 3000);
        reject(new Error(`could not read ${key}: ${msg.slice(-300)}`));
      });
    };
    tryOnce(1);
  });
}

/** Every page of a chapter from R2, hashed, and the chapter's revision. Refuses an incomplete set. */
async function readChapter(m, season, chapter) {
  if (!process.env.CLOUDFLARE_ACCOUNT_ID) throw new Error("set CLOUDFLARE_ACCOUNT_ID inline");
  const keys = m.proof.chapterPageKeys(season, chapter);
  const pages = new Array(keys.length);
  const missing = [];
  let next = 0;
  const worker = async () => {
    while (next < keys.length) {
      const i = next++;
      try {
        const bytes = await r2Get(keys[i]);
        pages[i] = { key: keys[i], bytes, sha256: await m.proof.sha256Hex(bytes) };
      } catch (err) { missing.push(err.message); }
    }
  };
  await Promise.all(Array.from({ length: 6 }, worker));
  if (missing.length) throw new Error(`chapter ${chapter} is incomplete in R2:\n  ${missing.join("\n  ")}`);
  return { pages, revision: await m.proof.revisionOf(pages) };
}

const sha256File = (m, f) => m.proof.sha256Hex(new Uint8Array(fs.readFileSync(f)));

/**
 * Copy one uploaded chapter's exact pages into the proof folder as an immutable revision:
 * html/ (the bytes in R2), png/ (the picture inside each, to look at), MANIFEST.json, and
 * READY written last, after every file has been read back and checked against the manifest.
 */
async function exportChapter(m, season, chapter, root) {
  const s = readSchedule(m).seasons.find((x) => x.number === season);
  const status = s?.approved.includes(chapter) ? "approved" : s?.pending.includes(chapter) ? "pending" : null;
  if (!status) throw new Error(`season ${season} chapter ${chapter} is not uploaded`);
  const { pages, revision } = await readChapter(m, season, chapter);
  const pad = (n) => String(n).padStart(2, "0");
  const chapterDir = path.join(root, `s${pad(season)}-c${pad(chapter)}`);
  const dir = path.join(chapterDir, `rev-${revision}`);
  const current = () => fs.writeFileSync(path.join(chapterDir, "CURRENT.txt"), `${revision}\n`);
  if (fs.existsSync(path.join(dir, "READY"))) { current(); return { revision, dir, reused: true }; }

  const staging = path.join(chapterDir, `.staging-${revision}-${process.pid}`);
  fs.rmSync(staging, { recursive: true, force: true });
  fs.mkdirSync(path.join(staging, "html"), { recursive: true });
  fs.mkdirSync(path.join(staging, "png"), { recursive: true });
  const files = [];
  for (const p of pages) {
    const name = path.basename(p.key, ".html");
    const png = m.proof.pagePNG(p.bytes.toString("utf8"));
    const size = png && m.proof.pngSize(png);
    if (!size) throw new Error(`${p.key} carries no PNG`);
    fs.writeFileSync(path.join(staging, "html", `${name}.html`), p.bytes);
    fs.writeFileSync(path.join(staging, "png", `${name}.png`), png);
    files.push({
      r2Key: p.key, page: name, kind: name === "teaser" ? "teaser" : name.startsWith("r") ? "replay" : "scene",
      html: { path: `html/${name}.html`, bytes: p.bytes.length, sha256: p.sha256 },
      png: { path: `png/${name}.png`, bytes: png.length, sha256: await m.proof.sha256Hex(png), width: size.width, height: size.height },
    });
  }
  const manifest = {
    season, chapter, revision, scheduleStatus: status, exportedAt: new Date().toISOString(),
    source: `R2 bucket ${BUCKET}, read back at export time: these are the bytes the panels are served`,
    revisionRule: "first 16 hex of SHA-256 over the sorted lines \"<r2Key> <html sha256>\\n\"",
    expected: { scenes: 25, replays: 25, teaser: 1, total: 51 }, count: files.length, files,
  };
  fs.writeFileSync(path.join(staging, "MANIFEST.json"), JSON.stringify(manifest, null, 2) + "\n");
  // Read every file back before READY: what we claim is what is on disk.
  for (const f of files) for (const part of [f.html, f.png]) {
    if ((await sha256File(m, path.join(staging, part.path))) !== part.sha256) throw new Error(`${part.path} did not write correctly`);
  }
  fs.renameSync(staging, dir);
  const manifestSha = await sha256File(m, path.join(dir, "MANIFEST.json"));
  fs.writeFileSync(path.join(dir, "READY"), `revision ${revision}\nfiles ${files.length * 2 + 1} (51 html, 51 png, MANIFEST.json)\nmanifest sha256 ${manifestSha}\nwritten ${new Date().toISOString()}\n`);
  current();
  return { revision, dir, reused: false };
}

async function cmdExport() {
  const season = Number(arg("--season")), chapter = Number(arg("--chapter"));
  if (!Number.isInteger(season) || !Number.isInteger(chapter)) { console.error("export needs --season S --chapter C"); process.exit(2); }
  const e = await exportChapter(compile(), season, chapter, path.resolve(arg("--to", PROOFS)));
  console.log(`proof export: season ${season} chapter ${chapter} revision ${e.revision} in ${e.dir}${e.reused ? " (already exported)" : ""}`);
}

async function cmdApprove() {
  const season = Number(arg("--season")), chapter = Number(arg("--chapter")), revision = String(arg("--revision", "")).toLowerCase();
  const m = compile();
  if (!m.proof.REVISION.test(revision)) { console.error("approve needs --revision: the proof export revision that was reviewed"); process.exit(1); }
  const s = readSchedule(m);
  const e = s.seasons.find((x) => x.number === season);
  if (!e || !(e.pending.includes(chapter) || e.approved.includes(chapter))) { console.error("that chapter is not uploaded"); process.exit(1); }
  // The pages on air must be the ones reviewed: recompute the revision from R2 itself.
  const now = (await readChapter(m, season, chapter)).revision;
  if (now !== revision) { console.error(`the pages changed since revision ${revision}; R2 now holds revision ${now}. Review that export first`); process.exit(1); }
  // The showrunner's own seasons also need the independent PASS for this revision (#79).
  if (m.mailbox.needsIndependentReview(season)) {
    const pad = (n) => String(n).padStart(2, "0");
    const f = path.join(os.homedir(), "series-work", "reviews", `s${pad(season)}-c${pad(chapter)}-${revision}.json`);
    const r = fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : null;
    if (r?.verdict !== "PASS" || r.revision !== revision) { console.error(`season ${season} needs the independent PASS for revision ${revision} first${r ? ` (the review said ${r.verdict})` : " (no review yet)"}`); process.exit(1); }
  }
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
/**
 * A style test: each picture as the panels will get it (DECISIONS #73), untouched, only
 * cropped to fill the 800x480 screen, without words: <name>.html (the one-request page the
 * watcher uploads) and <name>.png (the same picture, to look at).
 */
async function cmdPreview() {
  const src = path.resolve(process.argv[3]);
  const out = safeOut(arg("--out", path.join(os.homedir(), "series-work", "previews")));
  const m = compile();
  const files = fs.statSync(src).isDirectory()
    ? fs.readdirSync(src).filter((f) => /\.(png|jpe?g|webp|heic|tiff?)$/i.test(f)).map((f) => path.join(src, f))
    : [src];
  const W = 800, H = 480, tmp = fs.mkdtempSync(path.join(os.tmpdir(), "series-preview-"));
  fs.mkdirSync(out, { recursive: true });
  const done = [];
  for (const f of files) {
    // Web-safe: the name becomes part of the style-test address (DECISIONS #72).
    const name = path.basename(f).replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "") || "picture";
    try {
      const png = path.join(tmp, `${name}.png`);
      execFileSync("sips", ["-s", "format", "png", f, "--out", png], { stdio: "ignore" });
      const d = await m.dec.decodePNG(new Uint8Array(fs.readFileSync(png)));
      const rgbIn = d.rgb ?? Uint8Array.from({ length: d.width * d.height * 3 }, (_, i) => d.gray[Math.floor(i / 3)]);
      const img = { w: d.width, h: d.height, rgb: m.art.flattenAlpha(rgbIn, d.alpha) };
      if (img.w < W || img.h < H) { done.push(`${name}: ${img.w}x${img.h} is smaller than the ${W}x${H} screen`); continue; }
      const { image } = m.art.cropToFill(img, W, H);
      const page = await m.png.encodePNGRGB(image.rgb, W, H);
      fs.writeFileSync(path.join(out, `${name}.png`), page);
      fs.writeFileSync(path.join(out, `${name}.html`), m.page.hudPageHTML("The Series", m.png.pngToBase64(page)));
      done.push(`${name}: as-is page written`);
    } catch (err) {
      done.push(`${name}: could not be read (${err.message})`);
    }
  }
  console.log(done.join("\n"));
}

const run = { build: cmdBuild, upload: cmdUpload, approve: cmdApprove, export: cmdExport, templates: cmdTemplates, preview: cmdPreview }[cmd];
if (!run) { console.error("usage: series-import.mjs build|upload|approve|export|templates|preview"); process.exit(2); }
await run();
