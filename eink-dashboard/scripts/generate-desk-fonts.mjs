// Offline glyph generator for the desk panel — NOT run in the Worker.
//
// Rasterises real fonts (fonts/desk/) into pure 1-bit glyph bitmaps and writes
// src/hud/fonts-data.ts. The Worker cannot rasterise a font, and letting
// SenseCraft's cloud browser do it is what fogs text on the mono panel
// (DECISIONS #48). Glyphs that are already black and white leave it nothing to
// blur, and the type is still a real typeface (DECISIONS #67).
//
// Each glyph is rendered at 4x and averaged down before thresholding, which
// keeps thin strokes more faithfully than thresholding a 1x render.
//
// Run: npm run desk-fonts
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { Resvg } from "@resvg/resvg-js";

const FONT_DIR = "fonts/desk";
const OUT = "src/hud/fonts-data.ts";
const SUPERSAMPLE = 4;
/** A pixel is black when its averaged value is below this (0 = black, 255 = white). */
const THRESHOLD = 150;

const ASCII = Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i)).join("");
const UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const DIGITS = "0123456789";
const BIG = " " + UPPER + DIGITS + "/.:-%";

const PLEX = { family: "IBM Plex Mono", files: ["IBMPlexMono-Regular.ttf", "IBMPlexMono-Medium.ttf", "IBMPlexMono-SemiBold.ttf"] };
const AF = { family: "Aurebesh AF", files: ["AurebeshAF-Canon.otf"] };

/** key → what to rasterise. `metrics` names the file whose tables give the advances. */
const FACES = {
  plex10s: { ...PLEX, metrics: "IBMPlexMono-SemiBold.ttf", weight: 600, size: 10, chars: ASCII },
  plex11m: { ...PLEX, metrics: "IBMPlexMono-Medium.ttf", weight: 500, size: 11, chars: ASCII },
  plex12m: { ...PLEX, metrics: "IBMPlexMono-Medium.ttf", weight: 500, size: 12, chars: ASCII },
  plex12s: { ...PLEX, metrics: "IBMPlexMono-SemiBold.ttf", weight: 600, size: 12, chars: ASCII },
  plex13r: { ...PLEX, metrics: "IBMPlexMono-Regular.ttf", weight: 400, size: 13, chars: ASCII },
  plex13s: { ...PLEX, metrics: "IBMPlexMono-SemiBold.ttf", weight: 600, size: 13, chars: ASCII },
  plex15m: { ...PLEX, metrics: "IBMPlexMono-Medium.ttf", weight: 500, size: 15, chars: ASCII },
  plex15s: { ...PLEX, metrics: "IBMPlexMono-SemiBold.ttf", weight: 600, size: 15, chars: ASCII },
  plex28r: { ...PLEX, metrics: "IBMPlexMono-Regular.ttf", weight: 400, size: 28, chars: BIG },
  plex36r: { ...PLEX, metrics: "IBMPlexMono-Regular.ttf", weight: 400, size: 36, chars: BIG },
  plex50r: { ...PLEX, metrics: "IBMPlexMono-Regular.ttf", weight: 400, size: 50, chars: BIG },
  af20: { ...AF, metrics: AF.files[0], weight: 400, size: 20, chars: UPPER },
  af36: { ...AF, metrics: AF.files[0], weight: 400, size: 36, chars: DIGITS },
  af44: { ...AF, metrics: AF.files[0], weight: 400, size: 44, chars: UPPER },
  af50: { ...AF, metrics: AF.files[0], weight: 400, size: 50, chars: UPPER + DIGITS },
  af60: { ...AF, metrics: AF.files[0], weight: 400, size: 60, chars: UPPER },
  af72: { ...AF, metrics: AF.files[0], weight: 400, size: 72, chars: UPPER },
  af88: { ...AF, metrics: AF.files[0], weight: 400, size: 88, chars: UPPER },
  af112: { ...AF, metrics: AF.files[0], weight: 400, size: 112, chars: UPPER },
};

// --- Minimal sfnt reader: just enough for advances and vertical metrics --------

function readMetrics(file) {
  const b = readFileSync(path.join(FONT_DIR, file));
  const tables = {};
  for (let i = 0, n = b.readUInt16BE(4); i < n; i++) {
    const o = 12 + i * 16;
    tables[b.toString("ascii", o, o + 4)] = b.readUInt32BE(o + 8);
  }
  const unitsPerEm = b.readUInt16BE(tables.head + 18);
  const ascender = b.readInt16BE(tables.hhea + 4);
  const descender = b.readInt16BE(tables.hhea + 6);
  const numHMetrics = b.readUInt16BE(tables.hhea + 34);
  const advanceOf = (gid) => b.readUInt16BE(tables.hmtx + Math.min(gid, numHMetrics - 1) * 4);

  // cmap: prefer a Unicode BMP subtable in format 4.
  const cmap = tables.cmap;
  let sub = 0;
  for (let i = 0, n = b.readUInt16BE(cmap + 2); i < n; i++) {
    const rec = cmap + 4 + i * 8;
    const off = cmap + b.readUInt32BE(rec + 4);
    if (b.readUInt16BE(off) === 4) { sub = off; if (b.readUInt16BE(rec) === 3) break; }
  }
  if (!sub) throw new Error(`${file}: no format 4 cmap subtable`);
  const segX2 = b.readUInt16BE(sub + 6);
  const endCodes = sub + 14;
  const startCodes = endCodes + segX2 + 2;
  const idDeltas = startCodes + segX2;
  const idRangeOffsets = idDeltas + segX2;
  const glyphOf = (code) => {
    for (let s = 0; s < segX2; s += 2) {
      if (code > b.readUInt16BE(endCodes + s)) continue;
      const start = b.readUInt16BE(startCodes + s);
      if (code < start) return 0;
      const delta = b.readUInt16BE(idDeltas + s);
      const rangeOffset = b.readUInt16BE(idRangeOffsets + s);
      if (rangeOffset === 0) return (code + delta) & 0xffff;
      const gid = b.readUInt16BE(idRangeOffsets + s + rangeOffset + (code - start) * 2);
      return gid === 0 ? 0 : (gid + delta) & 0xffff;
    }
    return 0;
  };
  return { unitsPerEm, ascender, descender, glyphOf, advanceOf };
}

// --- Rasterise one face -----------------------------------------------------

const xml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function rasterise(key, face) {
  const m = readMetrics(face.metrics);
  const scale = face.size / m.unitsPerEm;
  const baseline = Math.round(m.ascender * scale);
  const height = baseline + Math.round(-m.descender * scale);
  const chars = [...face.chars].filter((ch) => ch === " " || m.glyphOf(ch.codePointAt(0)) !== 0);
  const missing = [...face.chars].filter((ch) => !chars.includes(ch));
  if (missing.length) console.warn(`  ${key}: no glyph for ${JSON.stringify(missing.join(""))}`);
  const advances = chars.map((ch) => Math.round(m.advanceOf(m.glyphOf(ch.codePointAt(0))) * scale));

  // One strip per face: every character in its own cell, so one render suffices.
  const pad = Math.ceil(face.size / 2);
  const pitch = Math.max(...advances) + pad * 2;
  const width = pitch * chars.length;
  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><rect width="${width}" height="${height}" fill="#fff"/>`;
  chars.forEach((ch, i) => {
    if (ch === " ") return;
    svg += `<text x="${i * pitch + pad}" y="${baseline}" font-family="${face.family}" font-size="${face.size}" font-weight="${face.weight}">${xml(ch)}</text>`;
  });
  svg += "</svg>";
  const img = new Resvg(svg, {
    fitTo: { mode: "zoom", value: SUPERSAMPLE },
    font: { fontFiles: face.files.map((f) => path.join(FONT_DIR, f)), loadSystemFonts: false, defaultFontFamily: face.family },
  }).render();
  const px = img.pixels; // one read: the getter copies the whole buffer each time
  const iw = img.width;
  const dark = (x, y) => {
    let sum = 0;
    for (let j = 0; j < SUPERSAMPLE; j++) for (let i = 0; i < SUPERSAMPLE; i++) sum += px[((y * SUPERSAMPLE + j) * iw + x * SUPERSAMPLE + i) * 4];
    return sum / (SUPERSAMPLE * SUPERSAMPLE) < THRESHOLD;
  };

  const glyphs = {};
  chars.forEach((ch, i) => {
    const a = advances[i];
    const bits = new Uint8Array(Math.ceil((a * height) / 8));
    let ink = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < a; x++) {
        if (!dark(i * pitch + pad + x, y)) continue;
        const n = y * a + x;
        bits[n >> 3] |= 0x80 >> (n & 7);
        ink++;
      }
    }
    if (ch !== " " && ink === 0) throw new Error(`${key}: "${ch}" rendered blank — is the font file loading?`);
    glyphs[ch] = { a, d: Buffer.from(bits).toString("base64") };
  });
  return { size: face.size, height, baseline, glyphs };
}

const out = {};
for (const [key, face] of Object.entries(FACES)) {
  out[key] = rasterise(key, face);
  console.log(`${key}: ${Object.keys(out[key].glyphs).length} glyphs, ${out[key].height}px tall`);
}

const body = `// GENERATED by scripts/generate-desk-fonts.mjs (npm run desk-fonts) — do not edit by hand.
// 1-bit glyph bitmaps rasterised from the fonts in fonts/desk/ (DECISIONS #67).
// Each glyph is \`a\` pixels wide (its advance) and the face's \`height\` tall; \`d\` is the
// bitmap, row-major, most significant bit first, 1 = ink, base64-encoded.

export interface GlyphFace {
  /** Nominal font size in pixels. */
  size: number;
  /** Cell height in pixels. */
  height: number;
  /** Distance from the top of the cell to the text baseline. */
  baseline: number;
  glyphs: Record<string, { a: number; d: string }>;
}

export type FaceKey = ${Object.keys(out).map((k) => JSON.stringify(k)).join(" | ")};

export const FACES: Record<FaceKey, GlyphFace> = ${JSON.stringify(out)};
`;
writeFileSync(OUT, body);
console.log(`Wrote ${OUT} (${(body.length / 1024).toFixed(0)} KiB)`);
