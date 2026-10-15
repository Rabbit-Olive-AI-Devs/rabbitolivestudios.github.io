/**
 * 1-bit drawing surface for the desk panel pages (DECISIONS #66). Text is drawn by
 * type.ts from pre-rendered glyph bitmaps of real fonts (DECISIONS #67).
 *
 * Values: 0 = black, 1 = white (as encodePNG1Bit), and 2..254 a grey tone, which only
 * pictures use (the crest and the planet, `setTone`): the panel converts those itself,
 * like every other picture since DECISIONS #81. Lines and text stay pure black and
 * white. A canvas with no tone is still encoded as 1-bit (page.ts). Every primitive clips to
 * the buffer, so callers never need to bounds-check and a miscalculated layout
 * degrades to a cut-off shape instead of an exception or a wrapped row.
 */

export const PANEL_W = 800;
export const PANEL_H = 480;

export interface Canvas {
  w: number;
  h: number;
  px: Uint8Array;
}

export function createCanvas(w: number = PANEL_W, h: number = PANEL_H): Canvas {
  return { w, h, px: new Uint8Array(w * h).fill(1) };
}

export function setPixel(c: Canvas, x: number, y: number, v: number = 0): void {
  const xi = Math.round(x);
  const yi = Math.round(y);
  if (xi < 0 || yi < 0 || xi >= c.w || yi >= c.h) return;
  c.px[yi * c.w + xi] = v;
}

/** Grey level 0 (black) to 255 (white), stored so pure black and white stay 0 and 1. */
export function toneValue(g: number): number {
  const v = Math.round(g);
  return v <= 1 ? 0 : v >= 255 ? 1 : v;
}

/** Brightness 0..255 of a stored value. */
export function toneOf(v: number): number {
  return v === 0 ? 0 : v === 1 ? 255 : v;
}

export function setTone(c: Canvas, x: number, y: number, g: number): void {
  setPixel(c, x, y, toneValue(g));
}

export function fillRect(c: Canvas, x: number, y: number, w: number, h: number, v: number = 0): void {
  const x0 = Math.max(0, Math.round(x));
  const y0 = Math.max(0, Math.round(y));
  const x1 = Math.min(c.w, Math.round(x + w));
  const y1 = Math.min(c.h, Math.round(y + h));
  for (let yy = y0; yy < y1; yy++) {
    for (let xx = x0; xx < x1; xx++) c.px[yy * c.w + xx] = v;
  }
}

export function hline(c: Canvas, x: number, y: number, len: number, t: number = 1): void {
  fillRect(c, x, y, len, t);
}

export function vline(c: Canvas, x: number, y: number, len: number, t: number = 1): void {
  fillRect(c, x, y, t, len);
}

export function strokeRect(c: Canvas, x: number, y: number, w: number, h: number, t: number = 1): void {
  hline(c, x, y, w, t);
  hline(c, x, y + h - t, w, t);
  vline(c, x, y, h, t);
  vline(c, x + w - t, y, h, t);
}

export function invertRect(c: Canvas, x: number, y: number, w: number, h: number): void {
  const x0 = Math.max(0, Math.round(x));
  const y0 = Math.max(0, Math.round(y));
  const x1 = Math.min(c.w, Math.round(x + w));
  const y1 = Math.min(c.h, Math.round(y + h));
  for (let yy = y0; yy < y1; yy++) {
    for (let xx = x0; xx < x1; xx++) {
      const i = yy * c.w + xx;
      c.px[i] = toneValue(255 - toneOf(c.px[i]));
    }
  }
}

/** Stamp a t×t square centred on (x, y); the building block for thick lines. */
function stamp(c: Canvas, x: number, y: number, t: number): void {
  const off = Math.floor((t - 1) / 2);
  fillRect(c, x - off, y - off, t, t);
}

/** Bresenham line. Steps are capped so a wild coordinate cannot spin the CPU. */
export function line(c: Canvas, x0: number, y0: number, x1: number, y1: number, t: number = 1): void {
  let x = Math.round(x0);
  let y = Math.round(y0);
  const xe = Math.round(x1);
  const ye = Math.round(y1);
  const dx = Math.abs(xe - x);
  const dy = -Math.abs(ye - y);
  const sx = x < xe ? 1 : -1;
  const sy = y < ye ? 1 : -1;
  let err = dx + dy;
  for (let i = 0; i < 4000; i++) {
    stamp(c, x, y, t);
    if (x === xe && y === ye) break;
    const e2 = 2 * err;
    if (e2 >= dy) { err += dy; x += sx; }
    if (e2 <= dx) { err += dx; y += sy; }
  }
}

/** How much ink the canvas carries, in pixels: black counts 1, a grey its share. */
export function inkCoverage(c: Canvas): number {
  let n = 0;
  for (let i = 0; i < c.px.length; i++) n += 1 - toneOf(c.px[i]) / 255;
  return n;
}

export function countBlack(c: Canvas): number {
  let n = 0;
  for (let i = 0; i < c.px.length; i++) if (c.px[i] === 0) n++;
  return n;
}
