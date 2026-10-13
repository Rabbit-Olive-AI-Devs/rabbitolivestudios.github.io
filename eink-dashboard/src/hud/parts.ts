/**
 * The reusable pieces of the desk panel's "targeting HUD" look. Every function
 * draws black on the white canvas; nothing here knows about any specific page.
 */

import type { Canvas } from "./canvas";
import {
  circle, fillRect, hline, invertRect, setPixel, strokeRect, text, textWidth, vline,
} from "./canvas";
import { drawStrokeText, LATIN, strokeTextWidth } from "./stroke";
import { LABEL_SCALE, MARGIN, RULE, SMALL_SCALE, STROKE_BIG } from "./style";

/** A "+" registration mark centred on (x, y). */
export function plus(c: Canvas, x: number, y: number, arm: number = 4): void {
  hline(c, x - arm, y, arm * 2 + 1);
  vline(c, x, y - arm, arm * 2 + 1);
}

/** Outer border inset by MARGIN, with a "+" inside each corner. */
export function frame(c: Canvas): void {
  strokeRect(c, MARGIN, MARGIN, c.w - MARGIN * 2, c.h - MARGIN * 2, RULE);
  const inset = MARGIN + 12;
  plus(c, inset, inset);
  plus(c, c.w - inset - 1, inset);
  plus(c, inset, c.h - inset - 1);
  plus(c, c.w - inset - 1, c.h - inset - 1);
}

/** Corner brackets around a box, leaving the middle of each edge open. */
export function brackets(
  c: Canvas, x: number, y: number, w: number, h: number, len: number = 18, t: number = 2,
): void {
  hline(c, x, y, len, t); vline(c, x, y, len, t);
  hline(c, x + w - len, y, len, t); vline(c, x + w - t, y, len, t);
  hline(c, x, y + h - t, len, t); vline(c, x, y + h - len, len, t);
  hline(c, x + w - len, y + h - t, len, t); vline(c, x + w - t, y + h - len, len, t);
}

/** A circle with four tick lines that stop `gap` pixels short of the centre. */
export function crosshair(c: Canvas, cx: number, cy: number, r: number, gap: number = 6): void {
  circle(c, cx, cy, r);
  hline(c, cx - r - 10, cy, r + 10 - gap);
  hline(c, cx + gap + 1, cy, r + 10 - gap);
  vline(c, cx, cy - r - 10, r + 10 - gap);
  vline(c, cx, cy + gap + 1, r + 10 - gap);
}

/** Horizontal ruler: a baseline at `y` with ticks rising above it. */
export function ruler(
  c: Canvas, x: number, y: number, len: number, step: number = 10, majorEvery: number = 5,
): void {
  hline(c, x, y, len);
  for (let i = 0, n = 0; i <= len; i += step, n++) {
    const tick = n % majorEvery === 0 ? 8 : 4;
    vline(c, x + i, y - tick, tick);
  }
}

/** Diagonal stripes, one pixel wide, every `step` pixels. */
export function hatch(c: Canvas, x: number, y: number, w: number, h: number, step: number = 6): void {
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      if ((xx + yy) % step === 0) setPixel(c, x + xx, y + yy);
    }
  }
}

const BAYER4 = [
  [0, 8, 2, 10],
  [12, 4, 14, 6],
  [3, 11, 1, 9],
  [15, 7, 13, 5],
];

/** Ordered-dither fill. `level` runs from 0 (white) to 1 (solid black). */
export function dotFill(c: Canvas, x: number, y: number, w: number, h: number, level: number): void {
  const threshold = Math.max(0, Math.min(1, level)) * 16;
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      if (BAYER4[(y + yy) & 3][(x + xx) & 3] < threshold) setPixel(c, x + xx, y + yy);
    }
  }
}

/** Page title on the left, a short stamp on the right, and a rule beneath. */
export function titleBar(c: Canvas, left: string, right: string): void {
  const x = MARGIN + 28;
  const y = MARGIN + 16;
  text(c, x, y, left, LABEL_SCALE);
  text(c, c.w - x - textWidth(right, LABEL_SCALE), y, right, LABEL_SCALE);
  hline(c, x, y + 8 * LABEL_SCALE + 8, c.w - x * 2, RULE);
}

/**
 * The one accent this look has on a mono panel: a solid black block with white
 * text. Returns the block's width so callers can place what follows.
 */
export function accentBlock(c: Canvas, x: number, y: number, label: string, scale: number = LABEL_SCALE): number {
  const w = textWidth(label, scale) + 12;
  const h = 8 * scale + 8;
  fillRect(c, x, y, w, h, 1);
  text(c, x + 6, y + 4, label, scale);
  invertRect(c, x, y, w, h);
  return w;
}

/** A large stroke-drawn value with a small unit caption to its right. */
export function bigNumber(c: Canvas, x: number, y: number, value: string, unit: string, cell: number = 8): void {
  drawStrokeText(c, LATIN, x, y, value, cell, STROKE_BIG);
  text(c, x + strokeTextWidth(LATIN, value, cell) + 8, y, unit, SMALL_SCALE);
}
