/**
 * The reusable pieces of the desk panel's "targeting HUD" look: fine rules,
 * corner marks, a sparse dot field, striped dividers. Every function draws black
 * on the white canvas and does work bounded by the canvas, never by its
 * arguments, so a page can pass it numbers that came from outside.
 */

import type { Canvas } from "./canvas";
import { hline, setPixel, vline } from "./canvas";

/** The part of a box that lies on the canvas, as whole-pixel bounds. */
function clip(c: Canvas, x: number, y: number, w: number, h: number) {
  return {
    x0: Math.max(0, Math.round(x)),
    y0: Math.max(0, Math.round(y)),
    x1: Math.min(c.w, Math.round(x + w)),
    y1: Math.min(c.h, Math.round(y + h)),
  };
}

/** A "+" registration mark centred on (x, y). */
export function plus(c: Canvas, x: number, y: number, arm: number = 4): void {
  hline(c, x - arm, y, arm * 2 + 1);
  vline(c, x, y - arm, arm * 2 + 1);
}

/** Hairline corner brackets around a box, leaving the middle of each edge open. */
export function brackets(c: Canvas, x: number, y: number, w: number, h: number, len: number = 22): void {
  hline(c, x, y, len); vline(c, x, y, len);
  hline(c, x + w - len, y, len); vline(c, x + w - 1, y, len);
  hline(c, x, y + h - 1, len); vline(c, x, y + h - len, len);
  hline(c, x + w - len, y + h - 1, len); vline(c, x + w - 1, y + h - len, len);
}

/** Horizontal ruler: a baseline at `y` with ticks rising above it. */
export function ruler(
  c: Canvas, x: number, y: number, len: number, step: number = 10, majorEvery: number = 5,
): void {
  if (step <= 0) return;
  const end = Math.min(len, c.w);
  hline(c, x, y, end + 1);
  for (let i = 0, n = 0; i <= end; i += step, n++) {
    const tick = n % majorEvery === 0 ? 8 : 4;
    vline(c, x + i, y - tick, tick);
  }
}

/** Diagonal stripes, one pixel wide, every `step` pixels. */
export function hatch(c: Canvas, x: number, y: number, w: number, h: number, step: number = 5): void {
  const b = clip(c, x, y, w, h);
  const ox = Math.round(x);
  const oy = Math.round(y);
  for (let yy = b.y0; yy < b.y1; yy++) {
    for (let xx = b.x0; xx < b.x1; xx++) {
      if ((((xx - ox) + (yy - oy)) % step + step) % step === 0) setPixel(c, xx, yy);
    }
  }
}

export interface Box { x: number; y: number; w: number; h: number }

/** Single-pixel dots every `step` pixels, skipping the optional keep-out box. */
export function dotGrid(
  c: Canvas, x: number, y: number, w: number, h: number, step: number = 12, keepOut?: Box,
): void {
  if (step <= 0) return;
  const b = clip(c, x, y, w, h);
  const ox = Math.round(x);
  const oy = Math.round(y);
  const first = (lo: number, origin: number) => lo + ((((origin - lo) % step) + step) % step);
  for (let yy = first(b.y0, oy); yy < b.y1; yy += step) {
    for (let xx = first(b.x0, ox); xx < b.x1; xx += step) {
      if (keepOut && xx >= keepOut.x && xx < keepOut.x + keepOut.w && yy >= keepOut.y && yy < keepOut.y + keepOut.h) continue;
      setPixel(c, xx, yy);
    }
  }
}

/** A circle, filled or as a one-pixel outline. */
export function disc(c: Canvas, cx: number, cy: number, r: number, filled: boolean): void {
  if (!Number.isFinite(r) || r <= 0) return;
  const radius = Math.min(r, c.w + c.h);
  const b = clip(c, cx - radius, cy - radius, radius * 2 + 1, radius * 2 + 1);
  const outer = radius * radius;
  const inner = (radius - 1) * (radius - 1);
  for (let y = b.y0; y < b.y1; y++) {
    for (let x = b.x0; x < b.x1; x++) {
      const d = (x - cx) * (x - cx) + (y - cy) * (y - cy);
      if (d <= outer && (filled || d > inner)) setPixel(c, x, y);
    }
  }
}
