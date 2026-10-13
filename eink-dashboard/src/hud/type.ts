/**
 * Text for the desk panel pages, drawn from the 1-bit glyph bitmaps that
 * `npm run desk-fonts` rasterises out of real fonts (DECISIONS #67). The Worker
 * only copies pixels, so the type is a real typeface and still has no grey
 * edge for SenseCraft's cloud renderer to blur (DECISIONS #48).
 */

import type { Canvas } from "./canvas";
import { setPixel } from "./canvas";
import { FACES } from "./fonts-data";
import type { FaceKey } from "./fonts-data";

export interface TypeOptions {
  /** Which part of the text sits at `x`. Default: left. */
  align?: "left" | "center" | "right";
  /** Extra pixels between characters (letter-spacing). */
  spacing?: number;
  /** Draw white instead of black, for text on a filled block. */
  white?: boolean;
}

/** Decoded bitmaps, kept for the life of the isolate. */
const decoded = new Map<string, Uint8Array>();

function bitmap(face: FaceKey, ch: string): { a: number; bits: Uint8Array } | null {
  const g = FACES[face].glyphs[ch];
  if (!g) return null;
  const key = face + ch;
  let bits = decoded.get(key);
  if (!bits) {
    const bin = atob(g.d);
    bits = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bits[i] = bin.charCodeAt(i);
    decoded.set(key, bits);
  }
  return { a: g.a, bits };
}

/** Width of `s` in pixels. Characters the face has no glyph for take no room. */
export function measureType(face: FaceKey, s: string, spacing: number = 0): number {
  let width = 0;
  let drawn = 0;
  for (const ch of s) {
    const g = FACES[face].glyphs[ch];
    if (!g) continue;
    width += g.a;
    drawn++;
  }
  return drawn === 0 ? 0 : width + spacing * (drawn - 1);
}

/**
 * Draw `s` with its baseline at `baseline`. Returns the width drawn. Characters
 * the face has no glyph for are skipped; anything off the canvas is clipped.
 */
export function drawType(
  c: Canvas, face: FaceKey, x: number, baseline: number, s: string, opts: TypeOptions = {},
): number {
  const spacing = opts.spacing ?? 0;
  const width = measureType(face, s, spacing);
  const { height, baseline: ascent } = FACES[face];
  const top = Math.round(baseline) - ascent;
  const value = opts.white ? 1 : 0;
  let penX = Math.round(opts.align === "right" ? x - width : opts.align === "center" ? x - width / 2 : x);
  for (const ch of s) {
    const g = bitmap(face, ch);
    if (!g) continue;
    for (let y = 0; y < height; y++) {
      for (let xx = 0; xx < g.a; xx++) {
        const n = y * g.a + xx;
        if (g.bits[n >> 3] & (0x80 >> (n & 7))) setPixel(c, penX + xx, top + y, value);
      }
    }
    penX += g.a + spacing;
  }
  return width;
}
