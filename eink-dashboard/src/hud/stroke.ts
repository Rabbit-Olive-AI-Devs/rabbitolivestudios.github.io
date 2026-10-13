/**
 * Stroke fonts: glyphs as polylines on a small grid, drawn with the canvas line
 * primitive. They scale to any size with a chosen thickness, need no font file,
 * and contain no grey — the properties the mono panel needs (DECISIONS #48, #66).
 *
 * Glyph syntax: polylines separated by "|", points as "x,y" separated by spaces,
 * y growing downward. A single-point polyline is a dot.
 */

import type { Canvas } from "./canvas";
import { line } from "./canvas";

export interface StrokeFont {
  /** Grid width and height of one glyph, in grid units. */
  w: number;
  h: number;
  /** Space between glyphs, in grid units. */
  gap: number;
  glyphs: Record<string, string>;
}

export function parseGlyph(g: string): number[][][] {
  return g.split("|").map((poly) =>
    poly.trim().split(/\s+/).map((p) => p.split(",").map(Number)));
}

export function strokeTextWidth(font: StrokeFont, s: string, cell: number): number {
  if (s.length === 0) return 0;
  return (s.length * font.w + (s.length - 1) * font.gap) * cell;
}

/**
 * Draw `s` with its top-left at (x, y). `cell` is pixels per grid unit and `t`
 * the line thickness in pixels. Characters with no glyph (including the space)
 * advance without drawing.
 */
export function drawStrokeText(
  c: Canvas, font: StrokeFont, x: number, y: number, s: string, cell: number, t: number,
): void {
  let gx = x;
  for (const ch of s) {
    const g = font.glyphs[ch];
    if (g) {
      for (const poly of parseGlyph(g)) {
        if (poly.length === 1) {
          line(c, gx + poly[0][0] * cell, y + poly[0][1] * cell, gx + poly[0][0] * cell, y + poly[0][1] * cell, t + 1);
          continue;
        }
        for (let i = 0; i < poly.length - 1; i++) {
          line(c, gx + poly[i][0] * cell, y + poly[i][1] * cell,
            gx + poly[i + 1][0] * cell, y + poly[i + 1][1] * cell, t);
        }
      }
    }
    gx += (font.w + font.gap) * cell;
  }
}

const O = "1,0 3,0 4,1 4,5 3,6 1,6 0,5 0,1 1,0";
const P = "0,6 0,0 3,0 4,1 4,2 3,3 0,3";

/** Angular capitals and digits on a 4x6 grid. Curves are chamfered corners. */
export const LATIN: StrokeFont = {
  w: 4,
  h: 6,
  gap: 2,
  glyphs: {
    A: "0,6 0,2 2,0 4,2 4,6|0,4 4,4",
    B: "0,0 0,6 3,6 4,5 4,4 3,3 0,3|3,3 4,2 4,1 3,0 0,0",
    C: "4,1 3,0 1,0 0,1 0,5 1,6 3,6 4,5",
    D: "0,0 0,6 3,6 4,5 4,1 3,0 0,0",
    E: "4,0 0,0 0,6 4,6|0,3 3,3",
    F: "4,0 0,0 0,6|0,3 3,3",
    G: "4,1 3,0 1,0 0,1 0,5 1,6 3,6 4,5 4,3 2,3",
    H: "0,0 0,6|4,0 4,6|0,3 4,3",
    I: "1,0 3,0|2,0 2,6|1,6 3,6",
    J: "4,0 4,5 3,6 1,6 0,5",
    K: "0,0 0,6|4,0 0,3 4,6",
    L: "0,0 0,6 4,6",
    M: "0,6 0,0 2,3 4,0 4,6",
    N: "0,6 0,0 4,6 4,0",
    O,
    P,
    Q: `${O}|2,4 4,6`,
    R: `${P}|2,3 4,6`,
    S: "4,1 3,0 1,0 0,1 0,2 1,3 3,3 4,4 4,5 3,6 1,6 0,5",
    T: "0,0 4,0|2,0 2,6",
    U: "0,0 0,5 1,6 3,6 4,5 4,0",
    V: "0,0 2,6 4,0",
    W: "0,0 1,6 2,3 3,6 4,0",
    X: "0,0 4,6|4,0 0,6",
    Y: "0,0 2,3 4,0|2,3 2,6",
    Z: "0,0 4,0 0,6 4,6",
    "0": `${O}|1,5 3,1`,
    "1": "1,1 2,0 2,6|1,6 3,6",
    "2": "0,1 1,0 3,0 4,1 4,2 0,6 4,6",
    "3": "0,1 1,0 3,0 4,1 4,2 3,3 4,4 4,5 3,6 1,6 0,5|2,3 3,3",
    "4": "3,6 3,0 0,4 4,4",
    "5": "4,0 0,0 0,3 3,3 4,4 4,5 3,6 1,6 0,5",
    "6": "4,1 3,0 1,0 0,1 0,5 1,6 3,6 4,5 4,4 3,3 0,3",
    "7": "0,0 4,0 1,6",
    "8": "1,3 0,2 0,1 1,0 3,0 4,1 4,2 3,3 1,3 0,4 0,5 1,6 3,6 4,5 4,4 3,3",
    "9": "4,3 1,3 0,2 0,1 1,0 3,0 4,1 4,5 3,6 1,6 0,5",
    "-": "1,3 3,3",
    ".": "2,6",
    ":": "2,2|2,5",
    "/": "0,6 4,0",
    "%": "0,6 4,0|0,0 1,0 1,1 0,1 0,0|3,5 4,5 4,6 3,6 3,5",
  },
};
