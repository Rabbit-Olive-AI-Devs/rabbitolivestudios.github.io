/**
 * Aurebesh, the Star Wars alphabet, as stroke data for the desk panel.
 *
 * The 26 single letters only; the digraph letters (CH, SH, TH and others) are
 * left out. Letters are wider than tall and vary in width, so each glyph keeps
 * its true proportions, centred in a 6x4 cell. The letterforms originate from Star Wars. The stroke data below was
 * drawn for this project against a public reference chart; no font file or film
 * artwork is in the repository, and the use is personal and non-commercial.
 */

import type { Canvas } from "./canvas";
import { createCanvas, text } from "./canvas";
import { frame, titleBar } from "./parts";
import type { StrokeFont } from "./stroke";
import { drawStrokeText } from "./stroke";

export const AUREBESH: StrokeFont = {
  w: 6,
  h: 4,
  gap: 2,
  glyphs: {
    A: "0.4,0 0.4,1.2 3,1.2 5.7,0.2|0.4,4 0.4,2.8 3,2.8 5.7,3.8",
    B: "0.97,1.53 1.81,0.47 4.19,0.47 5.03,1.53|0.97,2.47 1.81,3.53 4.19,3.53 5.03,2.47|2.41,2 3.59,2",
    C: "1.91,0.2 1.91,2.53|2.95,1.47 2.95,2.53|4.07,1.47 4.07,3.87",
    D: "1.3,0.4 4.55,0.4 1.73,3.87|1.45,2.2 3.28,2.2",
    E: "0.95,0.13 2.18,3.87 3.57,0.33 5.13,0.33 5.13,3.87|4.15,0.33 4.15,3.87",
    F: "0.56,3.67 5.63,3.67|3,0.2 3,3.67|0.56,3.67 0.56,1.47 3,1.33 4.5,0.93 5.63,0.47",
    G: "0.97,0.13 0.97,3.67 3.49,3.67 5.11,0.33 2.19,0.33 2.19,1.07",
    H: "0.52,0.33 5.48,0.33|1.23,2 4.77,2|0.52,3.67 5.48,3.67",
    I: "3.5,0.2 3.5,3.87|3.5,0.33 2.22,1.27",
    J: "0.52,3.67 3.53,3.67 5.31,0.2|0.52,2 2.11,1.87 3.53,1.33 4.6,0.93",
    K: "0.87,0.33 4.9,0.33 4.9,3.67 0.87,3.67",
    L: "4.25,0.13 4.25,3.73 1.75,2.27",
    M: "4.44,0.4 3.62,0.4 1.87,3.67 4.44,3.67",
    N: "1.46,0.13 0.87,1 0.78,2 1.04,3.2 1.63,3.87|1.63,3.33 3.17,0.27 5.22,3.87",
    O: "2.15,0.33 4.02,0.33 5.2,3.27 4.86,3.67 1.14,3.67 0.8,3.27 2.15,0.33",
    P: "3.11,0.4 2.23,0.4 1.73,1.2 1.67,2.4 1.89,3.2 2.34,3.67 4.33,3.67 4.33,0.13",
    Q: "4.82,2.13 4.82,0.33 1.1,0.33 1.1,3.6 2.85,3.6 3.53,3.27",
    R: "1.23,0.4 4.7,0.4 1.59,3.87",
    S: "1.08,0.13 5.1,3.33|5.1,0.13 5.1,3.87|0.73,2.8 2.83,3.47",
    T: "3,0.13 3,3.87|0.73,2.07 3,3.8 5.27,2.07",
    U: "1.12,0.13 1.12,3.67 4.88,3.67 4.88,0.4 3.75,0.4 2.32,1.33",
    V: "1.52,0.13 3,1.6 4.48,0.13|3,1.6 3,3.87",
    W: "0.78,0.4 5.22,0.4 5.22,3.6 0.78,3.6 0.78,0.4",
    X: "3,0.27 4.8,3.6 1.2,3.6 3,0.27",
    Y: "2.27,0.33 1.18,0.33 2.93,3.8 4.82,0.2|2.27,0.33 2.93,1.2",
    Z: "1.04,3.2 1.04,3.67 4.81,3.67 4.81,0.2|4.81,1.13 3,1.13 1.49,2.27",
  },
};

/** All 26 letters with their Latin equivalents, for checking against a chart. */
export function drawAurebeshSheet(): Canvas {
  const c = createCanvas();
  frame(c);
  titleBar(c, "AUREBESH / CHECK SHEET", "A-Z");
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const cols = 9;
  const cellW = 82;
  const cellH = 130;
  for (let i = 0; i < letters.length; i++) {
    const x = 40 + (i % cols) * cellW;
    const y = 70 + Math.floor(i / cols) * cellH;
    drawStrokeText(c, AUREBESH, x, y, letters[i], 9, 3);
    text(c, x + 20, y + 66, letters[i], 2);
  }
  return c;
}
