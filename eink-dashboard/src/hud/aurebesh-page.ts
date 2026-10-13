/**
 * The Aurebesh word-of-the-day page: the word in large Aurebesh glyphs, its
 * English spelling beneath, and a key strip that maps each letter used to its
 * glyph so the word can be decoded by eye.
 */

import type { Canvas } from "./canvas";
import { createCanvas, hline, text } from "./canvas";
import { AUREBESH } from "./aurebesh";
import { accentBlock, brackets, frame, titleBar } from "./parts";
import { drawStrokeText, LATIN, strokeTextWidth } from "./stroke";
import { RULE, STROKE_MED } from "./style";

const WORD_BOX_X = 60;
const WORD_BOX_Y = 78;
const WORD_BOX_W = 680;
const WORD_BOX_H = 190;
const WORD_MAX_W = 620;
const MAX_CELL = 30;

/** Pixels per grid unit so the Aurebesh word fits its box, capped for short words. */
export function fitCell(word: string): number {
  return Math.min(MAX_CELL, Math.floor(WORD_MAX_W / strokeTextWidth(AUREBESH, word, 1)));
}

export function drawAurebeshPage(word: string, dateStr: string): Canvas {
  const c = createCanvas();
  frame(c);
  titleBar(c, "AUREBESH / WORD OF THE DAY", dateStr);

  brackets(c, WORD_BOX_X, WORD_BOX_Y, WORD_BOX_W, WORD_BOX_H);
  const cell = fitCell(word);
  const wordW = strokeTextWidth(AUREBESH, word, cell);
  const wordH = AUREBESH.h * cell;
  drawStrokeText(
    c, AUREBESH,
    Math.round((c.w - wordW) / 2),
    WORD_BOX_Y + Math.round((WORD_BOX_H - wordH) / 2),
    word, cell, Math.max(STROKE_MED, Math.round(cell * 0.4)),
  );

  const latinCell = 7;
  drawStrokeText(
    c, LATIN, Math.round((c.w - strokeTextWidth(LATIN, word, latinCell)) / 2), 288,
    word, latinCell, STROKE_MED,
  );

  hline(c, 36, 348, c.w - 72, RULE);
  const keyX = 36 + accentBlock(c, 36, 360, "KEY") + 20;
  const letters = [...new Set(word.split(""))];
  const slot = Math.min(72, Math.floor((c.w - 36 - keyX) / letters.length));
  letters.forEach((ch, i) => {
    const x = keyX + i * slot;
    const keyCell = 6;
    const glyphX = x + Math.round((AUREBESH.w * keyCell - strokeTextWidth(AUREBESH, ch, keyCell)) / 2);
    drawStrokeText(c, AUREBESH, glyphX, 364, ch, keyCell, STROKE_MED);
    text(c, x + 10, 400, ch, 2);
  });

  return c;
}
