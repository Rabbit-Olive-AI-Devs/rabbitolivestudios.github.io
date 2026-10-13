/**
 * The Aurebesh word-of-the-day page: a key column that maps each letter used
 * to its glyph and name, the word in Aurebesh inside a full-width crosshair,
 * and three cards with the translation and where the word sits in the cycle.
 */

import { shiftDateStr } from "../date-utils";
import type { Canvas } from "./canvas";
import { createCanvas, fillRect, hline, strokeRect, vline } from "./canvas";
import { dayOfYear, groupOf, LETTER_NAMES, pickWord, WORDS } from "./aurebesh-words";
import type { Occasion } from "./aurebesh-words";
import { displayDate, topBar, TOP_BAR_BOTTOM } from "./chrome";
import type { FaceKey } from "./fonts-data";
import { brackets, dotGrid, hatch, plus, ruler } from "./parts";
import { drawType, measureType } from "./type";

const SIDEBAR_W = 176;
/** Centre and half-size of the bracketed box the word sits in. */
const WORD_CX = 488;
const WORD_CY = 186;
const WORD_HALF_W = 250;
const WORD_HALF_H = 78;
const WORD_MAX_W = 480;
const CARDS_TOP = 334;

/** Largest first. The smallest must hold nine of the widest letter (WWWWWWWWW). */
const WORD_FACES: readonly FaceKey[] = ["af112", "af88", "af72", "af60", "af50", "af44"];

/** The largest Aurebesh face in which the word fits its box. */
export function wordFace(word: string): FaceKey {
  return WORD_FACES.find((f) => measureType(f, word) <= WORD_MAX_W) ?? "af44";
}

const pad3 = (n: number) => String(n).padStart(3, "0");
const label = (c: Canvas, x: number, y: number, s: string) => drawType(c, "plex11m", x, y, s, { spacing: 1 });

/** A label on the left and its value on the right, over a hairline. */
function dataRow(c: Canvas, x: number, y: number, w: number, key: string, value: string, rule: boolean): void {
  label(c, x, y, key);
  drawType(c, "plex12s", x + w, y, value, { align: "right" });
  if (rule) hline(c, x, y + 6, w);
}

/**
 * `occasion` marks a birthday or special day: its tag is shown in the corner of
 * the view, and its count, if any, takes the place of the word's number.
 */
export function drawAurebeshPage(word: string, dateStr: string, occasion?: Occasion): Canvas {
  const c = createCanvas();
  const index = WORDS.indexOf(word);
  const position = index < 0 ? "---" : pad3(index + 1);
  const group = occasion?.group ?? groupOf(index);
  const unique = [...new Set(word)];

  topBar(
    c, ["Desk", "Aurebesh", "Word of the day"],
    [displayDate(dateStr), `DAY ${dayOfYear(dateStr)}`], `${position}/${WORDS.length}`,
  );

  // Key column
  vline(c, SIDEBAR_W, TOP_BAR_BOTTOM + 1, c.h - TOP_BAR_BOTTOM - 1);
  drawType(c, "plex15m", 20, 64, "[AUREBESH]");
  fillRect(c, 150, 54, 12, 9);
  dataRow(c, 20, 92, 140, "LETTERS", String(word.length), false);
  dataRow(c, 20, 112, 140, "UNIQUE", String(unique.length), false);
  dataRow(c, 20, 132, 140, "GROUP", group, false);
  hatch(c, 12, 148, 152, 9);
  drawType(c, "plex13r", 20, 182, "KEY");
  const rowStep = unique.length > 8 ? 27 : 29;
  unique.forEach((ch, i) => {
    const y = 214 + i * rowStep;
    drawType(c, "af20", 22, y, ch);
    drawType(c, "plex15s", 74, y, ch);
    label(c, 100, y - 1, LETTER_NAMES[ch] ?? "");
    hline(c, 20, y + 9, 140);
  });

  // The word, inside a crosshair that spans the whole view
  const boxX = WORD_CX - WORD_HALF_W;
  const boxY = WORD_CY - WORD_HALF_H;
  const clear = 12;
  dotGrid(c, 190, 50, 602, 276, 12, {
    x: boxX - clear, y: boxY - clear, w: WORD_HALF_W * 2 + clear * 2 + 1, h: WORD_HALF_H * 2 + clear * 2 + 1,
  });
  hline(c, SIDEBAR_W + 1, WORD_CY, boxX - clear - SIDEBAR_W - 1);
  hline(c, WORD_CX + WORD_HALF_W + clear, WORD_CY, c.w - (WORD_CX + WORD_HALF_W + clear));
  vline(c, WORD_CX, TOP_BAR_BOTTOM + 1, boxY - clear - TOP_BAR_BOTTOM - 1);
  vline(c, WORD_CX, WORD_CY + WORD_HALF_H + clear, CARDS_TOP - (WORD_CY + WORD_HALF_H + clear));
  brackets(c, boxX, boxY, WORD_HALF_W * 2 + 1, WORD_HALF_H * 2 + 1);

  const face = wordFace(word);
  const faceSize = Number(face.slice(2));
  drawType(c, face, WORD_CX, WORD_CY + Math.round(faceSize * 0.36), word, { align: "center" });

  if (occasion) {
    const tagW = measureType("plex12s", occasion.tag) + 16;
    fillRect(c, 190, 44, tagW, 19);
    drawType(c, "plex12s", 198, 58, occasion.tag, { white: true });
  } else {
    label(c, 190, 56, "TARGET");
  }
  drawType(c, "plex11m", 786, 56, "CANON", { align: "right", spacing: 1 });
  ruler(c, WORD_CX - 150, 296, 300);
  const caption = unique.length <= 5
    ? unique.map((ch) => LETTER_NAMES[ch] ?? ch).join("  /  ")
    : `${word.length} LETTERS  /  ${unique.length} UNIQUE`;
  // the caption sits on the crosshair line, so clear a strip for it first
  const captionW = measureType("plex11m", caption, 1);
  fillRect(c, WORD_CX - captionW / 2 - 6, 304, captionW + 12, 18, 1);
  drawType(c, "plex11m", WORD_CX, 316, caption, { align: "center", spacing: 1 });

  // Cards
  hline(c, SIDEBAR_W + 1, CARDS_TOP, c.w - SIDEBAR_W - 1);
  hatch(c, 392, 342, 7, 130);
  hatch(c, 596, 342, 7, 130);
  for (const [x0, x1] of [[190, 384], [410, 588], [614, 788]]) {
    for (const y of [348, 468]) { plus(c, x0, y); plus(c, x1, y); }
  }

  label(c, 204, 362, "TRANSLATION");
  const latinFace: FaceKey = word.length <= 5 ? "plex50r" : word.length <= 7 ? "plex36r" : "plex28r";
  drawType(c, latinFace, 204, 418, word);
  dataRow(c, 204, 446, 170, "LANG", "BASIC", true);

  label(c, 424, 362, "WORD DATA");
  [["LETTERS", String(word.length)], ["UNIQUE", String(unique.length)], ["GROUP", group], ["INDEX", position]]
    .forEach(([key, value], i) => dataRow(c, 424, 386 + i * 21, 154, key, value, true));

  const count = occasion?.count;
  label(c, 628, 362, count ? count.label : "CYCLE");
  strokeRect(c, 738, 350, 45, 17);
  drawType(c, "plex10s", 760, 362, "DAILY", { align: "center" });
  // a number in Aurebesh numerals with the same number in Latin beside it: the
  // occasion's count if it has one, otherwise the word's place in the cycle
  const value = count ? count.value : index + 1;
  const number = value > 0 ? String(value) : "--";
  const numberW = value > 0 ? drawType(c, "af36", 628, 410, number) : 0;
  const latinX = 628 + numberW + (numberW > 0 ? 10 : 0);
  drawType(c, "plex15s", latinX, 394, number);
  if (!count) drawType(c, "plex11m", latinX + measureType("plex15s", number) + 3, 394, `/${WORDS.length}`);
  label(c, latinX, 410, count ? count.unit : "WORDS");
  // word lengths for the ten days before today and the nineteen after; today is the wide bar
  for (let i = 0; i < 30; i++) {
    const h = pickWord(shiftDateStr(dateStr, i - 10)).length * 3;
    fillRect(c, 628 + i * 5, 458 - h, i === 10 ? 3 : 2, h);
  }

  return c;
}
