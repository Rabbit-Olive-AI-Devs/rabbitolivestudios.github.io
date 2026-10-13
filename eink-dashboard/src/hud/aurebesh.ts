/**
 * The Aurebesh check sheet (/desk/aurebesh?sheet): all 26 letters and the ten
 * numerals in Aurebesh AF, each with its Latin character, for comparing against
 * a chart.
 *
 * The 26 single letters only; the digraph letters (CH, SH, TH and others) are
 * dated and left out. The glyphs come from Aurebesh AF by AurekFonts, released
 * as CC0 (see fonts/desk/README.md). Aurebesh is the Star Wars alphabet; the
 * use here is personal and non-commercial.
 */

import type { Canvas } from "./canvas";
import { createCanvas, hline } from "./canvas";
import { LETTER_NAMES, WORDS } from "./aurebesh-words";
import { topBar } from "./chrome";
import { plus } from "./parts";
import { drawType } from "./type";

export function drawAurebeshSheet(): Canvas {
  const c = createCanvas();
  topBar(c, ["Desk", "Aurebesh", "Check sheet"], ["AUREBESH AF", "CANON"], `${WORDS.length} WORDS`);
  const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const rowH = 104;
  /** One cell: the Aurebesh glyph over a rule, its Latin character, and an optional name. */
  const cell = (cx: number, y: number, ch: string, name?: string) => {
    drawType(c, "af50", cx, y + 40, ch, { align: "center" });
    hline(c, cx - 28, y + 50, 57);
    drawType(c, "plex15s", cx, y + 70, ch, { align: "center" });
    if (name) drawType(c, "plex10s", cx, y + 85, name, { align: "center", spacing: 1 });
  };
  for (let i = 0; i < letters.length; i++) {
    cell(64 + (i % 9) * 84, 50 + Math.floor(i / 9) * rowH, letters[i], LETTER_NAMES[letters[i]]);
  }
  const digits = "0123456789";
  for (let i = 0; i < digits.length; i++) cell(58 + i * 76, 50 + 3 * rowH, digits[i]);
  for (const [x, y] of [[20, 48], [780, 48], [20, 468], [780, 468]]) plus(c, x, y);
  return c;
}
