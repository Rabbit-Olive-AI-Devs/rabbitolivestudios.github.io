/**
 * The 26-letter Aurebesh check sheet (/desk/aurebesh?sheet): every letter in
 * Aurebesh AF with its Latin letter and name, for comparing against a chart.
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
  const cols = 9;
  const cellW = 84;
  const cellH = 138;
  for (let i = 0; i < letters.length; i++) {
    const cx = 64 + (i % cols) * cellW;
    const y = 60 + Math.floor(i / cols) * cellH;
    drawType(c, "af50", cx, y + 52, letters[i], { align: "center" });
    hline(c, cx - 30, y + 66, 61);
    drawType(c, "plex15s", cx, y + 88, letters[i], { align: "center" });
    drawType(c, "plex10s", cx, y + 106, LETTER_NAMES[letters[i]], { align: "center", spacing: 1 });
  }
  for (const [x, y] of [[20, 52], [780, 52], [20, 466], [780, 466]]) plus(c, x, y);
  return c;
}
