/**
 * Calibration card for the desk panel (/desk/test). It puts the open style
 * questions side by side so they can be judged on the real panel, where
 * SenseCraft's cloud render and the e-ink itself decide what is legible
 * (DECISIONS #48). Keep it: later pages are tuned against it.
 */

import type { Canvas } from "./canvas";
import { createCanvas, fillRect, hline, text, vline } from "./canvas";
import { accentBlock, bigNumber, brackets, crosshair, dotFill, frame, hatch, ruler, titleBar } from "./parts";
import { drawStrokeText, LATIN } from "./stroke";

export function drawTestCard(): Canvas {
  const c = createCanvas();
  frame(c);
  titleBar(c, "DESK PANEL / CALIBRATION", "CARD 1");

  // Column 1: the 8x8 bitmap font at three sizes.
  text(c, 36, 66, "A  8X8 FONT", 1);
  text(c, 36, 82, "1X  STORAGE 512 GB / 89%", 1);
  text(c, 36, 98, "2X  STORAGE 89%", 2);
  text(c, 36, 122, "3X  STORAGE", 3);

  // Column 2: stroke text at three sizes and thicknesses.
  text(c, 420, 66, "B  STROKE FONT", 1);
  drawStrokeText(c, LATIN, 420, 82, "ORION 0123456789", 3, 2);
  drawStrokeText(c, LATIN, 420, 110, "ORION 89%", 5, 3);
  drawStrokeText(c, LATIN, 420, 150, "14:30", 9, 4);

  hline(c, 36, 214, 728);

  // Row 2: rule thickness.
  text(c, 36, 224, "C  RULES 1 / 2 / 3 PX", 1);
  hline(c, 36, 242, 150, 1);
  hline(c, 36, 252, 150, 2);
  hline(c, 36, 264, 150, 3);
  vline(c, 200, 240, 28, 1);
  vline(c, 210, 240, 28, 2);
  vline(c, 222, 240, 28, 3);

  // Row 2: dot fills from light to solid.
  text(c, 260, 224, "D  DOT FILL 12/25/50/75/100", 1);
  const levels = [0.125, 0.25, 0.5, 0.75, 1];
  levels.forEach((lv, i) => dotFill(c, 260 + i * 60, 240, 52, 30, lv));

  // Row 2: hatch band and accent block.
  text(c, 580, 224, "E  HATCH + ACCENT", 1);
  hatch(c, 580, 240, 90, 30, 6);
  accentBlock(c, 684, 240, "CRIT", 2);

  hline(c, 36, 284, 728);

  // Row 3: the composed look.
  text(c, 36, 294, "F  COMPOSED", 1);
  brackets(c, 36, 312, 220, 130);
  crosshair(c, 146, 377, 34);
  dotFill(c, 124, 355, 44, 44, 0.5);

  bigNumber(c, 290, 316, "14", "VSL");
  bigNumber(c, 290, 384, "89", "PCT");
  ruler(c, 420, 440, 330);
  text(c, 420, 316, "TARGET INSPECTOR", 2);
  text(c, 420, 344, "VESSEL  MV ORION", 1);
  text(c, 420, 358, "SPEED   9.4 KN", 1);
  text(c, 420, 372, "HDG     011 T", 1);
  fillRect(c, 420, 392, 60, 6);
  accentBlock(c, 620, 340, "GAP 18M", 2);

  return c;
}
