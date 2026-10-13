/**
 * Calibration card for the desk panel (/desk/test). It shows every type size
 * and line treatment the pages use, so their legibility can be judged on the
 * real panel, where SenseCraft's cloud render and the e-ink itself decide what
 * survives (DECISIONS #48, #67). Keep it: later pages are tuned against it.
 */

import type { Canvas } from "./canvas";
import { createCanvas, fillRect, hline, strokeRect, vline } from "./canvas";
import { topBar } from "./chrome";
import type { FaceKey } from "./fonts-data";
import { brackets, dotGrid, hatch, plus, ruler } from "./parts";
import { drawType } from "./type";

const SAMPLE = "STORAGE 512 GB / 89%  Mac mini";

export function drawTestCard(): Canvas {
  const c = createCanvas();
  topBar(c, ["Desk", "Calibration"], ["IBM PLEX MONO", "AUREBESH AF"], "CARD 2");

  // A: text sizes, smallest first
  drawType(c, "plex11m", 20, 58, "A  TEXT SIZES", { spacing: 1 });
  const faces: FaceKey[] = ["plex10s", "plex11m", "plex12m", "plex12s", "plex13r", "plex13s", "plex15m", "plex15s"];
  faces.forEach((face, i) => {
    const y = 80 + i * 21;
    drawType(c, "plex10s", 20, y, face.slice(4).toUpperCase());
    drawType(c, face, 64, y, SAMPLE);
  });

  // B: large numerals
  vline(c, 404, 46, 208);
  drawType(c, "plex11m", 420, 58, "B  LARGE TYPE", { spacing: 1 });
  drawType(c, "plex28r", 420, 96, "14:30 89%");
  drawType(c, "plex36r", 420, 142, "MANDALORE");
  drawType(c, "plex50r", 420, 204, "JEDI 041");

  hline(c, 0, 254, c.w);

  // C: Aurebesh at the smaller sizes the word page uses (88 and 112 are larger still)
  drawType(c, "plex11m", 20, 274, "C  AUREBESH", { spacing: 1 });
  drawType(c, "af20", 20, 300, "JEDI");
  drawType(c, "af44", 20, 350, "JEDI");
  drawType(c, "af50", 200, 350, "JEDI");
  drawType(c, "af60", 20, 408, "JEDI");
  drawType(c, "af72", 20, 470, "JEDI");

  // D: lines and fills
  vline(c, 404, 255, 225);
  drawType(c, "plex11m", 420, 274, "D  RULES 1 / 2 / 3 PX", { spacing: 1 });
  hline(c, 420, 290, 150, 1);
  hline(c, 420, 298, 150, 2);
  hline(c, 420, 308, 150, 3);
  drawType(c, "plex11m", 600, 274, "E  DOTS + HATCH", { spacing: 1 });
  dotGrid(c, 600, 286, 90, 30, 6);
  dotGrid(c, 600, 322, 90, 30, 12);
  hatch(c, 700, 286, 80, 30, 5);
  hatch(c, 700, 322, 80, 30, 3);

  drawType(c, "plex11m", 420, 346, "F  MARKS", { spacing: 1 });
  brackets(c, 420, 360, 150, 90);
  plus(c, 495, 405);
  ruler(c, 600, 400, 180);
  fillRect(c, 600, 416, 62, 19);
  drawType(c, "plex12s", 631, 430, "ALERT", { align: "center", white: true });
  strokeRect(c, 676, 417, 45, 17);
  drawType(c, "plex10s", 698, 429, "DAILY", { align: "center" });

  return c;
}
