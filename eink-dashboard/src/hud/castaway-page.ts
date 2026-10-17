/**
 * Layout of the castaway page (DECISIONS #93): the island across the top, the story below it,
 * and a two-line log on the right. Desk style: real fonts, nothing under plex11m.
 */

import type { CastawayView } from "../castaway";
import type { Canvas } from "./canvas";
import { createCanvas, hline, vline } from "./canvas";
import { topBar, TOP_BAR_BOTTOM } from "./chrome";
import { drawCastawayScene } from "./castaway-draw";
import type { FaceKey } from "./fonts-data";
import { drawType, measureType } from "./type";

const SCENE_BOTTOM = 352;
const TEXT_X = 20;
const TEXT_W = 548;
const LOG_X = 592;

export const CASTAWAY_TITLE = "SETE'S ISLAND";

/** Greedy word wrap to a pixel width. */
export function wrap(face: FaceKey, text: string, width: number): string[] {
  const lines: string[] = [];
  let cur = "";
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const next = cur ? `${cur} ${word}` : word;
    if (measureType(face, next) <= width || !cur) cur = next;
    else { lines.push(cur); cur = word; }
  }
  if (cur) lines.push(cur);
  return lines;
}

export function drawCastawayPage(v: CastawayView): Canvas {
  const c = createCanvas();
  topBar(c, ["DESK", CASTAWAY_TITLE], [`${v.setting.name} ${v.setting.value}%`, `BATTERY ${v.battery}%`, v.clock], `DAY ${v.day}`);
  drawCastawayScene(c, { x: 0, y: TOP_BAR_BOTTOM + 1, w: c.w, h: SCENE_BOTTOM - TOP_BAR_BOTTOM - 1 }, v.scene);
  hline(c, 0, SCENE_BOTTOM, c.w, 2);

  drawType(c, "plex13s", TEXT_X, SCENE_BOTTOM + 24, v.chapter, { spacing: 1 });
  // The story: the larger face when it fits in three lines, else the smaller one.
  let face: FaceKey = "plex15m";
  let lines = wrap(face, v.text, TEXT_W);
  if (lines.length > 3) { face = "plex13r"; lines = wrap(face, v.text, TEXT_W); }
  const step = face === "plex15m" ? 22 : 20;
  lines.slice(0, 4).forEach((line, i) => drawType(c, face, TEXT_X, SCENE_BOTTOM + 50 + i * step, line));

  vline(c, LOG_X - 16, SCENE_BOTTOM + 12, 480 - SCENE_BOTTOM - 24);
  drawType(c, "plex12s", LOG_X, SCENE_BOTTOM + 24, "EARLIER", { spacing: 1 });
  v.log.forEach((entry, i) => {
    const y = SCENE_BOTTOM + 50 + i * 40;
    drawType(c, "plex12m", LOG_X, y, entry.time);
    wrap("plex12m", entry.text, 800 - LOG_X - 16).slice(0, 1).forEach((l) => drawType(c, "plex12m", LOG_X, y + 18, l));
  });
  return c;
}
