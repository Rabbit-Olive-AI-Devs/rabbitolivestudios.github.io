/**
 * Layout of the castaway page (DECISIONS #93): the island across the top, the story below it,
 * and a two-line log on the right. Desk style: real fonts, nothing under plex11m.
 */

import type { CastawayView } from "../castaway";
import type { Canvas } from "./canvas";
import { createCanvas, hline, setColor, vline } from "./canvas";
import type { RGBImage } from "../series/art";
import { topBar, TOP_BAR_BOTTOM } from "./chrome";
import { drawCastawayScene } from "./castaway-draw";
import type { FaceKey } from "./fonts-data";
import { drawType, measureType } from "./type";

const SCENE_BOTTOM = 352;
const TEXT_X = 20;
const TEXT_W = 548;
const LOG_X = 592;
const LOG_W = 800 - LOG_X - 16;

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

/** The story text in the face that fits: the larger one up to three lines, else the smaller one. */
function storyLines(text: string): { face: FaceKey; lines: string[] } {
  const big = wrap("plex15m", text, TEXT_W);
  return big.length <= 3 ? { face: "plex15m", lines: big } : { face: "plex13r", lines: wrap("plex13r", text, TEXT_W) };
}

/** Everything below the picture: chapter line, story text and the "EARLIER" log. */
function drawStoryText(c: Canvas, chapter: string, text: string, log: readonly { time: string; text: string }[]): void {
  hline(c, 0, SCENE_BOTTOM, c.w, 2);
  drawType(c, "plex13s", TEXT_X, SCENE_BOTTOM + 24, chapter, { spacing: 1 });
  const { face, lines } = storyLines(text);
  const step = face === "plex15m" ? 22 : 20;
  lines.slice(0, 4).forEach((line, i) => drawType(c, face, TEXT_X, SCENE_BOTTOM + 50 + i * step, line));

  vline(c, LOG_X - 16, SCENE_BOTTOM + 12, 480 - SCENE_BOTTOM - 24);
  drawType(c, "plex12s", LOG_X, SCENE_BOTTOM + 24, "EARLIER", { spacing: 1 });
  log.forEach((entry, i) => {
    const y = SCENE_BOTTOM + 50 + i * 40;
    drawType(c, "plex12m", LOG_X, y, entry.time);
    wrap("plex12m", entry.text, LOG_W).slice(0, 1).forEach((l) => drawType(c, "plex12m", LOG_X, y + 18, l));
  });
}

export function drawCastawayPage(v: CastawayView): Canvas {
  const c = createCanvas();
  topBar(c, ["DESK", CASTAWAY_TITLE], [`${v.setting.name} ${v.setting.value}%`, `BATTERY ${v.battery}%`, v.clock], `DAY ${v.day}`);
  drawCastawayScene(c, { x: 0, y: TOP_BAR_BOTTOM + 1, w: c.w, h: SCENE_BOTTOM - TOP_BAR_BOTTOM - 1 }, v.scene);
  drawStoryText(c, v.chapter, v.text, v.log);
  return c;
}

/** One beat of the desk serial, built offline (docs/superpowers/specs/2026-10-21-desk-serial-design.md). */
export interface SerialPageView {
  day: number;
  time: string;
  chapter: string;
  text: string;
  log: { time: string; text: string }[];
}

/** The painting's band: full width, from under the top bar to the rule. */
export const SERIAL_PICTURE = { w: 800, h: SCENE_BOTTOM - TOP_BAR_BOTTOM - 1 } as const;

/** Why a beat's text would not fit the page, or null when it does (nothing is ever clipped). */
export function storyTextFits(chapter: string, text: string, log: readonly { time: string; text: string }[]): string | null {
  if (measureType("plex13s", chapter, 1) > TEXT_W) return "chapter line too wide";
  if (storyLines(text).lines.length > 4) return "story text longer than four lines";
  if (log.length > 2) return "more than two log entries";
  for (const entry of log) if (wrap("plex12m", entry.text, LOG_W).length > 1) return "log line too wide";
  return null;
}

/** A serial beat: the painting in full colour (the panel converts it, #81), the text in black. */
export function drawSerialPage(v: SerialPageView, picture: RGBImage): Canvas {
  if (picture.w !== SERIAL_PICTURE.w || picture.h !== SERIAL_PICTURE.h) throw new Error("serial picture must be 800x315");
  const c = createCanvas();
  topBar(c, ["DESK", CASTAWAY_TITLE], [v.time], `DAY ${v.day}`);
  for (let y = 0; y < picture.h; y++) {
    for (let x = 0; x < picture.w; x++) {
      const i = (y * picture.w + x) * 3;
      setColor(c, x, TOP_BAR_BOTTOM + 1 + y, picture.rgb[i], picture.rgb[i + 1], picture.rgb[i + 2]);
    }
  }
  drawStoryText(c, v.chapter, v.text, v.log);
  return c;
}
