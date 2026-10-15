/**
 * Compose finished series pages. Each scene becomes one full-colour page, the writer's
 * picture untouched, plus its weekend replay; every panel gets the same page and converts it
 * itself (DECISIONS #73). The chrome (top bar, frames, strip) and the lettering layers are laid
 * over the picture as solid black and white. Teasers and season cards are text-only 1-bit pages.
 */

import type { Canvas } from "../hud/canvas";
import { createCanvas, fillRect, hline, strokeRect, vline } from "../hud/canvas";
import { displayDate, topBar } from "../hud/chrome";
import { brackets, dotGrid } from "../hud/parts";
import { drawType, measureType } from "../hud/type";
import { encodePNG1Bit, encodePNGRGB } from "../png";
import type { RGBImage } from "./art";
import { nextSceneLabel, sceneTimeLabel } from "./clock";
import type { Layers } from "./letter";
import { captionBox, newLayers, placeBalloon, textBlock, wrapText } from "./letter";
import type { Rect } from "./templates";
import { STRIP_TOP, TEMPLATES } from "./templates";
import type { Chapter, Scene, SeasonInfo } from "./types";
import { SCENES_PER_CHAPTER } from "./types";

const W = 800, H = 480;

export interface PanelArt {
  image: RGBImage;
  /** Balloon anchors in page coordinates (null for off-picture voices), in balloon order. */
  anchors: ([number, number] | null)[];
  /** Keep-clear boxes in page coordinates. */
  clear: Rect[];
}

const pad = (n: number) => String(n).padStart(2, "0");

export function pillFor(info: SeasonInfo, chapter: number): string {
  return `${info.number === 0 ? "P" : `S${info.number}`} / CH ${pad(chapter)}`;
}

function chrome(info: SeasonInfo, chapter: Chapter, scene: Scene, replay: boolean): { c: Canvas; problems: string[] } {
  const c = createCanvas();
  const problems: string[] = [];
  const t = TEMPLATES[scene.template];
  const status = replay ? ["REPLAY", `SCENE ${scene.number}/${SCENES_PER_CHAPTER}`] : [`SCENE ${scene.number}/${SCENES_PER_CHAPTER}`, sceneTimeLabel(scene.number)];
  topBar(c, [info.show, `Chapter ${chapter.number}`], status, pillFor(info, chapter.number));
  for (const p of t.panels) if (t.id !== "splash") strokeRect(c, p.x - 2, p.y - 2, p.w + 4, p.h + 4, 2);
  const meta = (x: number, y: number, w: number) => {
    drawType(c, "plex11m", x, y, "STORY & ART", { spacing: 1 });
    const who = info.writer.toUpperCase(), ww = measureType("plex12s", who) + 14;
    fillRect(c, x, y + 6, ww, 18);
    drawType(c, "plex12s", x + ww / 2, y + 19, who, { align: "center", white: true });
    const title = chapter.title.toUpperCase();
    if (measureType("plex11m", title, 1) <= w - ww - 10) drawType(c, "plex11m", x + w, y + 19, title, { align: "right", spacing: 1 });
    drawType(c, "plex11m", x, y + 42, replay ? "WEEK REPLAY" : "NEXT SCENE", { spacing: 1 });
    drawType(c, "plex12s", x + w, y + 42, replay ? `${scene.number}/${SCENES_PER_CHAPTER}` : nextSceneLabel(scene.number), { align: "right" });
    const step = w / SCENES_PER_CHAPTER;
    for (let i = 0; i < SCENES_PER_CHAPTER; i++) {
      const bx = Math.round(x + i * step);
      if (i < scene.number) fillRect(c, bx, y + 50, Math.max(2, Math.round(step) - 2), 6);
      else strokeRect(c, bx, y + 50, Math.max(2, Math.round(step) - 2), 6);
    }
  };
  if (t.strip) {
    const err = textBlock(c, "plex15m", 16, 420, scene.narration ?? "", 548, 2, 21, "narration");
    if (err) problems.push(err);
    vline(c, 584, 402, 72);
    meta(598, 416, 188);
  }
  if (t.column) {
    const col = t.column;
    const err = textBlock(c, "plex13r", col.x + 4, col.y + 20, scene.narration ?? "", col.w - 8, Math.floor((col.h - 90) / 19), 19, "narration");
    if (err) problems.push(err);
    hline(c, col.x, col.y + col.h - 70, col.w);
    meta(col.x + 4, col.y + col.h - 56, col.w - 8);
  }
  return { c, problems };
}

function letterScene(info: SeasonInfo, chapter: Chapter, scene: Scene, art: PanelArt[]): { L: Layers; problems: string[] } {
  const L = newLayers();
  const problems: string[] = [];
  const t = TEMPLATES[scene.template];
  const taken: Rect[] = [];
  if (scene.template === "splash") {
    const p = t.panels[0];
    if (scene.number === 1) {
      const title = chapter.title.toUpperCase();
      const tw = measureType("plex28r", title);
      if (tw > p.w - 64) problems.push(`chapter ${chapter.number} title: too wide for the title card`);
      else {
        const boxW = Math.max(tw, 220) + 32;
        fillRect(L.paper, p.x + 16, p.y + p.h - 112, boxW, 96, 0);
        strokeRect(L.ink, p.x + 16, p.y + p.h - 112, boxW, 96, 2);
        drawType(L.ink, "plex11m", p.x + 32, p.y + p.h - 88, `CHAPTER ${pad(chapter.number)}`, { spacing: 2 });
        drawType(L.ink, "plex28r", p.x + 32, p.y + p.h - 50, title);
        const credit = `STORY & ART  ${info.writer.toUpperCase()}`, cw = measureType("plex11m", credit, 1) + 12;
        fillRect(L.ink, p.x + 32, p.y + p.h - 40, cw, 17);
        drawType(L.ink, "plex11m", p.x + 38, p.y + p.h - 28, credit, { spacing: 1, white: true });
      }
    } else if (scene.caption) {
      const r = captionBox(L, p.x + 16, p.y + p.h - 38, scene.caption, p.w - 32, "caption");
      if (typeof r === "string") problems.push(r); else taken.push(r);
    }
    return { L, problems };
  }
  if (scene.caption) {
    const p = t.panels[0];
    const r = captionBox(L, p.x + 10, p.y + 10, scene.caption, p.w - 20, "caption");
    if (typeof r === "string") problems.push(r); else taken.push(r);
  }
  scene.panels.forEach((pn, pi) => {
    const panel = t.panels[pi];
    if (pn.caption) {
      const r = captionBox(L, panel.x + 8, panel.y + (pi === 0 && scene.caption ? 40 : 8), pn.caption, panel.w - 16, `panel ${pi + 1} caption`);
      if (typeof r === "string") problems.push(r); else taken.push(r);
    }
    const anchors = art[pi].anchors.filter((a): a is [number, number] => a !== null);
    (pn.balloons ?? []).forEach((b, bi) => {
      const r = placeBalloon(L, { speaker: b.speaker, text: b.text, anchor: art[pi].anchors[bi] }, panel, [...taken, ...art[pi].clear], anchors);
      if (typeof r === "string") problems.push(r); else taken.push(r);
    });
  });
  return { L, problems };
}

export async function renderScene(
  info: SeasonInfo, chapter: Chapter, scene: Scene, art: PanelArt[],
): Promise<{ pages: Record<string, Uint8Array>; problems: string[] }> {
  const t = TEMPLATES[scene.template];
  const live = chrome(info, chapter, scene, false);
  const replay = chrome(info, chapter, scene, true);
  const lettering = letterScene(info, chapter, scene, art);
  const problems = [...live.problems, ...lettering.problems].map((p) => `chapter ${chapter.number} scene ${scene.number}: ${p}`);
  if (problems.length) return { pages: {}, problems };

  const rgb = new Uint8Array(W * H * 3).fill(255);
  const inArt = new Uint8Array(W * H);
  t.panels.forEach((p, i) => {
    const img = art[i].image;
    for (let y = 0; y < p.h; y++) {
      for (let x = 0; x < p.w; x++) {
        const s = (y * img.w + x) * 3, d = ((p.y + y) * W + p.x + x) * 3;
        rgb[d] = img.rgb[s]; rgb[d + 1] = img.rgb[s + 1]; rgb[d + 2] = img.rgb[s + 2];
        inArt[(p.y + y) * W + p.x + x] = 1;
      }
    }
  });
  const { L } = lettering;
  /** 0 black, 1 white, -1 keep the writer's picture untouched. */
  const over = (c: Canvas, i: number) => (!inArt[i] ? c.px[i] : L.ink.px[i] === 0 ? 0 : L.paper.px[i] === 0 ? 1 : -1);

  // Full colour, unconverted: each panel turns the page into what it can show (DECISIONS #73).
  const pages: Record<string, Uint8Array> = {};
  for (const [tag, c] of [["s", live.c], ["r", replay.c]] as const) {
    const page = rgb.slice();
    for (let i = 0; i < W * H; i++) {
      const v = over(c, i);
      if (v !== -1) page.fill(v ? 255 : 0, i * 3, i * 3 + 3);
    }
    pages[tag] = await encodePNGRGB(page, W, H);
  }
  return { pages, problems: [] };
}

/** The big number (or TODAY) and its label, in the band above the card's text. */
const COUNTDOWN_BOX = { x: 220, y: 66, w: 360, h: 112 };

function countdown(c: Canvas, days: number): void {
  const [big, label] = days === 0 ? ["TODAY", "AT 08:00"] : [String(days), days === 1 ? "DAY TO GO" : "DAYS TO GO"];
  drawType(c, "plex50r", 400, 132, big, { align: "center" });
  drawType(c, "plex15s", 400, 164, label, { align: "center", spacing: 2 });
}

function card(info: SeasonInfo, trail: string, pill: string, status: string[], lines: { face: "plex11m" | "plex15s" | "plex28r" | "plex13r"; text: string }[], days: number | null = null): Promise<Uint8Array> {
  const c = createCanvas();
  topBar(c, [info.show, trail], status, pill);
  brackets(c, 30, 52, 740, 410, 26);
  dotGrid(c, 40, 60, 720, 120, 12, days === null ? undefined : COUNTDOWN_BOX);
  if (days !== null) countdown(c, days);
  let y = 230;
  for (const l of lines) {
    for (const row of wrapText(l.face, l.text, 680)) {
      drawType(c, l.face, 400, y, row, { align: "center", spacing: l.face === "plex11m" ? 2 : 0 });
      y += l.face === "plex28r" ? 44 : 26;
    }
    y += 8;
  }
  return encodePNG1Bit(c.px, W, H);
}

export function renderTeaser(info: SeasonInfo, chapter: Chapter, finale: boolean): Promise<Uint8Array> {
  return card(info, finale ? "Season finale" : "Next week", pillFor(info, chapter.number), [], [
    { face: "plex11m", text: finale ? "END OF THE SEASON" : `CHAPTER ${pad(chapter.number + 1)}  /  MONDAY 08:00` },
    { face: "plex15s", text: info.show },
    { face: "plex13r", text: chapter.teaser },
  ]);
}

/**
 * A season card. `daysLeft` (0 or more) adds the premiere countdown to the "before" card;
 * the stored card has none, and the Worker draws the live one each request (DECISIONS #80).
 */
export function renderCard(info: SeasonInfo, which: "before" | "after", daysLeft: number | null = null): Promise<Uint8Array> {
  const name = info.number === 0 ? "THE PROLOGUE" : `SEASON ${info.number}`;
  return which === "before"
    ? card(info, "Coming soon", name, [displayDate(info.premiere)], [
      { face: "plex11m", text: `${name}  /  STORY & ART ${info.writer.toUpperCase()}` },
      { face: "plex28r", text: info.show },
      { face: "plex15s", text: `PREMIERES MONDAY ${displayDate(info.premiere)} 08:00` },
    ], daysLeft !== null && daysLeft >= 0 ? daysLeft : null)
    : card(info, "Between seasons", name, [], [
      { face: "plex11m", text: `END OF ${name}` },
      { face: "plex28r", text: info.show },
      { face: "plex15s", text: "RETURNS SOON" },
    ]);
}
