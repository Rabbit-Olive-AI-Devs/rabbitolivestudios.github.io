/**
 * Lettering for the series (spec section 4, "checked, never trusted"). Words are
 * drawn on two layers laid over the converted art: `paper` (0 = white) and `ink`
 * (0 = black). Each balloon is sized from its text, placed by search clear of
 * every speaker and keep-clear box, and its letters are checked pixel by pixel.
 * Nothing that does not fit is shrunk or cut: it comes back as a problem string.
 */

import type { Canvas } from "../hud/canvas";
import { createCanvas, fillRect, setPixel, strokeRect } from "../hud/canvas";
import type { FaceKey } from "../hud/fonts-data";
import { drawType, measureType } from "../hud/type";
import type { Rect } from "./templates";
import { CALM_FRACTION } from "./templates";

export interface Layers { paper: Canvas; ink: Canvas }

const BALLOON_FACE: FaceKey = "plex13r";
const BALLOON_LH = 18;
const MAX_BALLOON_LINES = 4;
const MARGIN = 6;
/** Letters must sit this far inside their balloon. */
const INSET = 4;

export function newLayers(): Layers {
  return { paper: createCanvas(), ink: createCanvas() };
}

export function wrapText(face: FaceKey, s: string, maxW: number): string[] {
  const lines: string[] = [];
  let cur = "";
  for (const word of s.split(/\s+/).filter(Boolean)) {
    const t = cur ? `${cur} ${word}` : word;
    if (measureType(face, t) > maxW && cur) { lines.push(cur); cur = word; } else cur = t;
  }
  if (cur) lines.push(cur);
  return lines;
}

const overlaps = (a: Rect, b: Rect, pad: number) =>
  a.x < b.x + b.w + pad && a.x + a.w + pad > b.x && a.y < b.y + b.h + pad && a.y + a.h + pad > b.y;

/** A white box with a black rule and one line of text. */
export function captionBox(L: Layers, x: number, y: number, text: string, maxW: number, label: string): Rect | string {
  const w = measureType("plex12s", text, 1) + 16;
  if (w > maxW) return `${label}: too long for its box (${text.length} chars)`;
  const box = { x, y, w, h: 22 };
  fillRect(L.paper, x, y, w, 22, 0);
  strokeRect(L.ink, x, y, w, 22, 2);
  drawType(L.ink, "plex12s", x + 8, y + 16, text, { spacing: 1 });
  return box;
}

/** Draw wrapped text on a canvas, or explain why it does not fit. */
export function textBlock(
  c: Canvas, face: FaceKey, x: number, y: number, text: string, maxW: number, maxLines: number, lineH: number, label: string,
): string | null {
  const lines = wrapText(face, text, maxW);
  if (lines.length > maxLines) return `${label}: needs ${lines.length} lines, room for ${maxLines}`;
  if (lines.some((l) => measureType(face, l) > maxW)) return `${label}: a word is wider than the space`;
  lines.forEach((l, i) => drawType(c, face, x, y + i * lineH, l));
  return null;
}

/** Most balloon-shaped wrap of at most four lines, or null. */
function bestWrap(text: string): { lines: string[]; tw: number; th: number } | null {
  let best: { lines: string[]; tw: number; th: number; score: number } | null = null;
  for (let w = 110; w <= 280; w += 10) {
    const lines = wrapText(BALLOON_FACE, text, w);
    if (lines.length > MAX_BALLOON_LINES) continue;
    const tw = Math.max(...lines.map((l) => measureType(BALLOON_FACE, l)));
    if (tw > 280) continue;
    const th = lines.length * BALLOON_LH;
    const score = Math.abs(th / tw - 0.42);
    if (!best || score < best.score) best = { lines, tw, th, score };
  }
  return best;
}

export function placeBalloon(
  L: Layers,
  b: { speaker: string; text: string; anchor: [number, number] | null },
  panel: Rect,
  taken: Rect[],
  keepClear: [number, number][],
): Rect | string {
  const wrap = bestWrap(b.text);
  if (!wrap) return `${b.speaker}: line too long for any balloon (${b.text.length} chars)`;
  const radio = b.anchor === null;
  const rx = radio ? Math.ceil(wrap.tw / 2) + 12 : Math.ceil((wrap.tw / 2) * 1.42) + 10;
  const ry = radio ? Math.ceil(wrap.th / 2) + 18 : Math.ceil((wrap.th / 2) * 1.42) + 8;
  const calmBottom = panel.y + panel.h * CALM_FRACTION;

  let spot: { cx: number; cy: number; cost: number } | null = null;
  for (let cy = panel.y + ry + MARGIN; cy <= panel.y + panel.h - ry - MARGIN; cy += 4) {
    for (let cx = panel.x + rx + MARGIN; cx <= panel.x + panel.w - rx - MARGIN; cx += 4) {
      const box = { x: cx - rx, y: cy - ry, w: rx * 2, h: ry * 2 };
      if (taken.some((t) => overlaps(box, t, MARGIN))) continue;
      // never over a head: keep each anchor well outside the ellipse
      if (keepClear.some(([ax, ay]) => ((ax - cx) / rx) ** 2 + ((ay - cy) / ry) ** 2 < 2.2)) continue;
      const calm = cy - ry < calmBottom ? 0 : 80;
      const cost = radio
        ? (cy - panel.y) + calm
        : Math.hypot(b.anchor![0] - cx, b.anchor![1] - cy) + (cy > b.anchor![1] ? 60 : 0) + calm;
      if (!spot || cost < spot.cost) spot = { cx, cy, cost };
    }
  }
  if (!spot) return `${b.speaker}: no room for the balloon in this panel`;
  const { cx, cy } = spot;
  const box = { x: cx - rx, y: cy - ry, w: rx * 2, h: ry * 2 };

  // Letters go on a scratch canvas first so each pixel can be checked.
  const scratch = createCanvas();
  const top = cy - wrap.th / 2 + (radio ? 8 : 0);
  wrap.lines.forEach((l, i) => drawType(scratch, BALLOON_FACE, cx, top + i * BALLOON_LH + 13, l, { align: "center" }));
  const inside = radio
    ? (x: number, y: number) => x >= box.x + INSET && x < box.x + box.w - INSET && y >= box.y + 18 && y < box.y + box.h - INSET
    : (x: number, y: number) => ((x - cx) / (rx - INSET)) ** 2 + ((y - cy) / (ry - INSET)) ** 2 <= 1;
  let outside = 0;
  for (let y = Math.max(0, box.y - 20); y < Math.min(480, box.y + box.h + 20); y++) {
    for (let x = Math.max(0, box.x - 20); x < Math.min(800, box.x + box.w + 20); x++) {
      if (scratch.px[y * 800 + x] === 0 && !inside(x, y)) outside++;
    }
  }
  if (outside > 0) return `${b.speaker}: ${outside} text pixels outside the balloon`;

  if (radio) {
    fillRect(L.paper, box.x, box.y, box.w, box.h, 0);
    strokeRect(L.ink, box.x, box.y, box.w, box.h, 2);
    fillRect(L.ink, box.x, box.y, box.w, 16);
    drawType(L.ink, "plex11m", box.x + 8, box.y + 12, `${b.speaker}  /  COMLINK`, { spacing: 1, white: true });
  } else {
    drawEllipseBalloon(L, cx, cy, rx, ry, b.anchor!);
  }
  for (let i = 0; i < scratch.px.length; i++) if (scratch.px[i] === 0) L.ink.px[i] = 0;
  return box;
}

function drawEllipseBalloon(L: Layers, cx: number, cy: number, rx: number, ry: number, anchor: [number, number]): void {
  const ang = Math.atan2(anchor[1] - cy, anchor[0] - cx);
  const dist = Math.hypot(anchor[0] - cx, anchor[1] - cy);
  const edge = (t: number): [number, number] => [cx + Math.cos(t) * rx * 0.9, cy + Math.sin(t) * ry * 0.9];
  const tip: [number, number] = [cx + Math.cos(ang) * (dist - 14), cy + Math.sin(ang) * (dist - 14)];
  const tail: [number, number][] = [edge(ang - 0.16), edge(ang + 0.16), tip];
  const inTri = (px: number, py: number) => {
    const s = (p: [number, number], q: [number, number]) => (px - q[0]) * (p[1] - q[1]) - (p[0] - q[0]) * (py - q[1]);
    const d1 = s(tail[0], tail[1]), d2 = s(tail[1], tail[2]), d3 = s(tail[2], tail[0]);
    return !((d1 < 0 || d2 < 0 || d3 < 0) && (d1 > 0 || d2 > 0 || d3 > 0));
  };
  const x0 = Math.floor(Math.min(cx - rx, tip[0]) - 4), x1 = Math.ceil(Math.max(cx + rx, tip[0]) + 4);
  const y0 = Math.floor(Math.min(cy - ry, tip[1]) - 4), y1 = Math.ceil(Math.max(cy + ry, tip[1]) + 4);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const e = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2;
      const t = inTri(x, y);
      if (e <= 1 || t) setPixel(L.paper, x, y, 0);
      const eo = ((x - cx) / (rx + 2)) ** 2 + ((y - cy) / (ry + 2)) ** 2;
      if (eo <= 1 && e > 1 && !t) setPixel(L.ink, x, y, 0);
    }
  }
  for (const base of [tail[0], tail[1]]) {
    const n = Math.ceil(Math.hypot(tip[0] - base[0], tip[1] - base[1]));
    for (let i = 0; i <= n; i++) {
      const x = Math.round(base[0] + ((tip[0] - base[0]) * i) / n), y = Math.round(base[1] + ((tip[1] - base[1]) * i) / n);
      if (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 > 1) { setPixel(L.ink, x, y, 0); setPixel(L.ink, x + 1, y, 0); }
    }
  }
}
