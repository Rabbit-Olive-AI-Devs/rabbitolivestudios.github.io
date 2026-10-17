/**
 * The castaway page's picture (DECISIONS #93): a small island in a wide sea, drawn in ink, with the
 * day's light, the weather, the moon's real phase and whatever the castaway is up to. Pure black
 * and white so it stays crisp on the mono desk panel. Everything is placed from the scene
 * description and a seed, so the same moment always draws the same picture.
 */

import type { Canvas } from "./canvas";
import { createCanvas, fillRect, hline, line, setPixel, vline } from "./canvas";
import { disc } from "./parts";
import type { Pt } from "./shapes";
import { arcPts, fillEllipse, fillPolygon, polyline, quadPts, rotRect, stipple, strokeEllipse, strokePolygon } from "./shapes";
import { drawType } from "./type";
import { seededRandom } from "../art-styles";

export type Light = "dawn" | "day" | "dusk" | "night";
export type Weather = "clear" | "cloudy" | "rain" | "storm";
export type Pose = "stand" | "walk" | "wave" | "fish" | "sit" | "sleep" | "kick" | "dig" | "charge" | "think" | "dance";
export type Face = "happy" | "calm" | "sleep" | "surprised" | "focused" | "sad" | "wink" | "love";
export type Prop =
  | "fire" | "rod" | "bottle" | "crab" | "gull" | "ship" | "raft" | "castle" | "hammock" | "antenna"
  | "ball" | "pumpkin" | "lights" | "cake" | "radio" | "fin" | "turtle" | "flash" | "kite" | "book"
  | "flag" | "telescope" | "umbrella" | "coconuts" | "sign" | "whale" | "stars-shoot" | "plane" | "balloon"
  | "nest" | "lantern";

export interface Scene {
  light: Light;
  weather: Weather;
  moon: number;            // 0-7, the real phase tonight (0 new, 4 full)
  pose: Pose;
  face: Face;
  at: number;              // 0..1 across the island, where the castaway stands
  props: readonly Prop[];
  hut: number;             // 0-3, how far the shelter has got
  antenna: number;         // 0-3, how far the radio mast has got
  sand?: string;           // a word written in the sand (Aurebesh face)
  seed: number;
}

/** The picture's box on the page. */
export interface Frame { x: number; y: number; w: number; h: number }

const HORIZON = 0.56;      // of the frame's height
const ISLAND_Y = 0.80;     // the sand's waterline
const ISLAND_RX = 0.29;    // of the frame's width
const ISLAND_RY = 0.13;    // of the frame's height
const ISLAND_CX = 0.53;
const HUT_U = 0.33;        // the shelter, left of the palm
const MAST_U = 0.77;       // the radio mast, near the right shore
const ROBOT_U0 = 0.47;     // where the castaway can stand (s.at = 0 .. 1)
const ROBOT_U1 = 0.74;
const CLEAR = 3;           // a layer pixel nothing was drawn on

/** Copy a layer's drawn pixels onto the scene, with a one-pixel white halo round its black ones. */
function composite(c: Canvas, layer: Canvas): void {
  const { w, h } = c;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = layer.px[i];
      if (v !== CLEAR) { c.px[i] = v; continue; }
      const near = (dx: number, dy: number) => {
        const xx = x + dx, yy = y + dy;
        return xx >= 0 && yy >= 0 && xx < w && yy < h && layer.px[yy * w + xx] === 0;
      };
      if (near(1, 0) || near(-1, 0) || near(0, 1) || near(0, -1) || near(1, 1) || near(-1, -1) || near(1, -1) || near(-1, 1)) c.px[i] = 1;
    }
  }
}

export function drawCastawayScene(c: Canvas, f: Frame, s: Scene): void {
  const rand = seededRandom(s.seed);
  const horizon = f.y + f.h * HORIZON;
  const dark = s.light === "night" || s.weather === "storm";

  drawSky(c, f, horizon, s, rand);
  drawSea(c, f, horizon, s, rand);
  drawFarIsland(c, f, horizon, s);
  if (s.props.includes("ship")) drawShip(c, f.x + f.w * 0.86, horizon, dark);
  if (s.props.includes("whale")) drawWhale(c, f.x + f.w * 0.78, horizon + f.h * 0.12, dark);
  if (s.props.includes("fin")) drawFin(c, f.x + f.w * 0.9, horizon + f.h * 0.3, dark);
  if (s.props.includes("plane")) drawPlane(c, f.x + f.w * 0.2, f.y + f.h * 0.14, dark);
  if (s.props.includes("balloon")) drawBalloon(c, f.x + f.w * 0.16, f.y + f.h * 0.22);

  // The island and everything on it go on their own layer, then onto the scene with a white
  // halo round every black stroke, so the palm and the castaway still read on a night sky.
  const layer = createCanvas(c.w, c.h);
  layer.px.fill(CLEAR);
  drawIsland(layer, f, rand);
  const groundAt = (u: number) => islandTop(f, u);
  const ground = (u: number) => groundAt(u - ISLAND_CX + 0.5);
  drawPalm(layer, f, s, groundAt);
  if (s.hut > 0) drawHut(layer, f.x + f.w * HUT_U, ground(HUT_U), s.hut);
  if (s.antenna > 0) drawAntenna(layer, f.x + f.w * MAST_U, ground(MAST_U), s.antenna, s.light === "night" || s.props.includes("radio"));
  if (s.sand) drawType(layer, "af20", f.x + f.w * 0.6, f.y + f.h * ISLAND_Y - 3, s.sand.toUpperCase(), { align: "center", spacing: 3 });
  drawPropsOnIsland(layer, f, s, groundAt, rand);
  const u = ROBOT_U0 + s.at * (ROBOT_U1 - ROBOT_U0);
  drawCastaway(layer, f.x + f.w * u, ground(u), s.pose, s.face, s.props, s.light === "night");
  composite(c, layer);
  if (s.weather === "rain" || s.weather === "storm") drawRain(c, f, rand, s.weather === "storm");
}

/** Height of the sand at a position across the frame (u = 0.5 is the island's middle). */
function islandTop(f: Frame, u: number): number {
  const dx = (u - 0.5) / ISLAND_RX;
  const k = Math.max(0, 1 - dx * dx);
  return f.y + f.h * ISLAND_Y - Math.sqrt(k) * f.h * ISLAND_RY;
}

/** A one-pixel line in either ink (canvas.line only draws black): white lines on a night sky. */
function inkLine(c: Canvas, x0: number, y0: number, x1: number, y1: number, v: number): void {
  const n = Math.min(2000, Math.ceil(Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0))));
  for (let k = 0; k <= n; k++) setPixel(c, x0 + ((x1 - x0) * k) / (n || 1), y0 + ((y1 - y0) * k) / (n || 1), v);
}

function inkPolyline(c: Canvas, pts: readonly Pt[], v: number, t: number = 1): void {
  for (let i = 0; i + 1 < pts.length; i++) for (let o = 0; o < t; o++) inkLine(c, pts[i][0], pts[i][1] + o, pts[i + 1][0], pts[i + 1][1] + o, v);
}

// --- Sky ---

function drawSky(c: Canvas, f: Frame, horizon: number, s: Scene, rand: () => number): void {
  if (s.light === "night" || s.weather === "storm") {
    fillRect(c, f.x, f.y, f.w, horizon - f.y, 0);
    if (s.light === "night" && s.weather !== "storm") {
      for (let i = 0; i < 140; i++) {
        const x = f.x + rand() * f.w;
        const y = f.y + rand() * (horizon - f.y - 12);
        setPixel(c, x, y, 1);
        if (rand() < 0.12) { setPixel(c, x + 1, y, 1); setPixel(c, x, y + 1, 1); setPixel(c, x - 1, y, 1); setPixel(c, x, y - 1, 1); }
      }
      if (s.props.includes("stars-shoot")) { inkLine(c, f.x + f.w * 0.25, f.y + 30, f.x + f.w * 0.37, f.y + 58, 1); shootingStar(c, f.x + f.w * 0.25, f.y + 30); }
      if (s.weather === "clear") drawMoon(c, f.x + f.w * 0.84, f.y + f.h * 0.18, 20, s.moon);
    }
    if (s.weather === "storm") {
      for (let i = 0; i < 5; i++) cloud(c, f.x + rand() * f.w, f.y + 20 + rand() * 50, 34 + rand() * 30, true);
      const bx = f.x + f.w * (0.2 + rand() * 0.6);
      inkPolyline(c, [[bx, f.y + 40], [bx - 12, f.y + 85], [bx + 4, f.y + 88], [bx - 10, horizon - 6]], 1, 3);
    }
    return;
  }
  if (s.light === "dawn" || s.light === "dusk") {
    // Bands thickening towards the horizon, and the sun sitting on it.
    for (let y = horizon - 2, gap = 3; y > f.y + 30; y -= gap, gap += gap < 12 ? 1 : 0) hline(c, f.x, y, f.w);
    const sx = f.x + f.w * (s.light === "dawn" ? 0.18 : 0.82);
    fillEllipse(c, sx, horizon, 30, 30, 1);
    for (let a = 0; a < 9; a++) {
      const ang = Math.PI + (a / 8) * Math.PI;
      line(c, sx + Math.cos(ang) * 38, horizon + Math.sin(ang) * 38, sx + Math.cos(ang) * 50, horizon + Math.sin(ang) * 50);
    }
    strokePolygon(c, arcPts(sx, horizon, 30, 30, Math.PI, Math.PI * 2, 30), 2);
  } else {
    const sx = f.x + f.w * 0.82;
    const sy = f.y + f.h * 0.17;
    if (s.weather === "clear") {
      disc(c, sx, sy, 18, false); disc(c, sx, sy, 17, false);
      for (let a = 0; a < 12; a++) {
        const ang = (a / 12) * Math.PI * 2;
        line(c, sx + Math.cos(ang) * 24, sy + Math.sin(ang) * 24, sx + Math.cos(ang) * (a % 2 ? 30 : 34), sy + Math.sin(ang) * (a % 2 ? 30 : 34));
      }
    }
  }
  const clouds = s.weather === "clear" ? 2 : s.weather === "cloudy" ? 5 : 7;
  for (let i = 0; i < clouds; i++) cloud(c, f.x + 40 + rand() * (f.w - 80), f.y + 24 + rand() * (horizon - f.y - 70), 28 + rand() * 34, s.weather === "rain");
  if (s.props.includes("gull")) for (let i = 0; i < 3; i++) gull(c, f.x + f.w * (0.35 + i * 0.07 + rand() * 0.03), f.y + 40 + rand() * 40, false);
  if (s.props.includes("kite")) drawKite(c, f.x + f.w * 0.45, f.y + 40);
}

function cloud(c: Canvas, x: number, y: number, w: number, heavy: boolean): void {
  const puffs: Array<[number, number, number]> = [[-0.35, 0.1, 0.32], [0, -0.12, 0.42], [0.34, 0.06, 0.34], [0.05, 0.18, 0.3]];
  for (const [dx, dy, r] of puffs) fillEllipse(c, x + dx * w, y + dy * w, r * w + 2, r * w * 0.7 + 2, 0);
  for (const [dx, dy, r] of puffs) fillEllipse(c, x + dx * w, y + dy * w, r * w, r * w * 0.7, heavy ? 0 : 1);
  if (heavy) for (const [dx, dy, r] of puffs) fillEllipse(c, x + dx * w, y + dy * w - 2, r * w - 2, r * w * 0.7 - 3, 1);
}

function gull(c: Canvas, x: number, y: number, white: boolean): void {
  inkPolyline(c, [...quadPts([x - 9, y], [x - 4, y - 6], [x, y], 6), ...quadPts([x, y], [x + 4, y - 6], [x + 9, y], 6)], white ? 1 : 0, 2);
}

function drawMoon(c: Canvas, x: number, y: number, r: number, phase: number): void {
  fillEllipse(c, x, y, r, r, 1);
  // Shadow: a black ellipse slid across, narrowest at the quarters, gone at full.
  if (phase !== 4) {
    const waxing = phase < 4;
    const k = phase === 0 ? 0 : Math.cos((phase / 8) * Math.PI * 2);
    for (let yy = -r; yy <= r; yy++) {
      const half = Math.sqrt(r * r - yy * yy);
      const edge = half * k;
      for (let xx = -Math.ceil(half); xx <= Math.ceil(half); xx++) {
        const dark = waxing ? xx < edge : xx > -edge;
        if (dark && phase !== 0 ? true : phase === 0) setPixel(c, x + xx, y + yy, 0);
      }
    }
    if (phase === 0) strokeEllipse(c, x, y, r, r, 1);
    else for (let a = 0; a < 64; a++) setPixel(c, x + Math.cos(a / 10.2) * r, y + Math.sin(a / 10.2) * r, 1);
  }
}

function shootingStar(c: Canvas, x: number, y: number): void {
  for (const [dx, dy] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) setPixel(c, x + dx, y + dy, 1);
}

// --- Sea ---

function drawSea(c: Canvas, f: Frame, horizon: number, s: Scene, rand: () => number): void {
  const night = s.light === "night" || s.weather === "storm";
  const bottom = f.y + f.h;
  if (night) fillRect(c, f.x, horizon, f.w, bottom - horizon, 0);
  hline(c, f.x, horizon, f.w, night ? 1 : 2);
  if (night) hline(c, f.x, horizon, f.w, 1);
  const ink = night ? 1 : 0;   // wave ticks: white on a night sea
  // Wave ticks: sparse and small far away, larger and denser near the viewer.
  for (let y = horizon + 6, row = 0; y < bottom - 2; row++) {
    const t = (y - horizon) / (bottom - horizon);
    const len = 4 + t * 14;
    const gapX = 26 - t * 10;
    const off = rand() * gapX;
    for (let x = f.x + off; x < f.x + f.w; x += gapX + rand() * gapX) {
      inkPolyline(c, quadPts([x, y], [x + len / 2, y - 2 - t * 2], [x + len, y], 6), ink);
    }
    y += 5 + t * 9;
  }
  if (night && s.light === "night" && s.weather === "clear" && s.moon !== 0) {
    const mx = f.x + f.w * 0.84;
    for (let y = horizon + 4; y < bottom - 4; y += 4) {
      const half = 3 + ((y - horizon) / (bottom - horizon)) * 22 * (s.moon <= 4 ? s.moon / 4 : (8 - s.moon) / 4);
      hline(c, mx - half + rand() * 4, y, half * 2 * (0.4 + rand() * 0.5), 1);
      for (let x = mx - half; x < mx + half; x += 2) if (rand() < 0.5) setPixel(c, x, y, 1);
    }
  }
}

function drawFarIsland(c: Canvas, f: Frame, horizon: number, s: Scene): void {
  const x = f.x + f.w * 0.12;
  const night = s.light === "night" || s.weather === "storm";
  const v = night ? 1 : 0;
  fillPolygon(c, arcPts(x, horizon, 26, 7, Math.PI, Math.PI * 2, 16), v);
  inkPolyline(c, [[x + 4, horizon - 6], [x + 1, horizon - 20]], v, 2);
  for (const d of [-1, 1]) inkPolyline(c, quadPts([x + 1, horizon - 20], [x + 1 + d * 5, horizon - 24], [x + 1 + d * 9, horizon - 18], 5), v);
  if (s.props.includes("flash")) {
    const lx = x - 8, ly = horizon - 9;
    for (let a = 0; a < 8; a++) {
      const ang = (a / 8) * Math.PI * 2;
      for (let r = 4; r < 10; r++) setPixel(c, lx + Math.cos(ang) * r, ly + Math.sin(ang) * r, night ? 1 : 0);
    }
    disc(c, lx, ly, 2, true);
    if (night) for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) if (dx * dx + dy * dy <= 4) setPixel(c, lx + dx, ly + dy, 1);
  }
}

function drawShip(c: Canvas, x: number, horizon: number, dark: boolean): void {
  const v = dark ? 1 : 0;
  fillPolygon(c, [[x - 30, horizon - 8], [x + 30, horizon - 8], [x + 22, horizon], [x - 24, horizon]], v);
  fillRect(c, x - 10, horizon - 16, 16, 8, v);
  fillRect(c, x - 2, horizon - 24, 5, 8, v);
  for (let i = 0; i < 3; i++) disc(c, x + 8 + i * 6, horizon - 30 - i * 5, 2 + i, false);
}

function drawWhale(c: Canvas, x: number, y: number, dark: boolean): void {
  const v = dark ? 1 : 0;
  fillPolygon(c, arcPts(x, y, 34, 12, Math.PI, Math.PI * 2, 20), v);
  polyline(c, [[x - 2, y - 12], [x - 6, y - 26], [x - 2, y - 22], [x + 2, y - 28], [x + 2, y - 12]], 1);
}

function drawFin(c: Canvas, x: number, y: number, dark: boolean): void {
  fillPolygon(c, [[x - 10, y], [x + 2, y - 18], [x + 8, y]], dark ? 1 : 0);
}

function drawPlane(c: Canvas, x: number, y: number, dark: boolean): void {
  const v = dark ? 1 : 0;
  fillPolygon(c, [[x - 18, y], [x + 18, y - 2], [x + 22, y + 1], [x - 18, y + 4]], v);
  fillPolygon(c, [[x - 2, y], [x + 6, y - 10], [x + 9, y - 10], [x + 6, y + 1]], v);
  // A banner on a rope behind it, blank: the castaway reads into it what he wants.
  line(c, x - 18, y + 2, x - 36, y + 4, 1);
  for (let i = 0; i < 3; i++) hline(c, x - 96, y + 1 + i * 5, 60);
  vline(c, x - 96, y + 1, 11); vline(c, x - 36, y + 1, 11);
}

function drawBalloon(c: Canvas, x: number, y: number): void {
  fillEllipse(c, x, y, 18, 22, 1);
  strokeEllipse(c, x, y, 18, 22, 2);
  for (const d of [-9, 0, 9]) polyline(c, quadPts([x + d * 0.3, y - 21], [x + d * 1.4, y], [x + d * 0.5, y + 20], 8), 1);
  line(c, x - 7, y + 20, x - 5, y + 32); line(c, x + 7, y + 20, x + 5, y + 32);
  fillRect(c, x - 6, y + 32, 12, 8);
}

function drawKite(c: Canvas, x: number, y: number): void {
  const pts: Pt[] = [[x, y - 14], [x + 10, y], [x, y + 18], [x - 10, y]];
  fillPolygon(c, pts, 1); strokePolygon(c, pts, 2);
  line(c, x, y - 14, x, y + 18); line(c, x - 10, y, x + 10, y);
  polyline(c, quadPts([x, y + 18], [x - 30, y + 60], [x + 20, y + 140], 20), 1);
}

// --- Island and what stands on it ---

function drawIsland(c: Canvas, f: Frame, rand: () => number): Pt[] {
  const cx = f.x + f.w * ISLAND_CX;
  const cy = f.y + f.h * ISLAND_Y;
  const top = arcPts(cx, cy, f.w * ISLAND_RX, f.h * ISLAND_RY, Math.PI, Math.PI * 2, 48);
  const pts: Pt[] = [...top, [cx + f.w * ISLAND_RX, cy + 4], [cx - f.w * ISLAND_RX, cy + 4]];
  fillPolygon(c, pts, 1);
  polyline(c, top, 2);
  stipple(c, pts, 260, rand);
  // Foam: a broken line along the waterline, and a second one further out.
  for (let x = cx - f.w * ISLAND_RX - 14; x < cx + f.w * ISLAND_RX + 14; x += 9 + rand() * 6) hline(c, x, cy + 5 + rand() * 2, 4 + rand() * 4);
  for (let x = cx - f.w * ISLAND_RX - 30; x < cx + f.w * ISLAND_RX + 30; x += 14 + rand() * 10) hline(c, x, cy + 12 + rand() * 3, 3 + rand() * 5);
  return pts;
}

function drawPalm(c: Canvas, f: Frame, s: Scene, groundAt: (u: number) => number): void {
  const bx = f.x + f.w * (ISLAND_CX - 0.12);
  const by = groundAt(0.5 - 0.12) + 2;
  const tx = bx - 34;
  const ty = f.y + f.h * 0.2;
  const trunk = quadPts([bx, by], [bx + 26, (by + ty) / 2], [tx, ty], 18);
  for (let i = 0; i + 1 < trunk.length; i++) line(c, trunk[i][0], trunk[i][1], trunk[i + 1][0], trunk[i + 1][1], 9 - Math.round(i / 5));
  for (let i = 0; i + 1 < trunk.length; i++) {
    const w = 9 - Math.round(i / 5);
    line(c, trunk[i][0] - w / 2 + 2, trunk[i][1], trunk[i][0] + w / 2 - 1, trunk[i][1] - 1, 1);
  }
  // Re-ink the trunk edge after the white ring marks.
  const fronds: Array<[number, number]> = [[-150, 70], [-120, 64], [-60, 74], [-20, 78], [20, 66], [55, 60], [100, 50]];
  for (const [deg, len] of fronds) frond(c, tx, ty, (deg * Math.PI) / 180, len, s.props.includes("lights"));
  for (const [dx, dy] of [[-4, 6], [5, 7], [0, 12]]) { disc(c, tx + dx, ty + dy, 4, true); setPixel(c, tx + dx - 1, ty + dy - 1, 1); }
  if (s.props.includes("hammock")) {
    const hx = bx + 120;
    vline(c, hx, by - 44, 44, 3);
    const sag = quadPts([bx + 8, by - 38], [(bx + hx) / 2, by - 6], [hx, by - 40], 14);
    polyline(c, sag, 2);
    polyline(c, sag.map(([x, y]) => [x, y - 4] as Pt), 1);
  }
}

function frond(c: Canvas, x: number, y: number, angle: number, len: number, lights: boolean): void {
  const ex = x + Math.cos(angle) * len;
  const ey = y + Math.sin(angle) * len * 0.55 + len * 0.35;
  const mid: Pt = [x + Math.cos(angle) * len * 0.55, y + Math.sin(angle) * len * 0.5 - 12];
  const spine = quadPts([x, y], mid, [ex, ey], 14);
  polyline(c, spine, 2);
  for (let i = 2; i < spine.length - 1; i++) {
    const [px, py] = spine[i];
    const [qx, qy] = spine[i + 1];
    const nx = -(qy - py), ny = qx - px;
    const n = Math.hypot(nx, ny) || 1;
    const leaf = 12 - i * 0.5;
    line(c, px, py, px + (nx / n) * leaf + (qx - px) * 0.8, py + (ny / n) * leaf + (qy - py) * 0.8 + 3, 1);
    line(c, px, py, px - (nx / n) * leaf + (qx - px) * 0.8, py - (ny / n) * leaf + (qy - py) * 0.8 + 3, 1);
    if (lights && i % 3 === 0) { disc(c, px, py + 5, 2, true); disc(c, px, py + 5, 3, false); }
  }
}

function drawHut(c: Canvas, x: number, ground: number, stage: number): void {
  // 1: two poles and a ridge; 2: a thatched roof; 3: a door, a flag and a mailbox.
  line(c, x - 34, ground, x - 30, ground - 40, 3);
  line(c, x + 34, ground, x + 30, ground - 40, 3);
  line(c, x - 40, ground - 40, x + 40, ground - 40, 3);
  if (stage >= 2) {
    const roof: Pt[] = [[x - 48, ground - 36], [x, ground - 70], [x + 48, ground - 36]];
    fillPolygon(c, roof, 1); strokePolygon(c, roof, 2);
    for (let i = -40; i <= 40; i += 6) line(c, x + i, ground - 37, x + i * 0.4, ground - 64 + Math.abs(i) * 0.1, 1);
  }
  if (stage >= 3) {
    fillRect(c, x - 9, ground - 26, 18, 26, 0);
    fillRect(c, x - 7, ground - 24, 14, 22, 1);
    setPixel(c, x + 4, ground - 13, 0);
    vline(c, x + 50, ground - 20, 20, 2);
    fillRect(c, x + 44, ground - 30, 14, 10, 1); fillRect(c, x + 44, ground - 30, 14, 2, 0);
    line(c, x + 44, ground - 30, x + 44, ground - 20); line(c, x + 58, ground - 30, x + 58, ground - 20); hline(c, x + 44, ground - 20, 15);
  }
}

function drawAntenna(c: Canvas, x: number, ground: number, stage: number, lit: boolean): void {
  const top = ground - 40 - stage * 22;
  line(c, x - 8, ground, x, top, 2);
  line(c, x + 8, ground, x, top, 2);
  for (let y = ground - 10; y > top + 6; y -= 12) hline(c, x - 8 + (ground - y) * 8 / (ground - top), y, 16 - (ground - y) * 16 / (ground - top) + 1);
  if (stage >= 2) { hline(c, x - 12, top + 8, 24, 2); vline(c, x - 12, top + 4, 8); vline(c, x + 12, top + 4, 8); }
  if (stage >= 3) {
    disc(c, x, top - 4, 4, true);
    if (lit) for (const r of [8, 12, 16]) for (let a = -0.9; a <= 0.9; a += 0.08) {
      setPixel(c, x + Math.cos(a - Math.PI / 2) * r * 1.3, top - 4 + Math.sin(a - Math.PI / 2) * r, 0);
    }
  }
}

function drawPropsOnIsland(c: Canvas, f: Frame, s: Scene, groundAt: (u: number) => number, rand: () => number): void {
  const at = (u: number) => [f.x + f.w * u, groundAt(u - ISLAND_CX + 0.5)] as const;
  if (s.props.includes("fire")) {
    const [x, y] = at(0.55);
    for (const d of [-8, -3, 3, 8]) line(c, x + d, y, x - d * 0.6, y - 6, 3);
    const flame: Pt[] = [[x - 9, y - 4], [x - 6, y - 18], [x - 2, y - 12], [x, y - 28], [x + 4, y - 14], [x + 8, y - 20], [x + 9, y - 4]];
    fillPolygon(c, flame, s.light === "night" ? 1 : 0);
    if (s.light === "night") fillPolygon(c, [[x - 4, y - 5], [x, y - 16], [x + 4, y - 5]], 0);
    for (let i = 0; i < 6; i++) setPixel(c, x - 6 + rand() * 12, y - 34 - rand() * 14, s.light === "night" ? 1 : 0);
  }
  if (s.props.includes("castle")) {
    const [x, y] = at(0.62);
    fillPolygon(c, [[x - 18, y], [x - 16, y - 14], [x + 16, y - 14], [x + 18, y]], 1); strokePolygon(c, [[x - 18, y], [x - 16, y - 14], [x + 16, y - 14], [x + 18, y]], 1);
    for (const d of [-12, 0, 12]) { fillRect(c, x + d - 4, y - 24, 8, 10, 1); strokePolygon(c, [[x + d - 4, y - 14], [x + d - 4, y - 24], [x + d + 4, y - 24], [x + d + 4, y - 14]], 1); }
    line(c, x, y - 24, x, y - 34); fillPolygon(c, [[x, y - 34], [x + 7, y - 31], [x, y - 28]], 0);
  }
  if (s.props.includes("crab")) {
    const [x, y] = at(0.44 + rand() * 0.04);
    fillEllipse(c, x, y - 4, 7, 4, 0);
    for (const d of [-1, 1]) { line(c, x + d * 6, y - 5, x + d * 11, y - 10, 1); disc(c, x + d * 12, y - 11, 2, true); line(c, x + d * 4, y - 2, x + d * 9, y + 1); }
    setPixel(c, x - 2, y - 9, 0); setPixel(c, x + 2, y - 9, 0); vline(c, x - 2, y - 8, 2); vline(c, x + 2, y - 8, 2);
  }
  if (s.props.includes("bottle")) {
    const [x, y] = at(0.78);
    const b = rotRect(x, y - 3, 18, 7, -0.3);
    fillPolygon(c, b, 1); strokePolygon(c, b, 1);
    line(c, x + 8, y - 6, x + 12, y - 8, 2);
    hline(c, x - 4, y - 4, 6);
  }
  if (s.props.includes("raft")) {
    const [x] = at(0.84);
    const y = f.y + f.h * ISLAND_Y + 8;
    for (let i = 0; i < 6; i++) { fillRect(c, x - 24 + i * 8, y - 4, 7, 8, 1); strokePolygon(c, [[x - 24 + i * 8, y - 4], [x - 17 + i * 8, y - 4], [x - 17 + i * 8, y + 4], [x - 24 + i * 8, y + 4]], 1); }
    vline(c, x, y - 40, 36, 2);
    const sail: Pt[] = [[x + 2, y - 38], [x + 24, y - 12], [x + 2, y - 10]];
    fillPolygon(c, sail, 1); strokePolygon(c, sail, 1);
  }
  if (s.props.includes("ball")) { const [x, y] = at(0.5 + s.at * 0.1); disc(c, x + 26, y - 6, 6, true); disc(c, x + 26, y - 6, 3, false); }
  if (s.props.includes("pumpkin")) {
    const [x, y] = at(0.6);
    fillEllipse(c, x, y - 9, 13, 10, 0);
    line(c, x, y - 19, x + 2, y - 24, 2);
    { fillPolygon(c, [[x - 7, y - 12], [x - 3, y - 12], [x - 5, y - 15]], 1); fillPolygon(c, [[x + 3, y - 12], [x + 7, y - 12], [x + 5, y - 15]], 1); for (let i = -6; i <= 6; i += 3) fillRect(c, x + i, y - 7 - (i % 2 ? 1 : 0), 2, 2, 1); }
  }
  if (s.props.includes("cake")) {
    const [x, y] = at(0.6);
    fillRect(c, x - 14, y - 12, 28, 12, 1); strokePolygon(c, [[x - 14, y], [x - 14, y - 12], [x + 14, y - 12], [x + 14, y]], 2);
    hline(c, x - 14, y - 6, 28);
    for (const d of [-7, 0, 7]) { vline(c, x + d, y - 20, 8); fillPolygon(c, [[x + d - 2, y - 21], [x + d, y - 26], [x + d + 2, y - 21]], 0); }
  }
  if (s.props.includes("radio")) {
    const [x, y] = at(0.46);
    fillRect(c, x - 13, y - 14, 26, 14, 1); strokePolygon(c, [[x - 13, y], [x - 13, y - 14], [x + 13, y - 14], [x + 13, y]], 2);
    disc(c, x - 5, y - 7, 4, false); for (let i = 0; i < 3; i++) hline(c, x + 3, y - 10 + i * 3, 7);
    line(c, x + 8, y - 14, x + 16, y - 28);
    for (const r of [5, 9]) for (let a = -2.4; a <= -0.6; a += 0.12) setPixel(c, x + 16 + Math.cos(a) * r, y - 28 + Math.sin(a) * r, 0);
  }
  if (s.props.includes("umbrella")) {
    const [x, y] = at(0.66);
    vline(c, x, y - 40, 40, 2);
    const top = arcPts(x, y - 40, 30, 14, Math.PI, Math.PI * 2, 18);
    fillPolygon(c, [...top], 1); polyline(c, top, 2); hline(c, x - 30, y - 40, 60, 2);
    for (const d of [-15, 0, 15]) line(c, x, y - 54, x + d * 2, y - 40);
  }
  if (s.props.includes("telescope")) {
    const [x, y] = at(0.4);
    line(c, x - 8, y, x, y - 18, 2); line(c, x + 8, y, x, y - 18, 2); line(c, x, y, x, y - 18, 2);
    const t = rotRect(x + 8, y - 24, 30, 7, -0.55);
    fillPolygon(c, t, 1); strokePolygon(c, t, 2);
  }
  if (s.props.includes("book")) { const [x, y] = at(0.52); fillPolygon(c, [[x - 10, y - 2], [x, y - 6], [x + 10, y - 2], [x, y + 1]], 1); strokePolygon(c, [[x - 10, y - 2], [x, y - 6], [x + 10, y - 2], [x, y + 1]], 1); line(c, x, y - 6, x, y + 1); }
  if (s.props.includes("flag")) {
    const [x, y] = at(0.3);
    vline(c, x, y - 52, 52, 2);
    const fl: Pt[] = [[x + 1, y - 52], [x + 26, y - 46], [x + 1, y - 38]];
    fillPolygon(c, fl, 0);
  }
  if (s.props.includes("turtle")) {
    const [x, y] = at(0.8);
    fillPolygon(c, arcPts(x, y - 1, 13, 9, Math.PI, Math.PI * 2, 14), 0);
    disc(c, x + 15, y - 4, 3, true); line(c, x - 10, y, x - 15, y + 2, 2); line(c, x + 9, y, x + 13, y + 2, 2);
    for (const d of [-5, 3]) disc(c, x + d, y - 5, 2, false);
  }
  if (s.props.includes("lantern")) {
    const [x, y] = at(0.58);
    fillRect(c, x - 5, y - 14, 10, 13, s.light === "night" ? 1 : 0); strokePolygon(c, [[x - 5, y - 1], [x - 5, y - 14], [x + 5, y - 14], [x + 5, y - 1]], 1);
    hline(c, x - 6, y - 15, 12); polyline(c, arcPts(x, y - 15, 4, 4, Math.PI, Math.PI * 2, 6), 1);
  }
  if (s.props.includes("nest")) {
    const [x, y] = at(0.74);
    polyline(c, arcPts(x, y - 4, 12, 6, 0, Math.PI, 10), 2);
    for (const d of [-4, 0, 4]) { fillEllipse(c, x + d, y - 6, 3, 4, 1); strokeEllipse(c, x + d, y - 6, 3, 4, 1); }
  }
  if (s.props.includes("sign")) {
    const [x, y] = at(0.88);
    vline(c, x, y - 34, 34, 2);
    fillRect(c, x - 18, y - 40, 36, 14, 1); strokePolygon(c, [[x - 18, y - 26], [x - 18, y - 40], [x + 18, y - 40], [x + 18, y - 26]], 2);
    drawType(c, "plex11m", x, y - 29, "SOS", { align: "center" });
  }
}

// --- The castaway ---

/**
 * A small robot made of four slabs hinged at the top, with a face screen on the middle two.
 * Poses swing the outer slabs (its arms and legs at once) around their hinges.
 */
export function drawCastaway(c: Canvas, x: number, ground: number, pose: Pose, face: Face, props: readonly Prop[], night: boolean): void {
  const SW = 13, SH = 64, GAP = 2;
  if (pose === "sleep") {
    // Lying on the sand, slabs stacked sideways, a Z or two.
    const y = ground - SW * 2 - 4;
    for (let i = 0; i < 4; i++) slab(c, [[x - SH / 2, y + i * (SW + GAP)], [x + SH / 2, y + i * (SW + GAP)], [x + SH / 2, y + i * (SW + GAP) + SW], [x - SH / 2, y + i * (SW + GAP) + SW]]);
    screen(c, x + SH / 2 - 22, y + SW + GAP + 1, 18, SW * 2, face, true);
    const zx = x + SH / 2 + 6;
    for (const [dx, dy, sz] of [[0, -6, 5], [9, -16, 7], [20, -30, 9]]) zed(c, zx + dx, y + dy, sz, night);
    return;
  }
  const sit = pose === "sit" || pose === "fish" || pose === "think";
  const h = sit ? SH * 0.7 : SH;
  const top = ground - h - (pose === "dance" ? 6 : 0);
  const xs = [x - 1.5 * (SW + GAP), x - 0.5 * (SW + GAP), x + 0.5 * (SW + GAP), x + 1.5 * (SW + GAP)];
  const swing: [number, number] = (() => {
    switch (pose) {
      case "walk": return [0.32, -0.32];
      case "wave": return [0.1, -2.5];
      case "fish": return [0.25, -1.1];
      case "kick": return [-0.9, 0.15];
      case "dig": return [0.2, 0.75];
      case "charge": return [1.35, -1.35];
      case "dance": return [0.9, 0.5];
      case "think": return [0.2, -1.9];
      default: return [0.06, -0.06];
    }
  })();
  // Outer slabs (left then right), swung around their top hinge.
  for (const [i, a] of [[0, swing[0]], [3, swing[1]]] as const) {
    const hx = xs[i];
    const cx = hx - Math.sin(a) * (h / 2);
    const cy = top + Math.cos(a) * (h / 2);
    slab(c, rotRect(cx, cy, SW, h, a));
    if (pose === "charge") for (let k = 1; k < 4; k++) {
      const px = hx - Math.sin(a) * (h * k / 4), py = top + Math.cos(a) * (h * k / 4);
      line(c, px - Math.cos(a) * 4, py - Math.sin(a) * 4, px + Math.cos(a) * 4, py + Math.sin(a) * 4, 1);
    }
  }
  for (const i of [1, 2]) slab(c, [[xs[i] - SW / 2, top], [xs[i] + SW / 2, top], [xs[i] + SW / 2, top + h], [xs[i] - SW / 2, top + h]]);
  screen(c, xs[1] - SW / 2 + 2, top + 6, SW * 2 + GAP - 4, 14, face, false);
  // Little feet so it stands on the sand rather than floating.
  hline(c, x - 2 * (SW + GAP), ground, 4 * (SW + GAP), 2);

  const handR: Pt = [xs[3] - Math.sin(swing[1]) * h, top + Math.cos(swing[1]) * h];
  if (pose === "fish" || props.includes("rod")) {
    const tip: Pt = [handR[0] + 70, handR[1] - 46];
    line(c, handR[0], handR[1], tip[0], tip[1], 2);
    polyline(c, quadPts(tip, [tip[0] + 40, tip[1] + 10], [tip[0] + 52, ground + 36], 12), 1);
    disc(c, tip[0] + 52, ground + 36, 2, false);
  }
  if (pose === "wave") for (const r of [6, 10]) for (let a = -2.6; a <= -1.6; a += 0.15) setPixel(c, handR[0] + Math.cos(a) * r, handR[1] + Math.sin(a) * r, night ? 1 : 0);
  if (pose === "think") { disc(c, x + 26, top - 14, 3, false); disc(c, x + 34, top - 26, 5, false); fillEllipse(c, x + 52, top - 46, 18, 11, 1); strokeEllipse(c, x + 52, top - 46, 18, 11, 1); drawType(c, "plex12s", x + 52, top - 42, "?", { align: "center" }); }
  if (pose === "dig") { const hand: Pt = handR; line(c, hand[0], hand[1], hand[0] + 8, hand[1] + 10, 2); }
}

function slab(c: Canvas, pts: readonly Pt[]): void {
  fillPolygon(c, pts, 1);
  strokePolygon(c, pts, 2);
}

/** The face screen: two eyes and a mouth in a few pixels, enough for a mood. */
function screen(c: Canvas, x: number, y: number, w: number, h: number, face: Face, sideways: boolean): void {
  fillRect(c, x, y, w, h, 0);
  const ex1 = x + w * 0.3, ex2 = x + w * 0.7, ey = y + h * (sideways ? 0.5 : 0.45);
  const eye = (ex: number) => {
    switch (face) {
      case "sleep": fillRect(c, ex - 2, ey, 5, 1, 1); break;
      case "happy": fillRect(c, ex - 2, ey, 1, 1, 1); fillRect(c, ex - 1, ey - 1, 3, 1, 1); fillRect(c, ex + 2, ey, 1, 1, 1); break;
      case "surprised": fillRect(c, ex - 2, ey - 2, 4, 4, 1); fillRect(c, ex - 1, ey - 1, 2, 2, 0); break;
      case "focused": fillRect(c, ex - 2, ey, 5, 2, 1); break;
      case "sad": fillRect(c, ex - 2, ey - 1, 1, 1, 1); fillRect(c, ex - 1, ey, 3, 1, 1); fillRect(c, ex + 2, ey - 1, 1, 1, 1); break;
      case "love": fillRect(c, ex - 2, ey - 2, 2, 2, 1); fillRect(c, ex + 1, ey - 2, 2, 2, 1); fillRect(c, ex - 2, ey, 5, 1, 1); fillRect(c, ex - 1, ey + 1, 3, 1, 1); fillRect(c, ex, ey + 2, 1, 1, 1); break;
      default: fillRect(c, ex - 1, ey - 1, 3, 3, 1);
    }
  };
  if (face === "wink") { eye(ex1); fillRect(c, ex2 - 2, ey, 5, 1, 1); } else { eye(ex1); eye(ex2); }
}

function zed(c: Canvas, x: number, y: number, s: number, night: boolean): void {
  const v = night ? 1 : 0;
  const pts: Pt[] = [[x, y], [x + s, y], [x, y + s], [x + s, y + s]];
  for (let i = 0; i + 1 < pts.length; i++) {
    const [ax, ay] = pts[i], [bx, by] = pts[i + 1];
    const n = Math.max(Math.abs(bx - ax), Math.abs(by - ay));
    for (let k = 0; k <= n; k++) { setPixel(c, ax + ((bx - ax) * k) / n, ay + ((by - ay) * k) / n, v); setPixel(c, ax + ((bx - ax) * k) / n, ay + 1 + ((by - ay) * k) / n, v); }
  }
}

function drawRain(c: Canvas, f: Frame, rand: () => number, storm: boolean): void {
  const n = storm ? 260 : 170;
  for (let i = 0; i < n; i++) {
    const x = f.x + rand() * f.w;
    const y = f.y + rand() * f.h;
    const len = 6 + rand() * 6;
    const k = storm ? 0.5 : 0.25;
    const v = c.px[Math.round(y) * c.w + Math.round(x)] === 0 ? 1 : 0;
    for (let t = 0; t < len; t++) setPixel(c, x - t * k, y + t, v);
  }
}
