/**
 * Filled and outlined shapes for drawn pictures on the desk panel (polygon, ellipse, arc, curve).
 * Like canvas.ts every shape clips to the buffer, and loops are bounded so a wild coordinate
 * cannot spin the CPU. Values follow canvas.ts: 0 = black, 1 = white.
 */

import type { Canvas } from "./canvas";
import { line, setPixel } from "./canvas";

export type Pt = readonly [number, number];

/** Even-odd scanline fill of a polygon. */
export function fillPolygon(c: Canvas, pts: readonly Pt[], v: number = 0): void {
  if (pts.length < 3) return;
  const ys = pts.map((p) => p[1]);
  const y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const y1 = Math.min(c.h - 1, Math.ceil(Math.max(...ys)));
  for (let y = y0; y <= y1; y++) {
    const yc = y + 0.5;
    const xs: number[] = [];
    for (let i = 0; i < pts.length; i++) {
      const [ax, ay] = pts[i];
      const [bx, by] = pts[(i + 1) % pts.length];
      if ((ay <= yc && by > yc) || (by <= yc && ay > yc)) xs.push(ax + ((yc - ay) / (by - ay)) * (bx - ax));
    }
    xs.sort((a, b) => a - b);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const xa = Math.max(0, Math.round(xs[k]));
      const xb = Math.min(c.w - 1, Math.round(xs[k + 1]) - 1);
      for (let x = xa; x <= xb; x++) c.px[y * c.w + x] = v;
    }
  }
}

/** A polygon's outline. */
export function strokePolygon(c: Canvas, pts: readonly Pt[], t: number = 1): void {
  for (let i = 0; i < pts.length; i++) {
    const [ax, ay] = pts[i];
    const [bx, by] = pts[(i + 1) % pts.length];
    line(c, ax, ay, bx, by, t);
  }
}

/** Connected line segments. */
export function polyline(c: Canvas, pts: readonly Pt[], t: number = 1): void {
  for (let i = 0; i + 1 < pts.length; i++) line(c, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], t);
}

/** Points around an ellipse (for filling, outlining or clipping). */
export function ellipsePts(cx: number, cy: number, rx: number, ry: number, n: number = 48): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    pts.push([cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]);
  }
  return pts;
}

export function fillEllipse(c: Canvas, cx: number, cy: number, rx: number, ry: number, v: number = 0): void {
  fillPolygon(c, ellipsePts(cx, cy, rx, ry), v);
}

export function strokeEllipse(c: Canvas, cx: number, cy: number, rx: number, ry: number, t: number = 1): void {
  strokePolygon(c, ellipsePts(cx, cy, rx, ry, 64), t);
}

/** Points on an elliptical arc, angles in radians (0 = right, clockwise on screen). */
export function arcPts(cx: number, cy: number, rx: number, ry: number, a0: number, a1: number, n: number = 24): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const a = a0 + ((a1 - a0) * i) / n;
    pts.push([cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]);
  }
  return pts;
}

/** Quadratic Bezier from p0 to p2 bending towards p1. */
export function quadPts(p0: Pt, p1: Pt, p2: Pt, n: number = 16): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    const u = 1 - t;
    pts.push([u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]]);
  }
  return pts;
}

/** A rectangle of size w x h centred on (cx, cy), rotated by `angle` radians. */
export function rotRect(cx: number, cy: number, w: number, h: number, angle: number): Pt[] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return ([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]] as const)
    .map(([x, y]) => [cx + x * cos - y * sin, cy + x * sin + y * cos] as Pt);
}

/** Scatter `n` single-pixel dots inside a polygon, from a seeded random source (sand, stars). */
export function stipple(c: Canvas, pts: readonly Pt[], n: number, rand: () => number, v: number = 0): void {
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  for (let i = 0, placed = 0; i < n * 4 && placed < n; i++) {
    const x = x0 + rand() * (x1 - x0);
    const y = y0 + rand() * (y1 - y0);
    if (inside(pts, x, y)) { setPixel(c, x, y, v); placed++; }
  }
}

export function inside(pts: readonly Pt[], x: number, y: number): boolean {
  let hit = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i];
    const [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}
