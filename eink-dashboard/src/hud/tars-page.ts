/**
 * The TARS page: the owner's computers down the side, AI allowances as a table
 * of segmented bars, and three cards — what TARS says about it all, what needs
 * attention, and when the next allowance refills. Named for the robot in
 * Interstellar, whose honesty and humour settings head the side column.
 */

import type { TarsView } from "../tars";
import type { Canvas } from "./canvas";
import { createCanvas, fillRect, hline, strokeRect, vline } from "./canvas";
import { displayDate, topBar, TOP_BAR_BOTTOM } from "./chrome";
import { hatch, plus } from "./parts";
import { drawType, measureType } from "./type";

const SIDEBAR_W = 176;
const CARDS_TOP = 334;
const HONESTY = "90%";
const HUMOR = "75%";

const label = (c: Canvas, x: number, y: number, s: string) => drawType(c, "plex11m", x, y, s, { spacing: 1 });
const value = (c: Canvas, x: number, y: number, s: string) => drawType(c, "plex12s", x, y, s, { align: "right" });

/** Segments filled in proportion to `used` (0–100); a row of dashes when there is no reading. */
function bar(c: Canvas, x: number, y: number, used: number | null, segments: number, h: number): void {
  const w = 9;
  const gap = 2;
  const filled = used === null ? 0 : used > 0 ? Math.max(1, Math.round((used / 100) * segments)) : 0;
  for (let i = 0; i < segments; i++) {
    const sx = x + i * (w + gap);
    if (used === null) hline(c, sx, y + h - 1, w);
    else if (i < filled) fillRect(c, sx, y, w, h);
    else strokeRect(c, sx, y, w, h);
  }
}

/** White text on a black block; returns its width. */
function pill(c: Canvas, right: number, y: number, s: string): void {
  const w = measureType("plex12s", s) + 14;
  fillRect(c, right - w, y, w, 19);
  drawType(c, "plex12s", right - w + 7, y + 14, s, { white: true });
}

function tag(c: Canvas, right: number, y: number, s: string): void {
  const w = measureType("plex11m", s) + 12;
  strokeRect(c, right - w, y, w, 17);
  drawType(c, "plex11m", right - w + 6, y + 12, s);
}

function chicagoTime(now: Date): { date: string; time: string } {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(now);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  return { date: `${get("year")}-${get("month")}-${get("day")}`, time: `${get("hour")}:${get("minute")}` };
}

export function drawTarsPage(v: TarsView, now: Date): Canvas {
  const c = createCanvas();
  const clock = chicagoTime(now);
  topBar(
    c, ["Desk", "TARS", "Systems"], [displayDate(clock.date), clock.time],
    v.alerts.length === 0 ? "ALL NOMINAL" : `${v.alerts.length} ALERT${v.alerts.length > 1 ? "S" : ""}`,
  );

  // Side column: TARS's settings, then the machines
  vline(c, SIDEBAR_W, TOP_BAR_BOTTOM + 1, c.h - TOP_BAR_BOTTOM - 1);
  drawType(c, "plex15m", 20, 64, "[TARS]");
  fillRect(c, 150, 54, 12, 9);
  label(c, 20, 92, "HONESTY"); value(c, 160, 92, HONESTY);
  label(c, 20, 112, "HUMOR"); value(c, 160, 112, HUMOR);
  hatch(c, 12, 128, 152, 9);
  if (v.machines.length === 0) {
    label(c, 20, 170, "NO MACHINE HAS");
    label(c, 20, 188, "REPORTED YET");
  }
  v.machines.forEach((m, i) => {
    const y = 150 + i * 160;
    drawType(c, "plex13s", 20, y + 12, m.name.slice(0, 10));
    if (m.awake) pill(c, 160, y - 2, "AWAKE");
    else tag(c, 160, y - 1, "ASLEEP");
    label(c, 20, y + 38, "DISK"); value(c, 160, y + 38, `${m.diskFreeGb} GB FREE`);
    bar(c, 20, y + 46, m.diskUsedPct, 13, 9);
    label(c, 20, y + 78, "MEMORY"); value(c, 160, y + 78, m.memUsedPct === null ? "-" : `${m.memUsedPct}%`);
    bar(c, 20, y + 86, m.memUsedPct, 13, 9);
    label(c, 20, y + 118, m.awake ? "UPTIME" : "LAST SEEN"); value(c, 160, y + 118, m.since);
    hline(c, 20, y + 128, 140);
  });

  // The table of AI allowances
  const X = { name: 190, session: 352, week: 524, right: 786 };
  label(c, X.name, 58, "AI QUOTAS");
  drawType(c, "plex11m", X.right, 58, v.reported ? `VIA CODEXBAR  /  REPORTED ${v.reported}` : "VIA CODEXBAR", { align: "right", spacing: 1 });
  label(c, X.session, 84, "SESSION");
  label(c, X.week, 84, "WEEK");
  drawType(c, "plex11m", X.right, 84, "RESETS IN", { align: "right", spacing: 1 });
  hline(c, X.name, 92, X.right - X.name);
  if (v.providers.length === 0) label(c, X.name, 118, "NO READINGS YET");
  v.providers.forEach((p, i) => {
    const y = 118 + i * 44;
    drawType(c, "plex15s", X.name, y, p.name);
    label(c, X.name, y + 16, p.plan);
    for (const [x, used] of [[X.session, p.session], [X.week, p.week]] as const) {
      if (used === undefined) { label(c, x, y + 4, "-"); continue; }
      bar(c, x, y - 9, used, 12, 11);
      if (used === null) label(c, x, y + 16, "NO DATA");
      else drawType(c, "plex12s", x + 12 * 11 + 4, y + 2, `${used}%`);
    }
    if (p.low) pill(c, X.right, y - 12, p.reset);
    else value(c, X.right, y + 2, p.reset);
    if (p.note) drawType(c, "plex11m", X.right, y + 16, p.note, { align: "right", spacing: 1 });
    hline(c, X.name, y + 24, X.right - X.name);
  });

  // Cards
  hline(c, SIDEBAR_W + 1, CARDS_TOP, c.w - SIDEBAR_W - 1);
  hatch(c, 478, 342, 7, 130);
  hatch(c, 642, 342, 7, 130);
  for (const [x0, x1] of [[190, 470], [496, 634], [660, 788]]) {
    for (const y of [348, 468]) { plus(c, x0, y); plus(c, x1, y); }
  }

  label(c, 204, 362, "TARS SAYS");
  v.quip.forEach((line, i) => drawType(c, "plex13r", 204, 390 + i * 20, line));
  label(c, 204, 456, "CUE LIGHT");
  // the light is lit when TARS is joking, as in the film
  if (v.joke) fillRect(c, 292, 446, 12, 11);
  else strokeRect(c, 292, 446, 12, 11);
  label(c, 312, 456, v.joke ? "ON  /  JOKING" : "OFF  /  SERIOUS");

  label(c, 508, 362, "ATTENTION");
  if (v.alerts.length === 0) {
    drawType(c, "plex12s", 508, 386, "NOTHING");
    hline(c, 508, 392, 118);
  }
  v.alerts.forEach((a, i) => {
    const y = 386 + i * 21;
    fillRect(c, 508, y - 9, 5, 9);
    // a long alert loses its tail rather than running into the next card
    drawType(c, "plex11m", 518, y, a.slice(0, 15));
    hline(c, 508, y + 6, 118);
  });

  label(c, 672, 362, "NEXT RESET");
  if (v.next) {
    const w = drawType(c, "af36", 672, 410, v.next.numeral);
    drawType(c, "plex15s", 672 + w + 10, 394, v.next.text);
    label(c, 672 + w + 10, 410, v.next.what.slice(0, 8));
  } else {
    drawType(c, "plex15s", 672, 394, "--");
  }
  label(c, 672, 440, "SPEND"); value(c, 788, 440, v.spend);
  hline(c, 672, 446, 116);
  label(c, 672, 462, "30D, LIST PRICE");

  return c;
}
