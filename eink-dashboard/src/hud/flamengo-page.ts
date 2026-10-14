/**
 * The Flamengo board: the whole league table down the left, the next match with
 * both clubs' crests on the right, and three cards — the last result, the last
 * five, and the countdown to kickoff. Times are Chicago's, where the panel is.
 */

import type { FlaSide, FlaView } from "../flamengo";
import { FLAMENGO_ID } from "../flamengo";
import type { Canvas } from "./canvas";
import { createCanvas, fillRect, hline, strokeRect, vline } from "./canvas";
import { displayDate, topBar, TOP_BAR_BOTTOM } from "./chrome";
import { CREST_SIZE, drawCrest } from "./crest";
import { FACES } from "./fonts-data";
import type { FaceKey } from "./fonts-data";
import { brackets, dotGrid, hatch, ruler } from "./parts";
import { drawType, measureType } from "./type";

/** X of the rule between the table and the match. */
const SPLIT = 340;
const CARDS_TOP = 334;
const ROW_PITCH = 19;
/** Rows the table has room for: a whole 20-club league. */
const MAX_ROWS = 20;
const COL = { pos: 30, name: 42, p: 196, w: 222, d: 248, l: 274, gd: 302, pts: 330 };
const OUTCOME_WORD = { W: "WIN", D: "DRAW", L: "LOSS" } as const;

const label = (c: Canvas, x: number, y: number, s: string) => drawType(c, "plex11m", x, y, s, { spacing: 1 });

/**
 * Capitals the faces can print. A letter with no glyph falls back to its plain
 * form rather than vanishing from a club's name.
 */
export function printable(s: string): string {
  const glyphs = FACES.plex11m.glyphs;
  let out = "";
  for (const ch of s.toUpperCase()) {
    if (glyphs[ch]) out += ch;
    else for (const base of ch.normalize("NFD")) if (glyphs[base]) out += base;
  }
  return out.replace(/\s+/g, " ").trim();
}

/** `s` cut back to whole words until it fits `maxW`; a single long word is cut by letters. */
export function fitWords(face: FaceKey, s: string, maxW: number, spacing: number = 0): string {
  const words = printable(s).split(" ");
  const joiners = new Set(["DE", "DO", "DA", "DEL", "OF", "IN", "/", "-"]);
  while (words.length > 1 && measureType(face, words.join(" "), spacing) > maxW) {
    words.pop();
    while (words.length > 1 && joiners.has(words[words.length - 1])) words.pop();
  }
  let out = words.join(" ");
  while (out.length > 1 && measureType(face, out, spacing) > maxW) out = out.slice(0, -1);
  return out;
}

function chicago(at: number): { date: string; time: string; day: string; dayMonth: string } {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", weekday: "short", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hourCycle: "h23",
  }).formatToParts(new Date(at));
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  return {
    date: `${get("year")}-${get("month")}-${get("day")}`, time: `${get("hour")}:${get("minute")}`,
    day: get("weekday").toUpperCase().slice(0, 3), dayMonth: `${get("day")}/${get("month")}`,
  };
}

function drawTable(c: Canvas, v: FlaView): void {
  drawType(c, "plex15m", 16, 60, "[SERIE A]");
  if (v.played) {
    drawType(c, "plex11m", SPLIT - 12, 60, `PLAYED ${v.played.done} OF ${v.played.of}`, { align: "right", spacing: 1 });
  }
  label(c, COL.name, 80, "CLUB");
  for (const [head, x] of [["P", COL.p], ["W", COL.w], ["D", COL.d], ["L", COL.l], ["GD", COL.gd], ["PTS", COL.pts]] as const) {
    drawType(c, "plex11m", x, 80, head, { align: "right" });
  }
  hline(c, 12, 86, SPLIT - 22);

  const rows = v.table.slice(0, MAX_ROWS);
  if (rows.length === 0) {
    label(c, COL.name, 110, "NO TABLE YET");
    return;
  }
  rows.forEach((r, i) => {
    const y = 103 + i * ROW_PITCH;
    const us = r.id === FLAMENGO_ID;
    if (us) fillRect(c, 10, y - 14, SPLIT - 18, ROW_PITCH);
    const face: FaceKey = us ? "plex12s" : "plex12m";
    drawType(c, face, COL.pos, y, String(r.pos), { align: "right", white: us });
    drawType(c, face, COL.name, y, fitWords(face, r.name, COL.p - 22 - COL.name), { white: us });
    for (const [n, x] of [[r.p, COL.p], [r.w, COL.w], [r.d, COL.d], [r.l, COL.l], [r.gd, COL.gd]] as const) {
      drawType(c, "plex12m", x, y, String(n), { align: "right", white: us });
    }
    drawType(c, "plex12s", COL.pts, y, String(r.pts), { align: "right", white: us });
    // A dashed line where one zone ends: Libertadores, its qualifiers, Sudamericana, relegation.
    if (v.zones.includes(r.pos) && i < rows.length - 1) for (let x = 12; x < SPLIT - 10; x += 6) hline(c, x, y + 5, 3);
  });
}

/** Text centred on `cx` over a cleared strip, so the dot grid does not run through it. */
function plate(c: Canvas, face: FaceKey, cx: number, y: number, s: string, spacing: number = 0): void {
  const w = measureType(face, s, spacing);
  fillRect(c, Math.round(cx - w / 2) - 8, y - 13, w + 16, 18, 1);
  drawType(c, face, cx, y, s, { align: "center", spacing });
}

function drawSide(c: Canvas, side: FlaSide, cx: number): void {
  const w = 209;
  brackets(c, cx - 104, 54, w, 158, 18);
  if (!drawCrest(c, side.crest, cx - CREST_SIZE / 2, 62)) {
    // No crest to show: the club's short code, large, in its place.
    drawType(c, "plex36r", cx, 126, printable(side.abbr).slice(0, 4), { align: "center" });
  }
  drawType(c, "plex15s", cx, 180, fitWords("plex15s", side.name, w - 12), { align: "center" });
  if (side.rank) drawType(c, "plex11m", cx, 198, fitWords("plex11m", side.rank, w - 8), { align: "center" });
}

function drawMatch(c: Canvas, v: FlaView): void {
  const left = SPLIT + 1;
  const cx = Math.round((left + c.w) / 2);
  dotGrid(c, left + 12, 50, c.w - left - 20, 270, 12, { x: cx - 226, y: 46, w: 453, h: 176 });
  if (!v.next) {
    plate(c, "plex15s", cx, 138, "NO MATCH SCHEDULED");
    return;
  }
  drawSide(c, v.next.home, cx - 116);
  drawSide(c, v.next.away, cx + 116);
  drawType(c, "plex15s", cx, 138, "V", { align: "center" });
  ruler(c, cx - 130, 244, 260);

  const when = chicago(v.next.at);
  const room = c.w - left - 40;
  plate(c, "plex15s", cx, 270, `${when.day} ${when.dayMonth}  ${when.time}`);
  if (v.next.venue) plate(c, "plex11m", cx, 290, fitWords("plex11m", v.next.venue, room, 1), 1);
  const stage = v.next.stage ? `  /  ${v.next.stage}` : "";
  plate(c, "plex11m", cx, 308, fitWords("plex11m", `${v.next.league}${stage}  /  ${v.next.where}`, room, 1), 1);
}

function drawCards(c: Canvas, v: FlaView): void {
  const left = SPLIT + 1;
  const x = [left + 14, left + 168, left + 322];
  const w = 140;
  const right = c.w - 14;
  hline(c, left, CARDS_TOP, c.w - left);
  hatch(c, left + 158, CARDS_TOP + 8, 5, 130);
  hatch(c, left + 312, CARDS_TOP + 8, 5, 130);

  label(c, x[0], 362, "LAST RESULT");
  if (v.last) {
    drawType(c, "plex12s", x[0] + w - 8, 362, OUTCOME_WORD[v.last.outcome], { align: "right" });
    drawType(c, "plex50r", x[0], 420, v.last.score);
    const versus = v.last.where === "HOME" ? "V" : "AT";
    label(c, x[0], 444, fitWords("plex11m", `${versus} ${v.last.opponent}`, w, 1));
    label(c, x[0], 460, fitWords("plex11m", `${v.last.league}  ${chicago(v.last.at).dayMonth}`, w, 1));
  } else {
    label(c, x[0], 400, "NONE YET");
  }

  // Last five, oldest first: a black box is a win, a striped one a draw, an empty one a loss.
  label(c, x[1], 362, "LAST FIVE");
  v.form.forEach((f, i) => {
    const bx = x[1] + i * 28;
    const by = 376;
    if (f.outcome === "W") {
      fillRect(c, bx, by, 24, 24);
    } else {
      strokeRect(c, bx, by, 24, 24);
      if (f.outcome === "D") {
        hatch(c, bx + 1, by + 1, 22, 22, 4);
        fillRect(c, bx + 6, by + 4, 12, 16, 1);
      }
    }
    drawType(c, "plex13s", bx + 12, by + 17, f.outcome, { align: "center", white: f.outcome === "W" });
    drawType(c, "plex11m", bx + 12, by + 40, printable(f.opponent).slice(0, 3), { align: "center" });
  });
  label(c, x[1], 446, "UNBEATEN");
  drawType(c, "plex12s", x[1] + w - 4, 446, String(v.unbeaten), { align: "right" });
  hline(c, x[1], 452, w - 4);

  label(c, x[2], 362, v.next?.live ? "KICKED OFF" : "KICKOFF IN");
  drawType(c, "plex28r", x[2], 404, v.countdown || "-");
  if (v.then) {
    const when = chicago(v.then.at);
    label(c, x[2], 430, "THEN");
    const room = right - x[2] - measureType("plex11m", "THEN", 1) - 10;
    drawType(c, "plex12s", right, 430, fitWords("plex12s", `${v.then.opponent}  ${v.then.where}`, room), { align: "right" });
    hline(c, x[2], 436, right - x[2]);
    label(c, x[2], 454, `${when.day} ${when.dayMonth}  ${when.time}`);
  }
}

export function drawFlamengoPage(v: FlaView, now: Date): Canvas {
  const c = createCanvas();
  const clock = chicago(now.getTime());
  const status = v.staleDays > 0 ? [`DATA ${v.staleDays}D OLD`] : [displayDate(clock.date), clock.time];
  topBar(c, ["Desk", "Flamengo", v.next?.live ? "Live" : "Matchday"], status, v.pill);
  vline(c, SPLIT, TOP_BAR_BOTTOM + 1, c.h - TOP_BAR_BOTTOM - 1);
  drawTable(c, v);
  drawMatch(c, v);
  drawCards(c, v);
  return c;
}
