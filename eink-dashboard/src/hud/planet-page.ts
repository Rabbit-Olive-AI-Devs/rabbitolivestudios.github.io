/**
 * The planet-of-the-day page: where the planet is and how it was found down the
 * side, the planet drawn in dots on the crosshair, and three cards with what is
 * known about it. Everything shown is from NASA's Exoplanet Archive; a value
 * the archive does not have reads UNKNOWN rather than being made up.
 */

import type { PlanetOfDay } from "../planet";
import type { Canvas } from "./canvas";
import { createCanvas, fillRect, hline, vline } from "./canvas";
import { dayOfYear } from "./aurebesh-words";
import { displayDate, topBar, TOP_BAR_BOTTOM } from "./chrome";
import type { FaceKey } from "./fonts-data";
import { brackets, disc, dotGrid, hatch, plus } from "./parts";
import { drawPlanet, planetType } from "./planet-draw";
import { drawType, measureType } from "./type";

const UNKNOWN = "UNKNOWN";
const SIDEBAR_W = 176;
const CX = 488;
const CY = 180;
const PLANET_R = 96;
const BOX = 112;
const CARDS_TOP = 334;
const FIRST_YEAR = 1995;

export function formatDistance(ly: number | null): string {
  if (ly === null) return UNKNOWN;
  return `${ly < 100 ? ly.toFixed(1) : Math.round(ly).toLocaleString("en-US")} LY`;
}

export function formatPeriod(days: number | null): string {
  if (days === null) return UNKNOWN;
  if (days < 10) return `${days.toFixed(1)} DAYS`;
  if (days < 1000) return `${Math.round(days)} DAYS`;
  return `${(days / 365.25).toFixed(1)} YEARS`;
}

export function formatTemp(kelvin: number | null): string {
  return kelvin === null ? UNKNOWN : `${Math.round(kelvin - 273.15)} C`;
}

export function formatMass(earths: number | null): string {
  if (earths === null) return UNKNOWN;
  return `${earths < 10 ? earths.toFixed(2) : Math.round(earths)} EARTHS`;
}

/** Right ascension: degrees to hours and minutes, the way star charts write it. */
export function formatRa(deg: number): string {
  let h = Math.floor(deg / 15);
  let m = Math.round((deg / 15 - h) * 60);
  if (m === 60) { m = 0; h = (h + 1) % 24; }
  return `RA ${String(h).padStart(2, "0")}H ${String(m).padStart(2, "0")}M`;
}

export function formatDec(deg: number): string {
  return `DEC ${deg < 0 ? "-" : "+"}${Math.abs(deg).toFixed(1).padStart(4, "0")}`;
}

/** How each discovery method is named on the page; anything else is shown shortened. */
const METHODS: Readonly<Record<string, string>> = {
  "Transit": "TRANSIT",
  "Radial Velocity": "WOBBLE",
  "Imaging": "IMAGING",
  "Microlensing": "LENSING",
  "Astrometry": "ASTROMETRY",
};

const label = (c: Canvas, x: number, y: number, s: string) => drawType(c, "plex11m", x, y, s, { spacing: 1 });

function dataRow(c: Canvas, x: number, y: number, w: number, key: string, value: string, rule: boolean): void {
  label(c, x, y, key);
  drawType(c, "plex12s", x + w, y, value, { align: "right" });
  if (rule) hline(c, x, y + 6, w);
}

/**
 * The largest face that has a glyph for every character of `s` and in which it
 * fits `maxW`. The big faces carry only capitals, digits and a little
 * punctuation, so a name with anything else drops to the last face, which is
 * used regardless.
 */
function fitFace(s: string, maxW: number, faces: readonly FaceKey[]): FaceKey {
  const complete = (f: FaceKey) => measureType(f, s) === s.length * measureType(f, "0"); // Plex Mono: every glyph is one width
  return faces.find((f) => complete(f) && measureType(f, s) <= maxW) ?? faces[faces.length - 1];
}

export function drawPlanetPage(day: PlanetOfDay, dateStr: string): Canvas {
  const c = createCanvas();
  const p = day.planet;
  const name = p.name.toUpperCase();

  topBar(
    c, ["Desk", "Planet", "Target of the day"],
    [displayDate(dateStr), `DAY ${dayOfYear(dateStr)}`],
    day.total > 0 ? `${String(day.index + 1).padStart(4, "0")}/${day.total}` : "OFFLINE",
  );

  // Side column: where it is and how it was found
  vline(c, SIDEBAR_W, TOP_BAR_BOTTOM + 1, c.h - TOP_BAR_BOTTOM - 1);
  drawType(c, "plex15m", 20, 64, "[EXOPLANET]");
  fillRect(c, 150, 54, 12, 9);
  dataRow(c, 20, 92, 140, "DISTANCE", formatDistance(p.distanceLy), false);
  dataRow(c, 20, 112, 140, "FOUND", p.year === null ? UNKNOWN : String(p.year), false);
  dataRow(c, 20, 132, 140, "METHOD", METHODS[p.method] ?? (p.method.toUpperCase().slice(0, 10) || UNKNOWN), false);
  hatch(c, 12, 148, 152, 9);

  // The system: its suns as filled discs, its planets as open ones, today's marked
  drawType(c, "plex13r", 20, 182, "SYSTEM");
  const suns = Math.min(p.stars, 3);
  for (let i = 0; i < suns; i++) disc(c, 27 + i * 16, 206, 7, true);
  const shown = Math.min(p.planets, 8);
  const firstX = 27 + suns * 16 + 8;
  const pitch = Math.min(18, Math.floor((158 - firstX) / shown));
  for (let i = 0; i < shown; i++) {
    const x = firstX + i * pitch;
    if (i === day.slot) {
      disc(c, x, 206, 4, true);
      brackets(c, x - 7, 199, 15, 15, 4);
    } else {
      disc(c, x, 206, 3, false);
    }
  }
  label(c, 20, 234, p.host.toUpperCase().slice(0, 17));
  drawType(c, "plex11m", 20, 252, `${p.planets} ${p.planets === 1 ? "PLANET" : "PLANETS"}`);
  drawType(c, "plex11m", 160, 252, `${p.stars} ${p.stars === 1 ? "SUN" : "SUNS"}`, { align: "right" });
  hline(c, 20, 260, 140);

  // Scale: Earth as the open circle, the planet as the filled one, in true proportion
  drawType(c, "plex13r", 20, 284, "SCALE");
  const earthR = Math.max(2, Math.min(16, 46 / p.radius));
  const targetR = Math.min(46, earthR * p.radius);
  disc(c, Math.round(28 + earthR), Math.round(392 - earthR), earthR, false);
  disc(c, Math.round(160 - targetR), Math.round(392 - targetR), targetR, true);
  hline(c, 20, 394, 140);
  label(c, 20, 412, "EARTH");
  drawType(c, "plex11m", 160, 412, "TARGET", { align: "right", spacing: 1 });
  dataRow(c, 20, 440, 140, "RADIUS", `${p.radius.toFixed(2)} X`, true);

  // The planet, on a crosshair that spans the view
  const clear = 12;
  const boxX = CX - BOX;
  const boxY = CY - BOX;
  dotGrid(c, 190, 50, 602, 276, 12, {
    x: boxX - clear, y: boxY - clear, w: BOX * 2 + clear * 2 + 1, h: BOX * 2 + clear * 2 + 1,
  });
  hline(c, SIDEBAR_W + 1, CY, boxX - clear - SIDEBAR_W - 1);
  hline(c, CX + BOX + clear, CY, c.w - (CX + BOX + clear));
  vline(c, CX, TOP_BAR_BOTTOM + 1, boxY - clear - TOP_BAR_BOTTOM - 1);
  vline(c, CX, CY + BOX + clear, CARDS_TOP - (CY + BOX + clear));
  brackets(c, boxX, boxY, BOX * 2 + 1, BOX * 2 + 1);
  drawPlanet(c, CX, CY, PLANET_R, p);

  if (day.tag) {
    const tagW = measureType("plex12s", day.tag) + 16;
    fillRect(c, 190, 44, tagW, 19);
    drawType(c, "plex12s", 198, 58, day.tag, { white: true });
  } else {
    label(c, 190, 56, "TARGET");
  }
  drawType(c, "plex11m", 786, 56, "CONFIRMED", { align: "right", spacing: 1 });
  if (p.ra !== null && p.dec !== null) {
    label(c, 190, 322, formatRa(p.ra));
    drawType(c, "plex11m", 786, 322, formatDec(p.dec), { align: "right", spacing: 1 });
  }

  // Cards
  hline(c, SIDEBAR_W + 1, CARDS_TOP, c.w - SIDEBAR_W - 1);
  hatch(c, 478, 342, 7, 130);
  hatch(c, 642, 342, 7, 130);
  for (const [x0, x1] of [[190, 470], [496, 634], [660, 788]]) {
    for (const y of [348, 468]) { plus(c, x0, y); plus(c, x1, y); }
  }

  label(c, 204, 362, "TARGET INSPECTOR");
  drawType(c, fitFace(name, 256, ["plex36r", "plex28r", "plex15s"]), 204, 414, name);
  dataRow(c, 204, 446, 256, "TYPE", planetType(p), true);

  label(c, 508, 362, "PLANET DATA");
  const facts: [string, string][] = [
    ["MASS", formatMass(p.mass)],
    ["ORBIT", formatPeriod(p.periodDays)],
    ["TEMP", formatTemp(p.tempK)],
    ["STAR", p.starTempK === null ? UNKNOWN : `${Math.round(p.starTempK)} K`],
  ];
  facts.forEach(([key, value], i) => dataRow(c, 508, 386 + i * 21, 118, key, value, true));

  // Found: the year in Aurebesh numerals, and how many planets each year has given us
  label(c, 672, 362, "FOUND");
  if (p.year === null) {
    drawType(c, "plex15s", 672, 394, "--");
  } else {
    const shortW = drawType(c, "af36", 672, 410, String(p.year).slice(2));
    drawType(c, "plex15s", 672 + shortW + 10, 394, String(p.year));
    label(c, 672 + shortW + 10, 410, "YEAR");
  }
  const counts = day.years.filter(([year]) => year >= FIRST_YEAR).slice(0, 38);
  const most = Math.max(1, ...counts.map(([, n]) => n));
  counts.forEach(([year, n], i) => {
    const h = Math.max(2, Math.round(Math.sqrt(n / most) * 30));
    fillRect(c, 672 + i * 3, 458 - h, 2, h);
    if (year === p.year) fillRect(c, 672 + i * 3, 461, 2, 3); // marks this planet's year
  });

  return c;
}
