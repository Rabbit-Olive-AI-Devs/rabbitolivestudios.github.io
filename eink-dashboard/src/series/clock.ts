/**
 * The story clock. A pure function of the time and the schedule, so every panel
 * shows the same page and nothing is stored. Times are America/Chicago.
 *
 * Weekdays: scenes at 08, 10, 12, 14, 16; the 16:00 scene holds overnight.
 * Saturday 08-21 replays scenes 1-14 hourly, Sunday 08-18 replays 15-25,
 * Sunday 19:00 until Monday 08:00 shows the chapter's teaser.
 */

import { shiftDateStr } from "../date-utils";
import type { Schedule, ScheduleSeason, Slot } from "./types";
import { SCENES_PER_CHAPTER } from "./types";

const WEEKDAYS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"];
const SCENE_HOURS = [8, 10, 12, 14, 16];

export function chicagoParts(now: Date): { date: string; weekday: number; hour: number } {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", weekday: "short", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", hourCycle: "h23",
  }).formatToParts(now);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  return {
    date: `${get("year")}-${get("month")}-${get("day")}`,
    weekday: WEEKDAYS.indexOf(get("weekday").toUpperCase().slice(0, 3)),
    hour: Number(get("hour")),
  };
}

/** Whole days from `a` to `b` (YYYY-MM-DD), negative when b is earlier. */
function dayDiff(a: string, b: string): number {
  return Math.round((Date.parse(b + "T00:00:00Z") - Date.parse(a + "T00:00:00Z")) / 86400000);
}

/**
 * Whole Chicago calendar days from `now` to a premiere date: 0 on the day itself,
 * negative once it has passed. Drives the countdown on the "coming soon" card.
 */
export function daysUntil(premiere: string, now: Date): number {
  return dayDiff(chicagoParts(now).date, premiere);
}

function seasonEnd(s: ScheduleSeason): string {
  return shiftDateStr(s.premiere, s.chapters * 7);
}

export function seriesSlot(now: Date, schedule: Schedule): Slot {
  let { date, weekday, hour } = chicagoParts(now);
  // Monday before 08:00 still belongs to Sunday night (the teaser).
  if (weekday === 0 && hour < 8) {
    date = shiftDateStr(date, -1);
    weekday = 6;
    hour = 23;
  }
  for (const s of schedule.seasons) {
    const d = dayDiff(s.premiere, date);
    if (d < 0 || d >= s.chapters * 7) continue;
    const base = { season: s.number, chapter: Math.floor(d / 7) + 1 };
    if (weekday <= 4) {
      if (hour < 8) return { kind: "scene", ...base, scene: weekday * 5, replay: false };
      return { kind: "scene", ...base, scene: weekday * 5 + Math.min(4, Math.floor((hour - 8) / 2)) + 1, replay: false };
    }
    if (weekday === 5) {
      if (hour < 8) return { kind: "scene", ...base, scene: SCENES_PER_CHAPTER, replay: false };
      return { kind: "scene", ...base, scene: Math.min(14, hour - 7), replay: true };
    }
    if (hour < 8) return { kind: "scene", ...base, scene: 14, replay: true };
    if (hour < 19) return { kind: "scene", ...base, scene: hour + 7, replay: true };
    return { kind: "teaser", ...base };
  }
  const byPremiere = [...schedule.seasons].sort((a, b) => a.premiere.localeCompare(b.premiere));
  const next = byPremiere.find((s) => dayDiff(date, s.premiere) > 0);
  const prev = [...byPremiere].reverse().find((s) => dayDiff(seasonEnd(s), date) >= 0);
  return { kind: "between", next: next ? next.number : null, prev: prev ? prev.number : null };
}

/**
 * Only approved chapters air. A slot in a chapter that is not approved shows the
 * last approved chapter's final scene, or the season's "before" card if none is.
 * Previews (includePending) also see pending chapters.
 */
export function resolveSlot(slot: Slot, schedule: Schedule, includePending: boolean): Slot {
  if (slot.kind === "between") return slot;
  const s = schedule.seasons.find((x) => x.number === slot.season);
  if (!s) return { kind: "between", next: null, prev: null };
  const airs = (c: number) => s.approved.includes(c) || (includePending && s.pending.includes(c));
  if (airs(slot.chapter)) return slot;
  const earlier = [...s.approved, ...(includePending ? s.pending : [])].filter((c) => c < slot.chapter);
  if (earlier.length === 0) return { kind: "between", next: s.number, prev: null };
  return { kind: "scene", season: s.number, chapter: Math.max(...earlier), scene: SCENES_PER_CHAPTER, replay: false };
}

const hh = (h: number) => `${String(h).padStart(2, "0")}:00`;

/** When scene n (1-25) first airs in its week, e.g. "WED 10:00". */
export function sceneTimeLabel(scene: number): string {
  const day = Math.floor((scene - 1) / 5);
  return `${WEEKDAYS[day]} ${hh(SCENE_HOURS[(scene - 1) % 5])}`;
}

/** When the scene after n airs: a time today, or the next weekday's 08:00. */
export function nextSceneLabel(scene: number): string {
  const slot = (scene - 1) % 5;
  if (slot < 4) return hh(SCENE_HOURS[slot + 1]);
  const day = Math.floor((scene - 1) / 5);
  return `${day < 4 ? WEEKDAYS[day + 1] : "MON"} 08:00`;
}
