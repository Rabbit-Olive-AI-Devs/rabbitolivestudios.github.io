/**
 * What the six-hourly alert check watches for the series (spec section 7):
 * a chapter airing within 7 days that is not approved, a season ending within 14
 * days with nothing after it, and the page for "now" missing from R2. R2 reads
 * only, never KV.
 */

import { displayDate } from "../hud/chrome";
import { shiftDateStr } from "../date-utils";
import { loadSchedule } from "../pages/series";
import type { Env } from "../types";
import { chicagoParts, resolveSlot, seriesSlot } from "./clock";
import { slotKey } from "./keys";
import type { Schedule, ScheduleSeason, Slot } from "./types";

const name = (s: ScheduleSeason) => (s.number === 0 ? "the Prologue" : `season ${s.number}`);
const days = (a: string, b: string) => Math.round((Date.parse(b + "T00:00:00Z") - Date.parse(a + "T00:00:00Z")) / 86400000);

export function scheduleProblems(schedule: Schedule | null, now: Date): string[] {
  if (!schedule) return [];
  const today = chicagoParts(now).date;
  const out: string[] = [];
  for (const s of schedule.seasons) {
    for (let c = 1; c <= s.chapters; c++) {
      const airs = shiftDateStr(s.premiere, (c - 1) * 7);
      const until = days(today, airs);
      if (until < 0 || until > 7 || s.approved.includes(c)) continue;
      const state = s.pending.includes(c) ? "is waiting for approval" : "is not imported";
      out.push(`The series: chapter ${c} of ${name(s)} airs ${displayDate(airs).slice(0, 5)} and ${state}`);
    }
    const end = shiftDateStr(s.premiere, s.chapters * 7 - 3); // the last Friday
    const left = days(today, end);
    const later = schedule.seasons.some((x) => x.premiere > s.premiere);
    if (left >= 0 && left <= 14 && !later) {
      out.push(`The series: ${name(s)} ends ${displayDate(end).slice(0, 5)} and no next season is imported`);
    }
  }
  return out;
}

export async function seriesProblems(env: Env, now: Date): Promise<string[]> {
  if (!env.SERIES) return [];
  const schedule = await loadSchedule(env.SERIES);
  const out = scheduleProblems(schedule, now);
  if (schedule) {
    const key = slotKey(resolveSlot(seriesSlot(now, schedule), schedule, false), "office");
    if (key && !(await env.SERIES.get(key))) out.push(`The series: the page on the panels now is missing (${key})`);
  }
  return out;
}

export async function seriesHealth(env: Env, now: Date): Promise<{ schedule: boolean; slot: Slot | null; key: string | null; present: boolean }> {
  const schedule = env.SERIES ? await loadSchedule(env.SERIES) : null;
  if (!schedule) return { schedule: false, slot: null, key: null, present: false };
  const slot = resolveSlot(seriesSlot(now, schedule), schedule, false);
  const key = slotKey(slot, "office");
  const present = key ? (await env.SERIES.get(key)) !== null : false;
  return { schedule: true, slot, key, present };
}
