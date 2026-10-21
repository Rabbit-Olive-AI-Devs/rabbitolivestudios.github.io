/**
 * The desk serial (docs/superpowers/specs/2026-10-21-desk-serial-design.md): each beat is a page
 * built offline on the Mac mini (scripts/serial-build.mjs) and stored in the private series bucket.
 * The Worker only picks the current beat and passes its HTML through: no drawing, no KV.
 */

import type { Env } from "./types";
import { chicagoParts } from "./castaway";
import { shiftDateStr } from "./date-utils";

export interface SerialBeat {
  id: string;
  /** Chicago clock, "HH:MM". */
  start: string;
  key: string;
}

export interface SerialDay {
  date: string;
  day: number;
  beats: SerialBeat[];
}

export const serialDayKey = (date: string) => `serial/days/${date}.json`;
export const serialPageKey = (id: string) => `serial/pages/${id}.html`;

const minutes = (hhmm: string) => Number(hhmm.slice(0, 2)) * 60 + Number(hhmm.slice(3, 5));

/** The beat showing at `min` (minutes after Chicago midnight): today's latest started, else yesterday's last. */
export function currentBeat(today: SerialDay | null, yesterday: SerialDay | null, min: number): SerialBeat | null {
  const latest = (beats: SerialBeat[], before: number) =>
    beats.filter((b) => minutes(b.start) <= before).sort((a, b) => minutes(a.start) - minutes(b.start)).at(-1) ?? null;
  return latest(today?.beats ?? [], min) ?? latest(yesterday?.beats ?? [], 24 * 60);
}

async function readDay(env: Env, date: string): Promise<SerialDay | null> {
  try {
    const obj = await env.SERIES.get(serialDayKey(date));
    if (!obj) return null;
    const day = (await obj.json()) as SerialDay;
    return Array.isArray(day?.beats) ? day : null;
  } catch (err) {
    console.error(`serial: day ${date} unreadable:`, err);
    return null;
  }
}

/** The current beat's stored page, or null (the caller falls back to the engine). */
export async function serialPage(env: Env, now: Date): Promise<string | null> {
  const { date, min } = chicagoParts(now);
  const today = await readDay(env, date);
  const startedToday = today?.beats.some((b) => minutes(b.start) <= min);
  const yesterday = startedToday ? null : await readDay(env, shiftDateStr(date, -1));
  const beat = currentBeat(today, yesterday, min);
  if (!beat) return null;
  try {
    const obj = await env.SERIES.get(beat.key);
    if (!obj) {
      console.error(`serial: missing page ${beat.key}`);
      return null;
    }
    return await obj.text();
  } catch (err) {
    console.error("serial: page read failed:", err);
    return null;
  }
}
