/**
 * The schedule kept in R2 (`series/schedule.json`): which seasons exist and which
 * chapters are approved or pending. Shared by the Worker (reading it) and the
 * import CLI (updating it), so both agree on what a valid schedule is.
 */

import type { Schedule, ScheduleSeason, SeasonInfo } from "./types";
import { plannedChapters } from "./types";

const isIntList = (x: unknown) => Array.isArray(x) && x.every((n) => Number.isInteger(n));

/** True when every season has the fields the clock and the routes rely on. */
export function isSchedule(x: unknown): x is Schedule {
  const s = x as Schedule;
  return !!s && Array.isArray(s.seasons) && s.seasons.every((e: ScheduleSeason) =>
    !!e && Number.isInteger(e.number) && typeof e.premiere === "string" && /^\d{4}-\d{2}-\d{2}$/.test(e.premiere)
    && Number.isInteger(e.chapters) && isIntList(e.approved) && isIntList(e.pending));
}

/**
 * Fold an upload into the schedule. Chapters already approved are not replaced
 * unless `replaceApproved` (the showrunner approved the correction); their pages
 * would otherwise go on air unreviewed. New chapters become pending.
 */
export function mergeUpload(
  schedule: Schedule, info: SeasonInfo, chapters: number[], replaceApproved: boolean,
): { schedule: Schedule; upload: number[]; skipped: number[] } {
  const seasons = schedule.seasons.map((s) => ({ ...s, approved: [...s.approved], pending: [...s.pending] }));
  let entry = seasons.find((s) => s.number === info.number);
  if (!entry) {
    entry = { ...info, chapters: plannedChapters(info.number), approved: [], pending: [] };
    seasons.push(entry);
  }
  Object.assign(entry, info, { chapters: plannedChapters(info.number) });
  const skipped = replaceApproved ? [] : chapters.filter((c) => entry!.approved.includes(c));
  const upload = chapters.filter((c) => !skipped.includes(c));
  for (const c of upload) {
    if (!entry.approved.includes(c) && !entry.pending.includes(c)) entry.pending.push(c);
  }
  entry.pending.sort((a, b) => a - b);
  return { schedule: { seasons }, upload, skipped };
}
