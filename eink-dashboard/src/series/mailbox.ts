/**
 * The showrunner's mailbox (DECISIONS #72). Jarvis writes one Markdown file per
 * message into Dropbox `00_SHARED_INBOX/series/to-claude/`; its first line says what
 * it is. The watcher (`scripts/series-mailbox.mjs`) acts on the routine kinds without
 * AI and hands the rest to a read-only Claude run. Pure parsing lives here so it is
 * tested; the file and process work stays in the script.
 */

import { shiftDateStr } from "../date-utils";
import { REVISION } from "./proof";
import type { Schedule } from "./types";
import { plannedChapters } from "./types";

export type Message =
  | { kind: "APPROVE"; season: number; chapters: number[]; revisions: string[] }
  | { kind: "DELIVERY"; package: string; replaceApproved: boolean }
  | { kind: "STYLE TEST"; folder: string }
  | { kind: "ART DELIVERY"; date: string }
  | { kind: "QUESTION" | "FYI" | "OTHER" }
  | { kind: "INVALID"; reason: string };

/** A plain file or folder name: no paths, nothing that could climb out of the mailbox. */
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

export function parseMessage(text: string): Message {
  const first = text.split("\n").map((l) => l.trim()).find((l) => l.length > 0) ?? "";
  const upper = first.toUpperCase();
  const param = (name: string) => first.match(new RegExp(`\\b${name}=(\\S+)`, "i"))?.[1];

  if (upper.startsWith("APPROVE")) {
    const season = param("season");
    const list = param("chapters") ?? param("chapter");
    if (season === undefined || !/^\d+$/.test(season)) return { kind: "INVALID", reason: "APPROVE needs season=N on its first line" };
    if (!list || !/^\d+(,\d+)*$/.test(list)) return { kind: "INVALID", reason: "APPROVE needs chapter=N (or chapters=N,M) on its first line" };
    // The revision of the proof export reviewed (DECISIONS #78): pages changed since then are not approved.
    const revs = (param("revisions") ?? param("revision"))?.toLowerCase().split(",") ?? [];
    const chapters = list.split(",").map(Number);
    if (revs.length !== chapters.length || !revs.every((r) => REVISION.test(r))) {
      return { kind: "INVALID", reason: "APPROVE needs revision=<16 hex> from the proof export (revisions=A,B for chapters=N,M, in the same order)" };
    }
    return { kind: "APPROVE", season: Number(season), chapters, revisions: revs };
  }
  if (upper.startsWith("DELIVERY")) {
    const pkg = param("package");
    if (!pkg) return { kind: "INVALID", reason: "DELIVERY needs package=<zip in packages/> on its first line" };
    if (!SAFE_NAME.test(pkg)) return { kind: "INVALID", reason: `package=${pkg} must be a plain file name in packages/` };
    return { kind: "DELIVERY", package: pkg, replaceApproved: /\breplace-approved\b/i.test(first) };
  }
  if (upper.startsWith("ART DELIVERY")) {
    // Jarvis's daily pictures (DECISIONS #89): the files are to-claude/art/<date>-<kind>.<ext>.
    const date = param("date");
    if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return { kind: "INVALID", reason: "ART DELIVERY needs date=YYYY-MM-DD on its first line" };
    return { kind: "ART DELIVERY", date };
  }
  if (upper.startsWith("STYLE TEST")) {
    const folder = param("folder");
    if (!folder) return { kind: "INVALID", reason: "STYLE TEST needs folder=<name in style-tests/> on its first line" };
    if (!SAFE_NAME.test(folder)) return { kind: "INVALID", reason: `folder=${folder} must be a plain folder name in style-tests/` };
    return { kind: "STYLE TEST", folder };
  }
  if (upper.startsWith("QUESTION")) return { kind: "QUESTION" };
  if (upper.startsWith("FYI")) return { kind: "FYI" };
  return { kind: "OTHER" };
}

/** Chapters are delivered at least 12 days ahead; a weekly run starts each one 19 days ahead. */
const LEAD_DAYS = 19;

/**
 * The next chapter Claude must write for a season it is writing, once it is within
 * LEAD_DAYS of airing and nothing for it has been uploaded. One at a time: the next
 * one becomes due only after this one is uploaded.
 */
export function dueClaudeChapter(schedule: Schedule, today: string): { season: number; chapter: number; airs: string } | null {
  for (const s of schedule.seasons) {
    if (s.writer !== "Claude") continue;
    const uploaded = [...s.approved, ...s.pending];
    const next = uploaded.length ? Math.max(...uploaded) + 1 : 1;
    if (next > s.chapters) continue;
    const airs = shiftDateStr(s.premiere, (next - 1) * 7);
    if (airs <= shiftDateStr(today, LEAD_DAYS)) return { season: s.number, chapter: next, airs };
  }
  return null;
}

/** The decided calendar (DECISIONS #71): who writes each season and the Monday it premieres. */
export const CALENDAR: readonly { number: number; writer: string; premiere: string }[] = [
  // Chapters 1-3 are Claude's and stay so; Jarvis writes the rest, and Season 4 (owner, 2026-10-23, DECISIONS #98).
  { number: 0, writer: "Jarvis", premiere: "2026-10-26" },
  { number: 1, writer: "Jarvis", premiere: "2027-01-04" }, // in place of Zoe (owner, 2026-10-15)
  { number: 2, writer: "Volt", premiere: "2027-04-05" },
  { number: 3, writer: "Pepper", premiere: "2027-07-05" },
  { number: 4, writer: "Jarvis", premiere: "2027-10-04" },
];

/** The showrunner, who approves every chapter (DECISIONS #71). */
export const SHOWRUNNER = "Jarvis";

/**
 * A season the showrunner writes himself cannot be cleared by him alone: each chapter also
 * needs Claude's independent PASS for the same proof revision (owner, 2026-10-15; DECISIONS #79).
 */
export function needsIndependentReview(season: number): boolean {
  return CALENDAR.find((c) => c.number === season)?.writer === SHOWRUNNER;
}

/** Production milestones, in days before a premiere (docs/series/SHOWRUNNER.md section 5). */
const MILESTONES: [number, string][] = [
  [63, "writer briefed"],
  [56, "style test done (the look is the writer's; Jarvis accepts it); character and ship sheets reviewed"],
  [49, "season outline approved"],
  [28, "chapters 1-4 scripted, drawn and reviewed"],
  [21, "first delivery to Claude: chapters 1 and 2"],
];

const label = (n: number) => (n === 0 ? "Prologue" : `Season ${n}`);

/**
 * The Monday production heartbeat (DECISIONS #76): what is on air, what must be
 * delivered or approved this week, what is late, and the milestones of the next two
 * weeks. Plain text for the mailbox; no AI. `today` is YYYY-MM-DD (Chicago).
 */
export function heartbeat(schedule: Schedule, today: string): string {
  const weekEnd = shiftDateStr(today, 6);
  const onAir: string[] = [], due: string[] = [], late: string[] = [], soon: string[] = [];
  for (const cal of CALENDAR) {
    const s = schedule.seasons.find((x) => x.number === cal.number);
    const who = `${label(cal.number)} (${cal.writer})`;
    const chapters = s?.chapters ?? plannedChapters(cal.number);
    for (let c = 1; c <= chapters; c++) {
      const airs = shiftDateStr(cal.premiere, (c - 1) * 7);
      const deliverBy = c <= 2 && cal.number > 0 ? shiftDateStr(cal.premiere, -21) : shiftDateStr(airs, -12);
      const approveBy = shiftDateStr(airs, -8);
      const approved = s?.approved.includes(c) ?? false;
      const uploaded = approved || (s?.pending.includes(c) ?? false);
      if (airs >= today && airs <= weekEnd) onAir.push(`${label(cal.number)} chapter ${c} airs Monday ${airs}${approved ? "" : " (NOT APPROVED: it will not air)"}`);
      if (!uploaded && deliverBy < today && airs >= today) late.push(`${who}: chapter ${c} should have been delivered by ${deliverBy} (airs ${airs})`);
      else if (!uploaded && deliverBy >= today && deliverBy <= weekEnd) due.push(`${who}: deliver chapter ${c} by ${deliverBy} (airs ${airs})`);
      if (uploaded && !approved && approveBy < today && airs >= today) late.push(`Jarvis: approve ${label(cal.number)} chapter ${c}, due ${approveBy} (airs ${airs})`);
      else if (uploaded && !approved && approveBy >= today && approveBy <= weekEnd) due.push(`Jarvis: approve ${label(cal.number)} chapter ${c} by ${approveBy}`);
    }
    for (const [days, what] of MILESTONES) {
      const when = shiftDateStr(cal.premiere, -days);
      if (when >= today && when <= shiftDateStr(today, 13)) soon.push(`${who}: ${when}, ${what}`);
    }
  }
  const section = (title: string, lines: string[]) => (lines.length ? `${title}\n${lines.map((l) => `- ${l}`).join("\n")}\n` : "");
  const body = [section("LATE", late), section("Due this week", due), section("On air this week", onAir), section("Milestones in the next two weeks", soon)].filter(Boolean);
  return `FYI: production heartbeat, week of ${today}\n\nJarvis: this note comes every Monday at 08:00 Chicago. Use it to nudge the writers.\n\n${body.length ? body.join("\n") : "Nothing is due, late or on air this week.\n"}`;
}
