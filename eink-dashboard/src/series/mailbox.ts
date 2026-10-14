/**
 * The showrunner's mailbox (DECISIONS #72). Jarvis writes one Markdown file per
 * message into Dropbox `00_SHARED_INBOX/series/to-claude/`; its first line says what
 * it is. The watcher (`scripts/series-mailbox.mjs`) acts on the routine kinds without
 * AI and hands the rest to a read-only Claude run. Pure parsing lives here so it is
 * tested; the file and process work stays in the script.
 */

import { shiftDateStr } from "../date-utils";
import type { Schedule } from "./types";

export type Message =
  | { kind: "APPROVE"; season: number; chapters: number[] }
  | { kind: "DELIVERY"; package: string; replaceApproved: boolean }
  | { kind: "STYLE TEST"; folder: string }
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
    return { kind: "APPROVE", season: Number(season), chapters: list.split(",").map(Number) };
  }
  if (upper.startsWith("DELIVERY")) {
    const pkg = param("package");
    if (!pkg) return { kind: "INVALID", reason: "DELIVERY needs package=<zip in packages/> on its first line" };
    if (!SAFE_NAME.test(pkg)) return { kind: "INVALID", reason: `package=${pkg} must be a plain file name in packages/` };
    return { kind: "DELIVERY", package: pkg, replaceApproved: /\breplace-approved\b/i.test(first) };
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
