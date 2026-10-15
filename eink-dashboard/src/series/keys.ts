/**
 * Where every finished page lives in the private R2 bucket. Each key holds a complete
 * one-request HTML document, the same for every panel: the panels convert the full-colour
 * page themselves (DECISIONS #73), so the Worker only passes it through.
 */

import type { Slot } from "./types";

export const SCHEDULE_KEY = "series/schedule.json";

const pad = (n: number) => String(n).padStart(2, "0");

export function sceneKey(season: number, chapter: number, scene: number, replay: boolean): string {
  return `series/s${pad(season)}/c${pad(chapter)}/${replay ? "r" : "s"}${pad(scene)}.html`;
}

export function teaserKey(season: number, chapter: number): string {
  return `series/s${pad(season)}/c${pad(chapter)}/teaser.html`;
}

export function cardKey(season: number, which: "before" | "after"): string {
  return `series/s${pad(season)}/${which}.html`;
}

/** The page for a slot; null when there is nothing to show (no season known). */
export function slotKey(slot: Slot): string | null {
  if (slot.kind === "scene") return sceneKey(slot.season, slot.chapter, slot.scene, slot.replay);
  if (slot.kind === "teaser") return teaserKey(slot.season, slot.chapter);
  if (slot.next !== null) return cardKey(slot.next, "before");
  if (slot.prev !== null) return cardKey(slot.prev, "after");
  return null;
}

const PLAIN = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** A writer's style-test picture as a finished page (DECISIONS #72); null for unsafe names. */
export function testKey(folder: string, image: string): string | null {
  if (!PLAIN.test(folder) || !PLAIN.test(image)) return null;
  return `series/tests/${folder}/${image}.html`;
}
