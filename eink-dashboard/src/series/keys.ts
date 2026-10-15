/** Where every finished page lives in the private R2 bucket. */

import type { PanelId, Slot } from "./types";

export const SCHEDULE_KEY = "series/schedule.json";

const pad = (n: number) => String(n).padStart(2, "0");

export function sceneKey(season: number, chapter: number, scene: number, replay: boolean, panel: PanelId): string {
  return `series/s${pad(season)}/c${pad(chapter)}/${replay ? "r" : "s"}${pad(scene)}-${panel}.png`;
}

export function teaserKey(season: number, chapter: number, panel: PanelId): string {
  return `series/s${pad(season)}/c${pad(chapter)}/teaser-${panel}.png`;
}

export function cardKey(season: number, which: "before" | "after", panel: PanelId): string {
  return `series/s${pad(season)}/${which}-${panel}.png`;
}

/** The page for a slot; null when there is nothing to show (no season known). */
export function slotKey(slot: Slot, panel: PanelId): string | null {
  if (slot.kind === "scene") return sceneKey(slot.season, slot.chapter, slot.scene, slot.replay, panel);
  if (slot.kind === "teaser") return teaserKey(slot.season, slot.chapter, panel);
  if (slot.next !== null) return cardKey(slot.next, "before", panel);
  if (slot.prev !== null) return cardKey(slot.prev, "after", panel);
  return null;
}

const PLAIN = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** A writer's style-test picture converted for one panel (DECISIONS #72); null for unsafe names. */
export function testKey(folder: string, image: string, panel: PanelId): string | null {
  if (!PLAIN.test(folder) || !PLAIN.test(image)) return null;
  return `series/tests/${folder}/${image}-${panel}.png`;
}
