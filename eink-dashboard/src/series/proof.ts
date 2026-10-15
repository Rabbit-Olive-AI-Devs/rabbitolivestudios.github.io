/**
 * The proof export (DECISIONS #78): the exact pages of an uploaded chapter, copied from R2
 * into a private review folder so the showrunner can check every scene without the key.
 * A revision is a digest of the chapter's page bytes, so any change to art, lettering or
 * render gives a new revision, and an APPROVE names the revision it was given for.
 * Pure helpers live here so they are tested; the file and R2 work is in the import script.
 */

import { sceneKey, teaserKey } from "./keys";
import { SCENES_PER_CHAPTER } from "./types";

/** Every page of a chapter as stored in R2: 25 scenes, 25 weekend replays and the teaser. */
export function chapterPageKeys(season: number, chapter: number): string[] {
  const keys: string[] = [];
  for (let s = 1; s <= SCENES_PER_CHAPTER; s++) keys.push(sceneKey(season, chapter, s, false));
  for (let s = 1; s <= SCENES_PER_CHAPTER; s++) keys.push(sceneKey(season, chapter, s, true));
  keys.push(teaserKey(season, chapter));
  return keys;
}

/** The PNG inlined in a finished page (`hudPageHTML`); null if the page carries none. */
export function pagePNG(html: string): Uint8Array | null {
  const m = html.match(/src="data:image\/png;base64,([A-Za-z0-9+/=]+)"/);
  if (!m) return null;
  return Uint8Array.from(atob(m[1]), (c) => c.charCodeAt(0));
}

const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

/** Width and height from a PNG's IHDR; null for anything that is not a PNG. */
export function pngSize(png: Uint8Array): { width: number; height: number } | null {
  if (png.length < 24 || PNG_SIGNATURE.some((b, i) => png[i] !== b)) return null;
  const v = new DataView(png.buffer, png.byteOffset, png.byteLength);
  return { width: v.getUint32(16), height: v.getUint32(20) };
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** A chapter's revision: 16 hex of the SHA-256 over its sorted "key sha256" lines. */
export async function revisionOf(pages: { key: string; sha256: string }[]): Promise<string> {
  const lines = pages.map((p) => `${p.key} ${p.sha256}\n`).sort().join("");
  return (await sha256Hex(new TextEncoder().encode(lines))).slice(0, 16);
}

export const REVISION = /^[0-9a-f]{16}$/;
