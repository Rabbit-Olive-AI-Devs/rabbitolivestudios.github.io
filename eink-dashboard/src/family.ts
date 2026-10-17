/**
 * Family Memory (DECISIONS #92): a family photo from this date in an earlier year, redrawn as art,
 * on `/color/family`.
 *
 *   1. The Mac mini picks the photo the evening before (`scripts/family-pick.py`, run by the
 *      mailbox's art-brief job) and puts it in R2: `family/<date>/photo.jpg` and `photo.json`
 *      (`{"taken":"YYYY-MM-DD"}`, nothing else about the photo leaves the Mac).
 *   2. The brief asks Jarvis to repaint it in the day's family style.
 *   3. The nightly warm stores Jarvis's picture, or the photo itself when he did not deliver, cropped
 *      to the panel with the caption burned in, under `family:v1:<date>`. No Workers AI.
 */

import type { Env } from "./types";
import { stylesFor, dailyPick, momentStyleFor } from "./art-styles";
import type { ArtStyle } from "./art-styles";
import { parseDateParts, pickSkylineStyle, DEFAULT_ROTATE_MIN } from "./skyline";
import { familyCacheKey } from "./cache-keys";
import { jpegToRGB } from "./image-color";
import { finishSkylinePNG } from "./skyline-image";
import { pngToBase64 } from "./png";

export const FAMILY_TITLE = "Family Memory";

const PHOTO_KEY = (date: string) => `family/${date}/photo.jpg`;
const META_KEY = (date: string) => `family/${date}/photo.json`;
const ART_KEY = (date: string) => `art/${date}/family`;

const FAMILY_STYLES = stylesFor("portrait");
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** The family picture's style for a date: a portrait style that is neither the moment's nor the skyline's. */
export function familyStyleFor(date: string): ArtStyle {
  const avoid = new Set([
    momentStyleFor(date).id,
    pickSkylineStyle(parseDateParts(date), { mode: "daily", rotateMin: DEFAULT_ROTATE_MIN, bucket: 0 }).key,
  ]);
  let i = FAMILY_STYLES.indexOf(dailyPick(FAMILY_STYLES, date, 0x46414d));
  while (avoid.has(FAMILY_STYLES[i].id)) i = (i + 1) % FAMILY_STYLES.length;
  return FAMILY_STYLES[i];
}

/** Whole years between the photo and the date, at least 1 (photos are at least 11 months old). */
export function yearsAgo(taken: string, date: string): number {
  const [ty, tm, td] = taken.split("-").map(Number);
  const [y, m, d] = date.split("-").map(Number);
  const years = y - ty - (m < tm || (m === tm && d < td) ? 1 : 0);
  return Math.max(1, years);
}

/** The burned-in caption: the photo's real date, the title, how long ago. No names. */
export function familyCaption(taken: string, date: string): { left: string; center: string; right: string } {
  const [ty, tm, td] = taken.split("-").map(Number);
  const n = yearsAgo(taken, date);
  return { left: `${td} ${MONTHS[tm - 1]} ${ty}`, center: FAMILY_TITLE, right: `${n} year${n === 1 ? "" : "s"} ago` };
}

/** What Jarvis is asked to draw; the reference photograph travels with the brief. */
export function familyPrompt(style: ArtStyle): string {
  return `A faithful repainting of the reference photograph as ${style.prompt}: the same people, faces, expressions, ` +
    "poses, clothing and composition, keeping each person's exact likeness; only the medium changes. " +
    "No added people, no text, no letters, no signature, no borders.";
}

/** The brief's family picture for a date, or null when the Mac mini has not put a photo up. */
export async function familyPicture(env: Env, date: string): Promise<{
  kind: "family"; file: string; style: { id: string; name: string }; prompt: string;
  subject: Record<string, string | number>; references: number;
} | null> {
  if (!(await env.PHOTOS.head(PHOTO_KEY(date)))) return null;
  const style = familyStyleFor(date);
  return {
    kind: "family",
    file: `${date}-family.png`,
    style: { id: style.id, name: style.name },
    prompt: familyPrompt(style),
    subject: { occasion: "family memory: repaint the reference photograph (no names, no captions)" },
    references: 1,
  };
}

/**
 * Store the day's family picture under `family:v1:<date>`: Jarvis's when he delivered, else the
 * photo itself. Returns who made it, or null when no photo was put up for the date (nothing stored).
 */
export async function storeFamily(env: Env, date: string): Promise<"jarvis" | "photo" | null> {
  const meta = await env.PHOTOS.get(META_KEY(date));
  if (!meta) return null;
  let taken = "";
  try { taken = String((await meta.json<{ taken?: unknown }>()).taken ?? ""); } catch { /* checked below */ }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(taken)) { console.error(`Family ${date}: bad photo.json`); return null; }

  let source: "jarvis" | "photo" | null = null;
  let rgb: Uint8Array | null = null;
  for (const [key, who] of [[ART_KEY(date), "jarvis"], [PHOTO_KEY(date), "photo"]] as const) {
    const obj = await env.PHOTOS.get(key);
    if (!obj) continue;
    try {
      rgb = await jpegToRGB(env, new Uint8Array(await obj.arrayBuffer()));
      source = who;
      break;
    } catch (err) {
      console.error(`Family ${date}: ${key} unreadable:`, err);
    }
  }
  if (!rgb || !source) return null;

  const png = await finishSkylinePNG(rgb, familyCaption(taken, date));
  await env.CACHE.put(familyCacheKey(date), pngToBase64(png), { expirationTtl: 604800 });
  if (source === "jarvis") await env.PHOTOS.put(`art/used/${date}/family`, new Date().toISOString());
  await Promise.all([PHOTO_KEY(date), META_KEY(date), ART_KEY(date)].map((k) => env.PHOTOS.delete(k)));
  return source;
}
