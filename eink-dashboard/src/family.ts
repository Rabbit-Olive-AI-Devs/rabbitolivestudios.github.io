/**
 * Family Memory (DECISIONS #92): a family photo from this date in an earlier year, redrawn as art,
 * on `/color/family`.
 *
 *   1. The Mac mini picks the photo the evening before (`scripts/family-pick.py`, run by the
 *      mailbox's art-brief job) and puts it in R2: `family/<date>/photo.jpg` and `photo.json`
 *      (`{"taken":"YYYY-MM-DD","place":"..."}`). The place name is the one exception to "nothing
 *      else leaves the Mac" (Thiago, 2026-10-31). Raw coordinates stay on the Mac. No GPS, or a
 *      failed lookup, sends "Family", never the album name.
 *   2. The brief asks Jarvis to repaint it in the day's family style.
 *   3. The nightly warm stores Jarvis's picture, or the photo itself when he did not deliver, cropped
 *      to the panel with the caption burned in, under `family:v1:<date>`. No Workers AI.
 *   4. When that upload is missing, or both of its files are unreadable, the same warm fills the
 *      same key from the staged pool (`family/staged/`, DECISIONS #106). It never deletes those
 *      objects and never calls Workers AI.
 */

import type { Env } from "./types";
import { familyPortraitStyles, dailyPick, momentStyleFor } from "./art-styles";
import type { ArtStyle } from "./art-styles";
import { parseDateParts, pickSkylineStyle, DEFAULT_ROTATE_MIN } from "./skyline";
import { familyCacheKey } from "./cache-keys";
import { jpegToRGB } from "./image-color";
import { finishSkylinePNG } from "./skyline-image";
import { pngToBase64 } from "./png";
import {
  claimStaged,
  claimsFor,
  familyCacheSource,
  listStagedManifests,
  loadVerifiedStagedAsset,
  markStagedUsed,
  readStagedMarker,
  releaseStagedClaim,
  releaseStagedClaimsFor,
  stagedPickOrder,
} from "./family-staged";
import type { StagedAsset, StagedManifest, StagedRecord } from "./family-staged";

export const FAMILY_TITLE = "Family Memory";

/** The first date the alert expects a family picture for (the feature went live the evening before). */
export const FAMILY_WATCH_FROM = "2026-10-19";

const PHOTO_KEY = (date: string) => `family/${date}/photo.jpg`;
const META_KEY = (date: string) => `family/${date}/photo.json`;
const ART_KEY = (date: string) => `art/${date}/family`;

const FAMILY_STYLES = familyPortraitStyles();
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** The family picture's style for a date: a portrait style that is neither the moment's nor the skyline's. */
export function familyStyleFor(date: string): ArtStyle {
  const avoid = new Set([
    momentStyleFor(date).id,
    pickSkylineStyle(parseDateParts(date), { mode: "daily", rotateMin: DEFAULT_ROTATE_MIN, bucket: 0 }).key,
  ]);
  let i = FAMILY_STYLES.indexOf(dailyPick(FAMILY_STYLES, date, 0x46414d));
  if (i < 0) i = 0;
  for (let n = 0; n < FAMILY_STYLES.length; n++) {
    if (!avoid.has(FAMILY_STYLES[i].id)) return FAMILY_STYLES[i];
    i = (i + 1) % FAMILY_STYLES.length;
  }
  return FAMILY_STYLES[i];
}

/** Whole years between the photo and the date, at least 1 (photos are at least 11 months old). */
export function yearsAgo(taken: string, date: string): number {
  // Calendar years, not full years to the day: the photo may be from a nearby day, and a
  // 19 Oct 2024 photo shown on 18 Oct 2026 is "2 years ago" (owner, 2026-10-18).
  return Math.max(1, Number(date.slice(0, 4)) - Number(taken.slice(0, 4)));
}

const PLACE_MAX = 80;

/** A place label safe to store. Empty, or the album title, becomes "Family". */
export function cleanPlace(value: unknown): string {
  if (typeof value !== "string") return "Family";
  const place = value.replace(/[\u0000-\u001f\u007f]/g, " ").replace(/\s+/g, " ").trim().slice(0, PLACE_MAX);
  if (!place || place === "Family Album") return "Family";
  return place;
}

/** Header values are ASCII. A place with no ASCII letters still reads "Family". */
export function headerPlace(place: string): string {
  const ascii = place.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^\x20-\x7e]/g, "").trim().slice(0, PLACE_MAX);
  return ascii || "Family";
}

export interface FamilyCaptionMeta {
  date: string;
  yearsAgo: number;
  place: string;
  style: string;
}

interface StoredFamily {
  image?: unknown;
  date?: unknown;
  yearsAgo?: unknown;
  place?: unknown;
  style?: unknown;
}

/** KV holds either the old raw base64 or `{image, date, yearsAgo, place, style}`. */
export function parseStoredFamily(raw: string): { image: string; caption: FamilyCaptionMeta | null } {
  if (raw.startsWith("{")) {
    try {
      const o = JSON.parse(raw) as StoredFamily;
      if (typeof o.image === "string" && /^\d{4}-\d{2}-\d{2}$/.test(String(o.date ?? "")) &&
          typeof o.yearsAgo === "number" && Number.isFinite(o.yearsAgo) && typeof o.style === "string" && o.style) {
        return {
          image: o.image,
          caption: { date: String(o.date), yearsAgo: o.yearsAgo, place: cleanPlace(o.place), style: o.style },
        };
      }
      if (typeof o.image === "string") return { image: o.image, caption: null };
    } catch { /* not the wrapper; fall through */ }
  }
  return { image: raw, caption: null };
}

function parsePhotoMeta(raw: unknown): { taken: string; place: string } | null {
  if (!raw || typeof raw !== "object") return null;
  const taken = String((raw as { taken?: unknown }).taken ?? "");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(taken)) return null;
  return { taken, place: cleanPlace((raw as { place?: unknown }).place) };
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
  let taken = "";
  let place = "Family";
  const metaObj = await env.PHOTOS.get(META_KEY(date));
  if (metaObj) {
    try {
      const parsed = parsePhotoMeta(await metaObj.json());
      if (parsed) { taken = parsed.taken; place = parsed.place; }
    } catch { /* the picture still goes in the brief; the place stays Family */ }
  }
  const subject: Record<string, string | number> = {
    occasion: "family memory: repaint the reference photograph (no names, no captions)",
    place,
    style: style.id,
  };
  if (taken) {
    subject.date = taken;
    subject.yearsAgo = yearsAgo(taken, date);
  }
  return {
    kind: "family",
    file: `${date}-family.png`,
    style: { id: style.id, name: style.name },
    prompt: familyPrompt(style),
    subject,
    references: 1,
  };
}

export type FamilySource = "jarvis" | "photo" | "staged" | "staged-photo";

/**
 * Store the day's family picture under `family:v1:<date>`: Jarvis's when he delivered, else the
 * photo itself, else one staged spare. Returns who made it, or null when nothing was stored.
 */
export async function storeFamily(env: Env, date: string): Promise<FamilySource | null> {
  const dated = await storeDatedFamily(env, date);
  if (dated) {
    try { await releaseStagedClaimsFor(env, date); }
    catch (err) { console.error(`Family ${date}: could not release a staged claim:`, err); }
    return dated;
  }
  return storeStagedFallback(env, date);
}

/**
 * The ordinary `family/<date>/` path. Jarvis's repaint when he delivered, else the photo itself.
 * Returns null without writing when the date cannot be stored from those objects.
 */
async function storeDatedFamily(env: Env, date: string): Promise<"jarvis" | "photo" | null> {
  const meta = await env.PHOTOS.get(META_KEY(date));
  let parsed: { taken: string; place: string } | null = null;
  if (meta) {
    try { parsed = parsePhotoMeta(await meta.json()); } catch { /* checked below */ }
    if (!parsed) { console.error(`Family ${date}: bad photo.json`); return null; }
  } else {
    // The first warm may have stored the reference photo and removed its temporary
    // metadata before a late painting arrived. Reuse only this day's validated,
    // already-stored caption; do not retain the private source photo any longer.
    const raw = await env.CACHE.get(familyCacheKey(date));
    // A staged fill is not proof that a dated photo was stored. It must not caption a later painting.
    if (raw && familyCacheSource(raw)) return null;
    const stored = raw ? parseStoredFamily(raw) : null;
    const caption = stored?.caption;
    if (!caption || !stored?.image || caption.style !== familyStyleFor(date).id ||
        !Number.isInteger(caption.yearsAgo) || caption.yearsAgo !== yearsAgo(caption.date, date) ||
        caption.date > date) return null;
    const takenDate = new Date(`${caption.date}T00:00:00Z`);
    if (!Number.isFinite(takenDate.getTime()) || takenDate.toISOString().slice(0, 10) !== caption.date) return null;
    parsed = { taken: caption.date, place: caption.place };
  }
  const { taken, place } = parsed;

  let source: "jarvis" | "photo" | null = null;
  let rgb: Uint8Array | null = null;
  for (const [key, who] of [[ART_KEY(date), "jarvis"], [PHOTO_KEY(date), "photo"]] as const) {
    // Without fresh photo metadata, only the late painting can be promoted.
    if (!meta && who === "photo") continue;
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
  const style = familyStyleFor(date);
  await env.CACHE.put(familyCacheKey(date), JSON.stringify({
    image: pngToBase64(png),
    date: taken,
    yearsAgo: yearsAgo(taken, date),
    place,
    style: style.id,
  }), { expirationTtl: 604800 });
  if (source === "jarvis") await env.PHOTOS.put(`art/used/${date}/family`, new Date().toISOString());
  await Promise.all([PHOTO_KEY(date), META_KEY(date), ART_KEY(date)].map((k) => env.PHOTOS.delete(k)));
  return source;
}

type DatedMeta = "missing" | "bad" | "ok";

async function datedMetaState(env: Env, date: string): Promise<DatedMeta> {
  const meta = await env.PHOTOS.get(META_KEY(date));
  if (!meta) return "missing";
  try {
    return parsePhotoMeta(await meta.json()) ? "ok" : "bad";
  } catch {
    return "bad";
  }
}

async function writeStagedCache(env: Env, date: string, manifest: StagedManifest, asset: StagedAsset): Promise<boolean> {
  try {
    const rgb = await jpegToRGB(env, asset.bytes);
    const png = await finishSkylinePNG(rgb, familyCaption(manifest.taken, date));
    const payload = JSON.stringify({
      image: pngToBase64(png),
      date: manifest.taken,
      yearsAgo: yearsAgo(manifest.taken, date),
      place: cleanPlace(manifest.place),
      style: familyStyleFor(date).id,
      staged: { entry: manifest.date, asset: asset.kind },
    });
    await env.CACHE.put(familyCacheKey(date), payload, { expirationTtl: 604800 });
    return (await env.CACHE.get(familyCacheKey(date))) === payload;
  } catch (err) {
    console.error(`Family ${date}: staged ${manifest.date} unreadable:`, err);
    return false;
  }
}

/** Render a claimed entry, then mark it used. Null means nothing was cached for it. */
async function fulfillClaim(env: Env, date: string, record: StagedRecord): Promise<FamilySource | null> {
  const asset = await loadVerifiedStagedAsset(env, record.manifest);
  if (!asset) return null;
  if (!(await writeStagedCache(env, date, record.manifest, asset))) return null;
  try { await markStagedUsed(env, record.key, date, new Date().toISOString()); }
  catch (err) { console.error(`Family ${date}: staged used-mark failed:`, err); }
  return asset.kind === "repaint" ? "staged" : "staged-photo";
}

async function releaseIfUnstored(env: Env, date: string, record: StagedRecord): Promise<FamilySource | null> {
  const raw = await env.CACHE.get(familyCacheKey(date));
  if (readStagedMarker(raw)?.entry === record.manifest.date) return familyCacheSource(raw);
  await releaseStagedClaim(env, record.key, date);
  return null;
}

/**
 * Fill an empty `family:v1:<date>` from the staged pool. A picture already cached from the
 * ordinary path is left alone. A bad `photo.json` fails closed. Staged objects are not deleted.
 */
async function storeStagedFallback(env: Env, date: string): Promise<FamilySource | null> {
  const existing = await env.CACHE.get(familyCacheKey(date));
  if (existing && !familyCacheSource(existing)) return null;
  if (!existing && (await datedMetaState(env, date)) === "bad") return null;

  const records = await listStagedManifests(env);
  if (existing) {
    const marker = readStagedMarker(existing);
    for (const claim of claimsFor(records, date)) {
      if (marker && claim.manifest.date === marker.entry) {
        try { await markStagedUsed(env, claim.key, date, new Date().toISOString()); }
        catch (err) { console.error(`Family ${date}: staged used-mark failed:`, err); }
      } else {
        await releaseStagedClaim(env, claim.key, date);
      }
    }
    return familyCacheSource(existing);
  }

  for (const claim of claimsFor(records, date)) {
    const done = await fulfillClaim(env, date, claim);
    if (done) {
      for (const other of claimsFor(records, date)) {
        if (other.key !== claim.key) await releaseStagedClaim(env, other.key, date);
      }
      return done;
    }
    const kept = await releaseIfUnstored(env, date, claim);
    if (kept) return kept;
  }

  for (const candidate of stagedPickOrder(records, date)) {
    if (!(await loadVerifiedStagedAsset(env, candidate.manifest))) continue;
    const claimed = await claimStaged(env, candidate.key, date);
    if (!claimed) continue;
    const done = await fulfillClaim(env, date, claimed);
    if (done) return done;
    const kept = await releaseIfUnstored(env, date, claimed);
    if (kept) return kept;
  }
  return null;
}
