/**
 * Pictures drawn by Jarvis (DECISIONS #89). The owner's agent Jarvis makes the daily pictures
 * with GPT image generation; Workers AI stays as the backup.
 *
 *   1. The brief (`GET /art/brief?date=`) says what to draw for a date: the moment (or, on a
 *      family birthday, the portrait) and the skyline, with the same event, styles and prompt
 *      wording the Worker would use itself. The Mac mini's mailbox watcher writes it to the
 *      series mailbox as JSON; Jarvis turns it into his own prompts.
 *   2. Jarvis drops the pictures in the mailbox; the watcher puts them in R2 at
 *      `art/DATE/KIND` (PHOTOS bucket) with its own Cloudflare login. No secret is involved.
 *   3. The nightly warm takes a picture from R2 when one is there, crops and captions it like
 *      its own, and stores it under the usual cache key. When none is there, Workers AI makes it.
 */

import type { Env, MomentBeforeData } from "./types";
import { getTodayEvents } from "./fact";
import { getOrGenerateMoment, eventCandidates } from "./moment";
import { momentCacheKey, skylineCacheKey } from "./cache-keys";
import { getAiBudgetBlock } from "./cache-guard";
import { momentStyleFor } from "./art-styles";
import { getBirthdayToday, getBirthdayStyle } from "./birthday";
import type { BirthdayPerson } from "./birthday";
import { ageDescription, likenessPrompt } from "./birthday-image";
import { parseDateParts, pickSkylineCity, pickSkylineStyle, buildSkylinePrompt, formatSkylineCaption, DEFAULT_ROTATE_MIN } from "./skyline";
import { jpegToRGB } from "./image-color";
import { finishSkylinePNG } from "./skyline-image";
import { saveColorEntry } from "./pages/color-moment";
import { encodePNGRGB, pngToBase64 } from "./png";
import { WIDTH, HEIGHT } from "./image";

export type ArtKind = "moment" | "skyline" | "birthday";

const ANTI_TEXT_SUFFIX = "no text, no words, no letters, no writing, no signage, no captions, no watermark";

/** Where the watcher puts Jarvis's picture for a date and kind. */
export function artKey(dateStr: string, kind: ArtKind): string {
  return `art/${dateStr}/${kind}`;
}

export interface ArtPicture {
  kind: ArtKind;
  file: string;             // the name Jarvis gives the picture in to-claude/art/
  style: { id: string; name: string };
  prompt: string;           // the wording the Worker itself would send its image model
  subject: Record<string, string | number>;
  references?: number;      // birthday: how many face photos come with the brief
  chooseEvent?: {           // moment, when the Worker could not pick the event: Jarvis picks it
    instructions: string;
    candidates: Array<{ year: number; text: string }>;
  };
}

export interface ArtBrief {
  date: string;
  deliverBy: string;
  rules: string[];
  pictures: ArtPicture[];
  waiting?: ArtKind[];      // pictures that cannot be briefed yet (the day's event is not picked)
}

/** Rules every picture follows; the panels convert full colour themselves (DECISIONS #81). */
export const ART_RULES: readonly string[] = [
  "Landscape, 3:2 or wider, at least 1536x1024 pixels; the panel shows 800x480 (5:3) and crops the rest from the centre, so keep the subject in the middle.",
  "Full colour, as rich as the style allows. Do not make it black and white for the mono panel: each panel converts the picture itself.",
  "Follow the style exactly; it is chosen by a schedule so that no style repeats.",
  "No text, letters, numbers, signatures, seals, logos or watermarks anywhere. The caption is added afterwards.",
  "No frames or borders around the picture.",
  "Historical scenes: accurate to the period; no graphic injury or gore.",
];

/** What Jarvis is told when he picks the day's event himself (the Worker's own LLM rules, #7, #75). */
export const CHOOSE_EVENT =
  "Pick the one event from the candidates that makes the most striking picture: a famous, visual moment, " +
  "preferably 1800-2000. Draw the event itself at its defining moment of action, not the calm before or the aftermath, " +
  "historically accurate to the period (architecture, vehicles, clothing, technology). No graphic injury, bodies, blood " +
  "or close-up suffering; show the scene and its scale. Then, in the ART DELIVERY note, add three lines: " +
  "'title: <2-5 word name of the event>', 'year: <year>', 'place: <where it happened>'. The panel's caption is made from them.";

/** The skyline part of the brief (pure: city, style and prompt come from the date). */
export function skylinePicture(dateStr: string): ArtPicture {
  const parts = parseDateParts(dateStr);
  const opts = { mode: "daily" as const, rotateMin: DEFAULT_ROTATE_MIN, bucket: 0 };
  const city = pickSkylineCity(parts, opts);
  const style = pickSkylineStyle(parts, opts);
  return {
    kind: "skyline",
    file: `${dateStr}-skyline.png`,
    style: { id: style.key, name: style.label },
    prompt: buildSkylinePrompt(city, style),
    subject: { city: city.name, landmarks: city.landmarks },
  };
}

/**
 * The brief for a date. The moment needs the day's event, one LLM call when it is not cached;
 * none is attempted while the AI budget is blocked, and a refused call is not cached (#89). Then
 * the moment is listed under `waiting` and a later brief carries it; the skyline never waits.
 */
export async function buildArtBrief(env: Env, dateStr: string): Promise<ArtBrief> {
  const [year, month, day] = dateStr.split("-").map(Number);
  const pictures: ArtPicture[] = [];
  const birthday = getBirthdayToday(month, day);
  if (birthday) {
    const style = getBirthdayStyle(birthday, year);
    pictures.push({
      kind: "birthday",
      file: `${dateStr}-birthday.png`,
      style: { id: style.id, name: style.name },
      prompt: likenessPrompt("the person shown in the reference photos (the same person in every photo)", ageDescription(birthday, year), style.prompt),
      subject: { name: birthday.name, key: birthday.key, age: year - birthday.birthYear, occasion: "birthday portrait" },
      references: 4,
    });
  }
  const waiting: ArtKind[] = [];
  const moment = birthday ? null : await briefMoment(env, dateStr);
  if (!birthday && !moment) {
    // The event could not be picked (the LLM is refused or paused). Jarvis picks it from the
    // day's events instead and reports what he drew, so the caption stays right (#89).
    const { events } = await getTodayEvents(env, dateStr);
    const candidates = eventCandidates(events);
    const style = momentStyleFor(dateStr);
    if (candidates.length === 0) waiting.push("moment");
    else pictures.push({
      kind: "moment",
      file: `${dateStr}-moment.png`,
      style: { id: style.id, name: style.name },
      prompt: `${style.prompt}, [the scene you choose: the event itself at its defining moment of action], ${ANTI_TEXT_SUFFIX}`,
      subject: { note: "choose the event from chooseEvent.candidates" },
      chooseEvent: { instructions: CHOOSE_EVENT, candidates },
    });
  }
  if (moment) {
    const style = momentStyleFor(dateStr);
    pictures.push({
      kind: "moment",
      file: `${dateStr}-moment.png`,
      style: { id: style.id, name: style.name },
      prompt: `${style.prompt}, ${moment.imagePrompt}, ${ANTI_TEXT_SUFFIX}`,
      subject: { event: moment.title, year: moment.year, place: moment.location, scene: moment.scene },
    });
  }
  pictures.push(skylinePicture(dateStr));
  return {
    date: dateStr, deliverBy: "22:30 America/Chicago the evening before the date", rules: [...ART_RULES], pictures,
    ...(waiting.length ? { waiting } : {}),
  };
}

/** The day's event for the brief, or null when it cannot be picked now. */
async function briefMoment(env: Env, dateStr: string): Promise<MomentBeforeData | null> {
  if (!(await env.CACHE.get(momentCacheKey(dateStr))) && (await getAiBudgetBlock(env))) return null;
  const { events } = await getTodayEvents(env, dateStr);
  const moment = await getOrGenerateMoment(env, events, dateStr);
  return moment.title === "" ? null : moment;
}

/**
 * Jarvis's picture for a date and kind, cropped to the panel as 800x480 RGB, or null when none
 * was delivered or it cannot be read (the caller then uses Workers AI).
 */
export async function takeJarvisArt(env: Env, dateStr: string, kind: ArtKind): Promise<Uint8Array | null> {
  const obj = await env.PHOTOS.get(artKey(dateStr, kind));
  if (!obj) return null;
  try {
    return await jpegToRGB(env, new Uint8Array(await obj.arrayBuffer()));
  } catch (err) {
    console.error(`Jarvis art ${dateStr}/${kind} unreadable, using Workers AI:`, err);
    return null;
  }
}

/**
 * Store Jarvis's colour picture for a date (the birthday portrait on a family birthday, else the
 * moment) under the key the pages read, replacing any Workers AI picture. True when stored.
 */
export async function storeJarvisColor(
  env: Env, dateStr: string, birthday: BirthdayPerson | null, year: number, displayDate: string, moment: MomentBeforeData,
): Promise<boolean> {
  const kind: ArtKind = birthday ? "birthday" : "moment";
  const rgb = await takeJarvisArt(env, dateStr, kind);
  if (!rgb) return false;
  const imageB64 = pngToBase64(await encodePNGRGB(rgb, WIDTH, HEIGHT));
  if (birthday) {
    const style = getBirthdayStyle(birthday, year);
    await saveColorEntry(env, dateStr, birthday, {
      imageB64, displayDate, source: "jarvis",
      moment: { year, location: "", title: birthday.name, scene: "", imagePrompt: "" },
      birthdayInfo: { name: birthday.name, age: year - birthday.birthYear, styleName: style.name },
    });
  } else {
    const chosen = await jarvisMoment(env, dateStr);
    await saveColorEntry(env, dateStr, null, { imageB64, moment: chosen ?? moment, displayDate, source: "jarvis" });
  }
  await env.PHOTOS.delete(artKey(dateStr, kind));
  await markJarvisDrew(env, dateStr, kind);
  return true;
}

/** Record that a day's picture came from Jarvis, for /health-detailed and the alert check (#90). */
async function markJarvisDrew(env: Env, dateStr: string, kind: ArtKind): Promise<void> {
  await env.PHOTOS.put(`art/used/${dateStr}/${kind}`, new Date().toISOString());
}

/** Did Jarvis draw this picture for the date? (Written when the warm stores his delivery.) */
export async function jarvisDrew(env: Env, dateStr: string, kind: ArtKind): Promise<boolean> {
  return (await env.PHOTOS.head(`art/used/${dateStr}/${kind}`)) !== null;
}

/**
 * The event Jarvis chose and drew when the brief left the pick to him (`art/DATE/moment.json`,
 * written by the watcher from his delivery note), or null. It also becomes the day's shared
 * moment when the Worker has none of its own, so the other pages agree.
 */
async function jarvisMoment(env: Env, dateStr: string): Promise<MomentBeforeData | null> {
  const obj = await env.PHOTOS.get(`${artKey(dateStr, "moment")}.json`);
  if (!obj) return null;
  await env.PHOTOS.delete(`${artKey(dateStr, "moment")}.json`);
  let parsed: { title?: unknown; year?: unknown; place?: unknown };
  try { parsed = await obj.json(); } catch { return null; }
  const title = typeof parsed.title === "string" ? parsed.title.slice(0, 80) : "";
  const year = Number(parsed.year);
  if (!title || !Number.isInteger(year)) return null;
  const moment: MomentBeforeData = {
    title, year, location: typeof parsed.place === "string" ? parsed.place.slice(0, 80) : "", scene: "", imagePrompt: "",
  };
  const cached = await env.CACHE.get<{ data: MomentBeforeData }>(momentCacheKey(dateStr), "json");
  if (!cached?.data?.title) {
    await env.CACHE.put(momentCacheKey(dateStr), JSON.stringify({ data: moment, timestamp: Date.now() }), { expirationTtl: 604800 });
  }
  return moment;
}

/** Store Jarvis's skyline for a date under the daily skyline key, caption burned in. True when stored. */
export async function storeJarvisSkyline(env: Env, dateStr: string): Promise<boolean> {
  const rgb = await takeJarvisArt(env, dateStr, "skyline");
  if (!rgb) return false;
  const parts = parseDateParts(dateStr);
  const city = pickSkylineCity(parts, { mode: "daily", rotateMin: DEFAULT_ROTATE_MIN, bucket: 0 });
  const png = await finishSkylinePNG(rgb, formatSkylineCaption(city, parts.displayDate));
  await env.CACHE.put(skylineCacheKey(dateStr, "daily", DEFAULT_ROTATE_MIN, 0), pngToBase64(png), { expirationTtl: 604800 });
  await env.PHOTOS.delete(artKey(dateStr, "skyline"));
  await markJarvisDrew(env, dateStr, "skyline");
  return true;
}
