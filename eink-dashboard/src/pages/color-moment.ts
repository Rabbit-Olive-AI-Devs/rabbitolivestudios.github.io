/**
 * Color "Moment Before" page for reTerminal E1002 (Spectra 6).
 *
 * Generates an AI image and serves it untouched in full colour as HTML with an inline
 * base64 PNG; the panel converts it to its six colours itself (DECISIONS #81).
 *
 * Route: /color/moment
 * Test:  /color/test-moment?m=7&d=20
 */

import { fetchWithTimeout } from "../fetch-timeout";
import type { Env, MomentBeforeData } from "../types";
import { getChicagoDateParts, shiftDateStr } from "../date-utils";
import { findMostRecentCached } from "../stale-cache";
import { getTodayEvents } from "../fact";
import { getOrGenerateMoment, generateMomentBefore } from "../moment";
import { stylesFor, momentStyleFor } from "../art-styles";
import type { ArtStyle } from "../art-styles";
import { getBirthdayToday, getBirthdayByKey, getBirthdayStyle, resolveBirthdayStyle, BIRTHDAY_STYLES } from "../birthday";
import type { BirthdayPerson } from "../birthday";
import { callFluxPortrait, fetchReferencePhotos, ageDescription } from "../birthday-image";
import { generateAndDecodeColorFlux, generateAndDecodeColor, jpegToRGB } from "../image-color";
import { encodePNGRGB, pngToBase64 } from "../png";
import { spectra6CSS } from "../spectra6";
import { WIDTH, HEIGHT } from "../image";
import { escapeHTML } from "../escape";
import { htmlResponse } from "../response";
import { parseMonth, parseDay, parseStyleIdx } from "../validate";
import { colorBirthdayCacheKey, colorMomentCacheKey } from "../cache-keys";
import {
  assertAiBudgetAvailable,
  getAiBudgetBlock,
  isNeuronBudgetError,
  markAiBudgetExhausted,
  serveStaleWhileGenerating,
  withGenerationLock,
} from "../cache-guard";

const ANTI_TEXT_SUFFIX = "no text, no words, no letters, no writing, no signage, no captions, no watermark";

/** The moment's styles: every style in the shared library that suits a scene (DECISIONS #86). */
export const COLOR_MOMENT_STYLES = stylesFor("moment");

/** The colour moment style for a Chicago date: a random order with no repeat until all have had a day. */
export function getColorMomentStyle(dateStr: string): ArtStyle {
  return momentStyleFor(dateStr);
}

/** Generate the full-colour moment image, returns base64 PNG + styleId. */
export async function generateColorMoment(
  env: Env,
  moment: MomentBeforeData,
  dateStr: string,
  forceStyleId?: string,
): Promise<{ base64: string; styleId: string }> {
  const style = forceStyleId
    ? (COLOR_MOMENT_STYLES.find(s => s.id === forceStyleId) ?? getColorMomentStyle(dateStr))
    : getColorMomentStyle(dateStr);
  console.log(`Color moment style: ${style.name} (${style.id})`);
  const prompt = `${style.prompt}, ${moment.imagePrompt}, ${ANTI_TEXT_SUFFIX}`;

  let rgb: Uint8Array | null = null;

  // Single FLUX.2 attempt — no retry to conserve neuron budget (free tier)
  try {
    rgb = await generateAndDecodeColorFlux(env, prompt);
  } catch (err) {
    console.error("Color moment FLUX.2 failed:", err);
    if (isNeuronBudgetError(err)) throw err;
  }

  // Fallback to SDXL
  if (!rgb) {
    console.log("Color moment: FLUX.2 failed, falling back to SDXL");
    rgb = await generateAndDecodeColor(env, prompt);
  }

  // The picture untouched in full colour; the panel converts it itself (DECISIONS #81).
  const png = await encodePNGRGB(rgb, WIDTH, HEIGHT);
  return { base64: pngToBase64(png), styleId: style.id };
}

/** Generate a color birthday portrait, returns base64 PNG. */
async function generateColorBirthday(
  env: Env,
  person: BirthdayPerson,
  currentYear: number,
  styleOverride?: number,
): Promise<string> {
  const style = resolveBirthdayStyle(person, currentYear, styleOverride);

  const photos = await fetchReferencePhotos(env, person.key);
  console.log(`Color birthday: found ${photos.length} reference photo(s) for ${person.key}`);

  // Single FLUX.2 attempt — no retry to conserve neuron budget (free tier)
  let rgb: Uint8Array | null = null;
  try {
    const jpegBytes = await callFluxPortrait(env, person, style.prompt, photos, currentYear);
    rgb = await jpegToRGB(env, jpegBytes);
  } catch (err) {
    console.error("Color birthday FLUX.2 failed:", err);
    if (isNeuronBudgetError(err)) throw err;
  }

  // Fallback: SDXL with text-only prompt (no reference photos — API limitation)
  if (!rgb) {
    console.log("Color birthday: FLUX.2 failed, falling back to SDXL");
    const ageLine = ageDescription(person, currentYear);
    const prompt = `artistic portrait of ${ageLine}, ${style.prompt}, head and shoulders, centered composition, looking at viewer, smiling, ${ANTI_TEXT_SUFFIX}`;
    rgb = await generateAndDecodeColor(env, prompt);
  }

  // The picture untouched in full colour; the panel converts it itself (DECISIONS #81).
  const png = await encodePNGRGB(rgb, WIDTH, HEIGHT);
  return pngToBase64(png);
}

export interface BirthdayCaptionInfo {
  name: string;
  age: number;
  styleName: string;
}

function renderHTML(
  imageB64: string,
  moment: MomentBeforeData,
  displayDate: string,
  birthdayInfo?: BirthdayCaptionInfo,
): string {
  let captionHTML: string;
  if (birthdayInfo) {
    // Match E1001: "Happy Birthday!" left | "Name - age years" center | style right
    captionHTML = `<span class="cap-left">Happy Birthday!</span><span class="cap-center">${escapeHTML(birthdayInfo.name)} - ${birthdayInfo.age} years</span><span class="cap-right">${escapeHTML(birthdayInfo.styleName)}</span>`;
  } else {
    const location = moment.location.length > 35
      ? moment.location.slice(0, 32) + "..."
      : moment.location;
    const title = moment.title || moment.scene.slice(0, 40);
    const dateLine = `${displayDate}, ${moment.year}`;
    captionHTML = `<span class="cap-left">${escapeHTML(location)}</span><span class="cap-center">${escapeHTML(title)}</span><span class="cap-right">${dateLine}</span>`;
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>Moment Before</title>
<style>
  :root { ${spectra6CSS()} }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    width: 800px; height: 480px; overflow: hidden;
    background: #fff; color: #000;
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  }
  .image-container {
    width: 800px; height: 456px;
    overflow: hidden;
  }
  .image-container img {
    width: 800px; height: 456px;
    object-fit: cover;
    display: block;
  }
  .caption {
    width: 800px; height: 24px;
    background: #000; color: #fff;
    font-size: 14px; font-weight: 600;
    display: flex; align-items: center;
    padding: 0 12px;
    overflow: hidden;
    white-space: nowrap;
  }
  .cap-left { flex-shrink: 0; }
  .cap-center {
    flex: 1; text-align: center;
    overflow: hidden; text-overflow: ellipsis;
  }
  .cap-right { flex-shrink: 0; }
</style>
</head>
<body>
  <div class="image-container">
    <img src="data:image/png;base64,${imageB64}" alt="Moment Before">
  </div>
  <div class="caption">${captionHTML}</div>
</body>
</html>`;
}

/**
 * A background generation is nobody's critical path, so it can wait out a lock
 * held by the cron or another poll instead of overwriting it after the default
 * 18s and duplicating the work (DECISIONS #64).
 */
const BACKGROUND_LOCK_WAIT_MS = 60_000;

/**
 * Generate the day's colour picture — the birthday portrait on a family birthday (falling
 * back to the moment if it fails), the moment otherwise — and cache it under its key.
 * Shared by the page and the daily cron, which warms it first (DECISIONS #84).
 */
export async function buildColorMomentEntry(
  env: Env,
  dateStr: string,
  birthday: BirthdayPerson | null,
  currentYear: number,
  displayDate: string,
): Promise<{ imageB64: string; moment: MomentBeforeData; birthdayInfo?: BirthdayCaptionInfo }> {
  let imageB64: string;
  let moment: MomentBeforeData;
  let birthdayInfo: BirthdayCaptionInfo | undefined;

  if (birthday) {
    const age = currentYear - birthday.birthYear;
    const style = getBirthdayStyle(birthday, currentYear);
    moment = { year: currentYear, location: "", title: birthday.name, scene: "", imagePrompt: "" };
    try {
      imageB64 = await generateColorBirthday(env, birthday, currentYear);
      birthdayInfo = { name: birthday.name, age, styleName: style.name };
    } catch (err) {
      if (isNeuronBudgetError(err)) throw err;
      console.error("Color birthday failed, falling back to moment:", err);
      const { events } = await getTodayEvents(env, dateStr);
      moment = await getOrGenerateMoment(env, events, dateStr);
      const result = await generateColorMoment(env, moment, dateStr);
      imageB64 = result.base64;
    }
  } else {
    const { events } = await getTodayEvents(env, dateStr);
    moment = await getOrGenerateMoment(env, events, dateStr);
    const result = await generateColorMoment(env, moment, dateStr);
    imageB64 = result.base64;
  }

  await saveColorEntry(env, dateStr, birthday, { imageB64, moment, displayDate, birthdayInfo });
  return { imageB64, moment, birthdayInfo };
}

/**
 * Store a day's colour picture under its key: the birthday key on a family birthday, else the
 * moment key for the day's style. Used for Workers AI pictures and Jarvis's alike (#89);
 * `source` records which made it.
 */
export async function saveColorEntry(
  env: Env,
  dateStr: string,
  birthday: BirthdayPerson | null,
  entry: { imageB64: string; moment: MomentBeforeData; displayDate: string; birthdayInfo?: BirthdayCaptionInfo; source?: string },
): Promise<void> {
  const cacheKey = birthday
    ? colorBirthdayCacheKey(dateStr)
    : colorMomentCacheKey(dateStr, getColorMomentStyle(dateStr).id);
  await env.CACHE.put(cacheKey, JSON.stringify(entry), { expirationTtl: 604800 });
}

/**
 * /color/moment page. On a cold key the most recent cached moment is served
 * immediately and today's is generated in the background, so the panel never
 * waits on FLUX (DECISIONS #64). The blocking path only runs when nothing
 * older exists.
 */
export async function handleColorMomentPage(env: Env, url: URL, ctx?: ExecutionContext): Promise<Response> {
  const { year, month, day, dateStr } = getChicagoDateParts();
  const monthNum = parseInt(month);
  const dayNum = parseInt(day);

  // Check KV cache
  const birthday = getBirthdayToday(monthNum, dayNum);
  const colorStyle = getColorMomentStyle(dateStr);
  const cacheKey = birthday ? colorBirthdayCacheKey(dateStr) : colorMomentCacheKey(dateStr, colorStyle.id);

  const readCache = async (): Promise<Response | null> => {
    const cached = await env.CACHE.get(cacheKey);
    if (!cached) return null;
    console.log(`color/moment: cache hit (${cacheKey})`);
    try {
      const data = JSON.parse(cached);
      const html = renderHTML(data.imageB64, data.moment, data.displayDate, data.birthdayInfo);
      return htmlResponse(html, "public, max-age=86400");
    } catch {
      return null;
    }
  };

  const cached = await readCache();
  if (cached) return cached;

  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const displayDate = `${months[monthNum - 1]} ${dayNum}`;
  const currentYear = parseInt(year);

  const generateFresh = async (background: boolean): Promise<Response> => {
    try {
      await assertAiBudgetAvailable(env);
      return await withGenerationLock(env, cacheKey, readCache, async () => {
        const { imageB64, moment, birthdayInfo } = await buildColorMomentEntry(env, dateStr, birthday, currentYear, displayDate);
        const html = renderHTML(imageB64, moment, displayDate, birthdayInfo);
        return htmlResponse(html, "public, max-age=86400");
      }, background ? { waitMs: BACKGROUND_LOCK_WAIT_MS } : {});
    } catch (err) {
      await markAiBudgetExhausted(env, "color/moment", err);
      console.error("Color moment page error:", err);

      // Serve the most recent cached moment rather than a 503. The device renders a
      // failed fetch as an error page, so a day-old illustration is strictly better
      // than a broken panel — and entries live 7 days, so one is usually there
      // (DECISIONS #57).
      const stale = await findRecentColorMoment(env, dateStr);
      if (stale) return stale;

      return new Response("Color moment temporarily unavailable", {
        status: 503,
        headers: { "Content-Type": "text/plain; charset=utf-8", "Retry-After": "300" },
      });
    }
  };

  if (!(await getAiBudgetBlock(env))) {
    const swr = await serveStaleWhileGenerating(
      ctx,
      "color/moment",
      () => findRecentColorMoment(env, dateStr),
      () => generateFresh(true),
    );
    if (swr) return swr;
  }
  return generateFresh(false);
}

/**
 * Serve the most recent previously cached colour moment.
 *
 * Resolved by PREFIX rather than by rebuilding each day's key from its style,
 * so the fallback survives a COLOR_MOMENT_CACHE_VERSION bump — rebuilding embeds
 * the *current* version and would search for keys that never existed
 * (DECISIONS #57).
 */
async function findRecentColorMoment(
  env: Env, dateStr: string,
): Promise<Response | null> {
  const hit = await findMostRecentCached(env, "color-moment:", shiftDateStr(dateStr, -1));
  if (!hit) return null;
  try {
    const data = JSON.parse(hit.value);
    console.log(`color/moment: serving stale fallback from ${hit.key}`);
    const html = renderHTML(data.imageB64, data.moment, data.displayDate, data.birthdayInfo);
    // no-store so the stale copy never outlives the outage in a downstream cache.
    return htmlResponse(html, "no-store");
  } catch {
    return null; // corrupted entry
  }
}

/** Test endpoint for color moment with custom date. */
export async function handleColorTestMoment(env: Env, url: URL): Promise<Response> {
  try {
    await assertAiBudgetAvailable(env);
  } catch (err) {
    return new Response(String((err as any)?.message ?? err), {
      status: 503,
      headers: { "Content-Type": "text/plain; charset=utf-8", "Retry-After": "300" },
    });
  }

  const m = parseMonth(url.searchParams.get("m") ?? "7");
  const d = parseDay(url.searchParams.get("d") ?? "20");
  const forceStyle = url.searchParams.get("style") ?? undefined;
  const wikiUrl = `https://en.wikipedia.org/api/rest_v1/feed/onthisday/events/${m}/${d}`;
  const wikiRes = await fetchWithTimeout(wikiUrl, {
    headers: { "User-Agent": "eink-dashboard/1.0 (Cloudflare Worker)" },
  });
  const wikiData: any = await wikiRes.json();
  const testEvents = (wikiData.events ?? [])
    .filter((e: any) => e.year && e.text)
    .map((e: any) => ({ year: e.year as number, text: e.text as string }));

  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const displayDate = `${months[m - 1]} ${d}`;
  const testDateStr = `2026-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}`;

  let result: Awaited<ReturnType<typeof generateColorMoment>>;
  const moment = await generateMomentBefore(env, testEvents);
  try {
    result = await generateColorMoment(env, moment, testDateStr, forceStyle);
  } catch (err) {
    await markAiBudgetExhausted(env, "color/test-moment", err);
    console.error("Color test moment error:", err);
    return new Response("Failed to generate color moment image", { status: 503 });
  }

  const html = renderHTML(result.base64, moment, displayDate);
  return htmlResponse(html, "no-store");
}

/** Test endpoint for color birthday portrait. */
export async function handleColorTestBirthday(env: Env, url: URL): Promise<Response> {
  const nameParam = url.searchParams.get("name") ?? "thiago";
  const person = getBirthdayByKey(nameParam);
  if (!person) {
    const safeName = nameParam.slice(0, 50).replace(/[^\w-]/g, "");
    return new Response(`Unknown person key: ${safeName}`, {
      status: 400,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }
  const styleParam = url.searchParams.get("style");
  const styleIdx = parseStyleIdx(styleParam, BIRTHDAY_STYLES.length);
  const currentYear = new Date().getFullYear();

  try {
    await assertAiBudgetAvailable(env);
    const style = resolveBirthdayStyle(person, currentYear, styleIdx);
    const age = currentYear - person.birthYear;
    const imageB64 = await generateColorBirthday(env, person, currentYear, styleIdx);
    const moment: MomentBeforeData = { year: currentYear, location: "", title: person.name, scene: "", imagePrompt: "" };
    const birthdayInfo: BirthdayCaptionInfo = { name: person.name, age, styleName: style.name };
    const html = renderHTML(imageB64, moment, "Birthday", birthdayInfo);
    return htmlResponse(html, "no-store");
  } catch (err) {
    await markAiBudgetExhausted(env, "color/test-birthday", err);
    console.error("Color test birthday error:", err);
    return new Response("Failed to generate color birthday image", { status: 503 });
  }
}
