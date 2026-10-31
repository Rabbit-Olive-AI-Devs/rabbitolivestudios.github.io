/**
 * /color/family — Family Memory (DECISIONS #92). One request: the PNG inlined as a data URI (#65).
 * Never generates: today's entry, else the most recent earlier one, else the e-ink text page.
 */

import type { Env } from "../types";
import { htmlResponse } from "../response";
import { getChicagoDateParts, shiftDateStr } from "../date-utils";
import { findMostRecentCached } from "../stale-cache";
import { familyCacheKey } from "../cache-keys";
import { FAMILY_TITLE, headerPlace, parseStoredFamily } from "../family";
import { unavailableHTML } from "./unavailable";

const UNAVAILABLE_NOTE = "No family memory yet. The Mac mini picks tomorrow's photo each evening.";

export function renderFamilyHTML(base64: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>${FAMILY_TITLE}</title>
<style>
  * { margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; overflow: hidden; background: #fff; }
  img {
    display: block; width: 800px; height: 480px; object-fit: cover;
    color: #000; font: 700 26px Helvetica, Arial, sans-serif; text-align: center;
  }
</style>
</head>
<body>
  <img src="data:image/png;base64,${base64}" width="800" height="480" alt="${FAMILY_TITLE}">
</body>
</html>`;
}

export async function handleFamilyPage(env: Env, dateStr: string = getChicagoDateParts().dateStr): Promise<Response> {
  const today = await env.CACHE.get(familyCacheKey(dateStr));
  const raw = today ?? (await findMostRecentCached(env, "family:", shiftDateStr(dateStr, -1)))?.value;
  if (!raw) return htmlResponse(unavailableHTML(FAMILY_TITLE, UNAVAILABLE_NOTE), "no-store");
  const stored = parseStoredFamily(raw);
  const caption = stored.caption;
  const headers = caption ? {
    "x-family-date": caption.date,
    "x-family-ago": String(caption.yearsAgo),
    "x-family-place": headerPlace(caption.place),
    "x-family-style": caption.style,
  } : undefined;
  return htmlResponse(renderFamilyHTML(stored.image), "no-store", headers);
}
