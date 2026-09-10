/**
 * Skyline HTML pages for reTerminal E1002 (Spectra 6) and E1001 (mono).
 *
 * /skyline       — live skyline, PNG inlined as a data URI
 * /skyline-bw    — BW-only skyline for the mono panel, PNG inlined as a data URI
 * /skyline-test  — test with date/city/style overrides (forwards ALL query params
 *                  to /skyline-test.png through an <img src>; not a panel page)
 *
 * The live pages inline the image so the panel needs exactly ONE request.
 * SenseCraft's renderer screenshots about a second after the document arrives
 * and cancels any sub-resource still in flight; a separate <img src="/skyline.png">
 * was cancelled on the E1002 and the panel went fully white (DECISIONS #65).
 * /color/moment has always inlined its PNG and never blanked.
 *
 * HTML wrappers are always no-store so SenseCraft re-fetches on each screenshot.
 */

import type { Env } from "../types";
import { spectra6CSS } from "../spectra6";
import { htmlResponse } from "../response";
import { getChicagoDateParts } from "../date-utils";
import { imageUnavailable } from "../stale-cache";
import { pngToBase64 } from "../png";
import { unavailableHTML } from "./unavailable";

const SKYLINE_UNAVAILABLE_NOTE =
  "Today's skyline could not be generated. The daily image budget resets at midnight UTC and the view returns on its own.";

/** Produces the skyline PNG response — in production this is `handleSkylinePng`. */
export type SkylinePngFetcher = () => Promise<Response>;

/**
 * A skyline page must not try to show an image that is already known to fail.
 * `bwOnly` matters because the colour route can cross-fall-back to a B&W cache,
 * but the B&W route cannot use a colour one.
 */
async function skylineUnavailable(env: Env, bwOnly: boolean): Promise<boolean> {
  const { dateStr } = getChicagoDateParts();
  return imageUnavailable(env, "skyline:", dateStr,
    bwOnly ? { accept: (n) => n.endsWith(":bw") } : {});
}

function unavailablePage(): Response {
  return htmlResponse(unavailableHTML("World Skyline Series", SKYLINE_UNAVAILABLE_NOTE), "no-store");
}

const PAGE_CSS = `
  * { margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; overflow: hidden; background: #fff; }
  img {
    display: block; width: 800px; height: 480px; object-fit: cover;
    color: #000; font: 700 26px Helvetica, Arial, sans-serif; text-align: center;
  }`;

/**
 * Build the panel page with the PNG inlined as a data URI. `spectra` adds the
 * Spectra 6 palette variables for the colour panel; the mono page has no use
 * for them.
 */
export function renderSkylineInlineHTML(base64: string, spectra: boolean): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>World Skyline Series</title>
<style>${spectra ? `\n  :root { ${spectra6CSS()} }` : ""}${PAGE_CSS}
</style>
</head>
<body>
  <img src="data:image/png;base64,${base64}" width="800" height="480" alt="World Skyline Series">
</body>
</html>`;
}

/** Diagnostic headers the png handler sets; forwarded so `curl -I /skyline` still tells the story. */
const FORWARDED_HEADER_PREFIXES = ["X-Skyline-", "X-Stale-While-Generating"];

/**
 * Serve a live skyline page: run the png handler, inline its bytes. Anything
 * other than a PNG (503 from an exhausted budget with no fallback) becomes the
 * text page, so the panel never shows a broken-image glyph (DECISIONS #58).
 */
async function skylineInlinePage(env: Env, bwOnly: boolean, fetchPng: SkylinePngFetcher): Promise<Response> {
  if (await skylineUnavailable(env, bwOnly)) return unavailablePage();

  const png = await fetchPng();
  const contentType = png.headers.get("Content-Type") ?? "";
  if (!png.ok || !contentType.startsWith("image/png")) return unavailablePage();

  const base64 = pngToBase64(new Uint8Array(await png.arrayBuffer()));
  const res = htmlResponse(renderSkylineInlineHTML(base64, !bwOnly), "no-store");
  png.headers.forEach((value, name) => {
    if (FORWARDED_HEADER_PREFIXES.some((p) => name.toLowerCase().startsWith(p.toLowerCase()))) {
      res.headers.set(name, value);
    }
  });
  return res;
}

/** Serve /skyline (E1002 colour). `fetchPng` must honour the page's query string. */
export function skylinePageResponse(env: Env, fetchPng: SkylinePngFetcher): Promise<Response> {
  return skylineInlinePage(env, false, fetchPng);
}

/** Serve /skyline-bw (E1001 mono). `fetchPng` must request the bw=1 variant. */
export function skylineBwPageResponse(env: Env, fetchPng: SkylinePngFetcher): Promise<Response> {
  return skylineInlinePage(env, true, fetchPng);
}

/**
 * Serve /skyline-test — forwards ALL query params to /skyline-test.png. This is
 * a browser test page, not a panel page, so the <img src> form is fine here.
 */
export function skylineTestPageResponse(queryString: string): Response {
  const src = queryString ? `/skyline-test.png?${queryString}` : "/skyline-test.png";
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>World Skyline Series</title>
<style>
  :root { ${spectra6CSS()} }${PAGE_CSS}
</style>
</head>
<body>
  <img src="${src}" width="800" height="480" alt="World Skyline Series - image unavailable">
</body>
</html>`;
  return htmlResponse(html, "no-store");
}
