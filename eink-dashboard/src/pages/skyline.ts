/**
 * Skyline HTML pages for reTerminal E1002 (Spectra 6) and E1001 (mono).
 *
 * /skyline       — live skyline (img src points to /skyline.png with forwarded params)
 * /skyline-test  — test with date/city/style overrides (forwards ALL query params)
 *
 * HTML wrappers are always no-store so SenseCraft re-fetches on each screenshot,
 * and the <img src> triggers a fresh (or bucket-cached) .png fetch each time.
 */

import type { Env } from "../types";
import { spectra6CSS } from "../spectra6";
import { htmlResponse } from "../response";
import { getChicagoDateParts } from "../date-utils";
import { imageUnavailable } from "../stale-cache";
import { unavailableHTML } from "./unavailable";

const SKYLINE_UNAVAILABLE_NOTE =
  "Today's skyline could not be generated. The daily image budget resets at midnight UTC and the view returns on its own.";

/**
 * A skyline page must not emit an <img> that is already known to fail — the
 * panel would show a broken-image glyph (this is exactly what the E1002
 * displayed on 2026-08-20). `bwOnly` matters because the colour route can
 * cross-fall-back to a B&W cache, but the B&W route cannot use a colour one.
 */
async function skylineUnavailable(env: Env, bwOnly: boolean): Promise<boolean> {
  const { dateStr } = getChicagoDateParts();
  return imageUnavailable(env, "skyline:", dateStr,
    bwOnly ? { accept: (n) => n.endsWith(":bw") } : {});
}

/**
 * Build a skyline HTML page that loads the image via <img src>.
 * The query string is forwarded so rotation params reach the .png endpoint.
 */
function renderSkylineHTML(pngPath: string, queryString: string): string {
  const src = queryString ? `${pngPath}?${queryString}` : pngPath;
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>World Skyline Series</title>
<style>
  :root { ${spectra6CSS()} }
  * { margin: 0; padding: 0; }
  html, body { width: 100%; height: 100%; overflow: hidden; background: #fff; }
  img {
    display: block; width: 800px; height: 480px; object-fit: cover;
    color: #000; font: 700 26px Helvetica, Arial, sans-serif; text-align: center;
  }
</style>
</head>
<body>
  <img src="${src}" width="800" height="480" alt="World Skyline Series - image unavailable">
</body>
</html>`;
}

/** Serve /skyline — forwards query string to /skyline.png */
export async function skylinePageResponse(env: Env, queryString: string): Promise<Response> {
  if (await skylineUnavailable(env, false)) {
    return htmlResponse(unavailableHTML("World Skyline Series", SKYLINE_UNAVAILABLE_NOTE), "no-store");
  }
  const html = renderSkylineHTML("/skyline.png", queryString);
  return htmlResponse(html, "no-store");
}

/** Serve /skyline-test — forwards ALL query params to /skyline-test.png */
export function skylineTestPageResponse(queryString: string): Response {
  const html = renderSkylineHTML("/skyline-test.png", queryString);
  return htmlResponse(html, "no-store");
}

/**
 * Serve /skyline-bw — BW-only skyline for E1001 mono display.
 * Always passes bw=1 to restrict to grayscale styles.
 * No Spectra6 CSS needed (mono pipeline).
 */
export async function skylineBwPageResponse(env: Env): Promise<Response> {
  if (await skylineUnavailable(env, true)) {
    return htmlResponse(unavailableHTML("World Skyline Series", SKYLINE_UNAVAILABLE_NOTE), "no-store");
  }
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>World Skyline Series</title>
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
  <img src="/skyline.png?bw=1" width="800" height="480" alt="World Skyline Series - image unavailable">
</body>
</html>`;
  return htmlResponse(html, "no-store");
}
