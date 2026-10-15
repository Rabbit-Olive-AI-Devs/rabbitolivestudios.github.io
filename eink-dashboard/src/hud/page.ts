/**
 * Turn a drawn canvas into what the panel loads: one HTML document with the
 * PNG inlined as a data URI: 1-bit, or full-colour RGB when a picture on it has colour. SenseCraft screenshots about a second after
 * the document arrives and cancels any sub-resource still loading, so a panel
 * page must be a single request (DECISIONS #65).
 */

import type { Canvas } from "./canvas";
import { COLOR, colorAt } from "./canvas";
import { encodePNG1Bit, encodePNGRGB, pngToBase64 } from "../png";
import { htmlResponse } from "../response";
import { unavailableHTML } from "../pages/unavailable";

export function canvasToPng(c: Canvas): Promise<Uint8Array> {
  if (!c.px.includes(COLOR)) return encodePNG1Bit(c.px, c.w, c.h);
  const rgb = new Uint8Array(c.w * c.h * 3);
  for (let i = 0; i < c.px.length; i++) rgb.set(colorAt(c, i), i * 3);
  return encodePNGRGB(rgb, c.w, c.h);
}

export function hudPageHTML(title: string, base64: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>${title}</title>
<style>
  * { margin: 0; padding: 0; }
  html, body { width: 800px; height: 480px; overflow: hidden; background: #fff; }
  img { display: block; width: 800px; height: 480px; image-rendering: pixelated; }
</style>
</head>
<body>
  <img src="data:image/png;base64,${base64}" width="800" height="480" alt="${title}">
</body>
</html>`;
}

/**
 * Draw, encode and wrap. `title` must be a fixed string from our own code, never
 * request input: it is placed in the HTML unescaped. If drawing or encoding
 * fails, the panel gets the plain text page rather than a 500 (DECISIONS #58).
 */
export async function hudPageResponse(title: string, draw: () => Canvas): Promise<Response> {
  try {
    const png = await canvasToPng(draw());
    return htmlResponse(hudPageHTML(title, pngToBase64(png)), "no-store");
  } catch (err) {
    console.error(`${title}: draw failed:`, err);
    return htmlResponse(
      unavailableHTML(title, "This page could not be drawn. It will try again at the next refresh."),
      "no-store",
    );
  }
}
