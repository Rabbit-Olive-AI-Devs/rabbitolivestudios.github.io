import type { Env } from "../types";
import { htmlResponse } from "../response";
import { getChicagoDateParts } from "../date-utils";
import { imageUnavailable } from "../stale-cache";
import { unavailableHTML } from "./unavailable";

const IMAGE_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800, initial-scale=1, maximum-scale=1">
<title>Moment Before</title>
<style>
* { margin: 0; padding: 0; }
html, body { width: 100%; height: 100%; overflow: hidden; background: #fff; }
/* The font/colour rules only ever show if the image itself fails to load, in
   which case the browser paints the alt text in this box (DECISIONS #57). */
img {
  display: block; width: 100vw; height: 100vh; object-fit: cover;
  color: #000; font: 700 26px Helvetica, Arial, sans-serif; text-align: center;
}
</style>
</head>
<body>
<img src="/fact.png" width="800" height="480" alt="Moment Before - image unavailable">
</body>
</html>`;

export async function handleFactPage(env: Env): Promise<Response> {
  const { dateStr } = getChicagoDateParts();

  // Never point the panel at an image we already know cannot be produced — a
  // dead <img> renders as a broken-image glyph on the display. On a birthday the
  // picture comes from the birthday cache instead, so either family counts.
  const noFact = await imageUnavailable(env, "fact4:", dateStr);
  if (noFact && await imageUnavailable(env, "birthday:", dateStr)) {
    return htmlResponse(
      unavailableHTML(
        "Moment Before",
        "Today's illustration could not be generated. The daily image budget resets at midnight UTC and the picture returns on its own.",
      ),
      "no-store",
    );
  }

  return htmlResponse(IMAGE_HTML, "public, max-age=86400");
}
