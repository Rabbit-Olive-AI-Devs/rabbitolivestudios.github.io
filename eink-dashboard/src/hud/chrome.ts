/**
 * The frame every desk panel page shares: a slim top bar with a breadcrumb on
 * the left and a status group ending in an inverted pill on the right.
 */

import type { Canvas } from "./canvas";
import { fillRect, hline } from "./canvas";
import { drawType, measureType } from "./type";

/** YYYY-MM-DD as the owner reads dates: DD/MM/YYYY. */
export function displayDate(dateStr: string): string {
  const [y, m, d] = dateStr.split("-");
  return `${d}/${m}/${y}`;
}

/** Y of the rule under the top bar; page content starts below it. */
export const TOP_BAR_BOTTOM = 36;

/**
 * `trail` is the breadcrumb, last entry emphasised. `status` is short right-hand
 * text; `pill` is the one value worth an inverted block.
 */
export function topBar(c: Canvas, trail: readonly string[], status: readonly string[], pill: string): void {
  const baseline = 23;
  let x = 20;
  trail.forEach((part, i) => {
    const last = i === trail.length - 1;
    x += drawType(c, last ? "plex13s" : "plex13r", x, baseline, part) + 8;
    if (!last) x += drawType(c, "plex13r", x, baseline, "/") + 8;
  });

  const pillW = measureType("plex12s", pill) + 16;
  const pillX = c.w - 16 - pillW;
  fillRect(c, pillX, 9, pillW, 19);
  drawType(c, "plex12s", pillX + pillW / 2, baseline, pill, { align: "center", white: true });

  let right = pillX - 14;
  for (let i = status.length - 1; i >= 0; i--) {
    right -= drawType(c, "plex12m", right, baseline, status[i], { align: "right" }) + 16;
  }
  hline(c, 0, TOP_BAR_BOTTOM, c.w);
}
