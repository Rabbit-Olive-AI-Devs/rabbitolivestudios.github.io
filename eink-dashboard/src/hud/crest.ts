/**
 * Club crests for the desk panel: a small colour PNG turned into 96x96 RGB once,
 * when it is fetched, so the page only copies pixels afterwards. The crest goes to
 * the panel untouched in full colour and the panel converts it, as every picture
 * does since DECISIONS #81 (until then this was our own ordered dither to 1-bit).
 */

import { decodePNG } from "../png-decode";
import type { Canvas } from "./canvas";
import { setColor } from "./canvas";

/** Crests are square, this many pixels a side. */
export const CREST_SIZE = 96;
const PIXELS = CREST_SIZE * CREST_SIZE;
const BYTES = PIXELS * 3;

/**
 * A crest PNG as RGB, 3 bytes per pixel, row-major, base64-encoded. Transparent
 * areas become white paper. Null when the image is not a usable 96x96 PNG or has
 * no ink at all.
 */
export async function crestColors(png: Uint8Array): Promise<string | null> {
  try {
    const img = await decodePNG(png);
    if (img.width !== CREST_SIZE || img.height !== CREST_SIZE) return null;
    const out = new Uint8Array(BYTES);
    let ink = 0;
    for (let i = 0; i < PIXELS; i++) {
      const a = img.alpha ? img.alpha[i] / 255 : 1;
      for (let k = 0; k < 3; k++) {
        const v = img.rgb ? img.rgb[i * 3 + k] : img.gray[i];
        out[i * 3 + k] = Math.round(v * a + 255 * (1 - a));
        if (out[i * 3 + k] < 255) ink++;
      }
    }
    let bin = "";
    for (let i = 0; i < BYTES; i += 8192) bin += String.fromCharCode(...out.subarray(i, i + 8192));
    return ink > 0 ? btoa(bin) : null;
  } catch {
    return null;
  }
}

/** Draw a crest from `crestColors` with its top-left corner at (x, y). Returns false if it is not one. */
export function drawCrest(c: Canvas, crest: string | undefined, x: number, y: number): boolean {
  if (!crest) return false;
  let bin: string;
  try {
    bin = atob(crest);
  } catch {
    return false;
  }
  if (bin.length !== BYTES) return false;
  const ox = Math.round(x);
  const oy = Math.round(y);
  for (let i = 0; i < PIXELS; i++) {
    const [r, g, b] = [bin.charCodeAt(i * 3), bin.charCodeAt(i * 3 + 1), bin.charCodeAt(i * 3 + 2)];
    if (r < 255 || g < 255 || b < 255) setColor(c, ox + (i % CREST_SIZE), oy + Math.floor(i / CREST_SIZE), r, g, b);
  }
  return true;
}
