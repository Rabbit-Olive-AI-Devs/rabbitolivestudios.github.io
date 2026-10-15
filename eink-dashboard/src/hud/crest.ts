/**
 * Club crests for the desk panel: a small colour PNG turned into 96x96 grey
 * tones once, when it is fetched, so the page only copies pixels afterwards.
 * The tones go to the panel untouched and the panel converts them, as every
 * picture does since DECISIONS #81 (until then this was our own ordered dither).
 */

import { decodePNG } from "../png-decode";
import type { Canvas } from "./canvas";
import { setTone } from "./canvas";

/** Crests are square, this many pixels a side. */
export const CREST_SIZE = 96;
const BYTES = CREST_SIZE * CREST_SIZE;

/**
 * A crest PNG as one grey byte per pixel (0 black to 255 white, row-major),
 * base64-encoded. Transparent areas become paper. Null when the image is not a
 * usable 96x96 PNG or has no ink at all.
 */
export async function crestTones(png: Uint8Array): Promise<string | null> {
  try {
    const img = await decodePNG(png);
    if (img.width !== CREST_SIZE || img.height !== CREST_SIZE) return null;
    const tones = new Uint8Array(BYTES);
    let ink = 0;
    for (let i = 0; i < BYTES; i++) {
      const a = img.alpha ? img.alpha[i] / 255 : 1;
      tones[i] = Math.round(img.gray[i] * a + 255 * (1 - a));
      if (tones[i] < 255) ink++;
    }
    let bin = "";
    for (let i = 0; i < BYTES; i += 8192) bin += String.fromCharCode(...tones.subarray(i, i + 8192));
    return ink > 0 ? btoa(bin) : null;
  } catch {
    return null;
  }
}

/** Draw a crest from `crestTones` with its top-left corner at (x, y). Returns false if it is not one. */
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
  for (let i = 0; i < BYTES; i++) {
    const g = bin.charCodeAt(i);
    if (g < 255) setTone(c, ox + (i % CREST_SIZE), oy + Math.floor(i / CREST_SIZE), g);
  }
  return true;
}
