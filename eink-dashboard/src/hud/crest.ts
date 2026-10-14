/**
 * Club crests for the desk panel: a small colour PNG turned into a pure 1-bit
 * bitmap once, when it is fetched, so the page only copies pixels afterwards.
 * Colours become ordered-dither dot patterns, which stay crisp on the mono
 * panel where a grey would fog (DECISIONS #48).
 */

import { decodePNG } from "../png-decode";
import type { Canvas } from "./canvas";
import { setPixel } from "./canvas";

/** Crests are square, this many pixels a side. */
export const CREST_SIZE = 96;
const BYTES = (CREST_SIZE * CREST_SIZE) / 8;
/** Stretch tones about mid-grey so a dark crest keeps its shapes apart. */
const CONTRAST = 1.25;
const BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

/**
 * A crest PNG as packed bits (1 = ink, row-major, most significant bit first),
 * base64-encoded. Transparent areas become paper. Null when the image is not a
 * usable 96x96 PNG.
 */
export async function crestBits(png: Uint8Array): Promise<string | null> {
  try {
    const img = await decodePNG(png);
    if (img.width !== CREST_SIZE || img.height !== CREST_SIZE) return null;
    const bits = new Uint8Array(BYTES);
    let ink = 0;
    for (let y = 0; y < CREST_SIZE; y++) {
      for (let x = 0; x < CREST_SIZE; x++) {
        const i = y * CREST_SIZE + x;
        const a = img.alpha ? img.alpha[i] / 255 : 1;
        const tone = (img.gray[i] * a + 255 * (1 - a) - 128) * CONTRAST + 128;
        if ((tone / 255) * 16 > BAYER[y & 3][x & 3] + 0.5) continue;
        bits[i >> 3] |= 0x80 >> (i & 7);
        ink++;
      }
    }
    return ink > 0 ? btoa(String.fromCharCode(...bits)) : null;
  } catch {
    return null;
  }
}

/** Draw a crest from `crestBits` with its top-left corner at (x, y). Returns false if it is not one. */
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
  for (let i = 0; i < CREST_SIZE * CREST_SIZE; i++) {
    if (bin.charCodeAt(i >> 3) & (0x80 >> (i & 7))) setPixel(c, ox + (i % CREST_SIZE), oy + Math.floor(i / CREST_SIZE));
  }
  return true;
}
