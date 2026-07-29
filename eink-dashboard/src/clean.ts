/**
 * Screen-cleaner endpoint logic.
 *
 * When an e-paper panel holds one static image for days/weeks (e.g. the device
 * loses network and freezes on its last frame), charged pigment particles
 * "stick" and leave a faint ghost (image retention). The fix is to drive the
 * panel through several full-screen full-refresh cycles so every pixel swings
 * across its full range and every pigment is exercised.
 *
 * The device (SenseCraft HMI) screenshots ONE fixed URL on an interval, so a
 * single page cannot animate — the endpoint must return a DIFFERENT solid color
 * on each fetch. We rotate by wall-clock time.
 *
 * Aliasing note: the device's refresh interval is unknown. If the number of
 * frames divided the interval evenly the device would land on the same color
 * every time. Using a PRIME-length sequence (7) makes a full collapse happen
 * only when the fetch interval is an exact multiple of 7s — otherwise every
 * frame is visited over successive fetches. See DECISIONS.md #56.
 */

import { SPECTRA6_PALETTE, SPECTRA6_NAMES } from "./spectra6";
import { encodePNGIndexed } from "./png";

/** E-ink panel dimensions (both E1001 and E1002 are 800x480). */
const WIDTH = 800;
const HEIGHT = 480;

/**
 * Cleaning sequence of Spectra-6 palette indices (length 7, prime).
 * Covers all 6 pigments; black appears twice for extra full-swing flushes,
 * which are the most effective at clearing retention.
 *   1=white, 0=black, 2=red, 3=yellow, 4=green, 5=blue, 0=black
 */
export const CLEAN_SEQUENCE: number[] = [1, 0, 2, 3, 4, 5, 0];

/**
 * Pick the palette index to show, rotating through CLEAN_SEQUENCE by time.
 * @param epochSeconds - current time in whole seconds (Date.now()/1000 | 0)
 * @param secondsPerFrame - how long each frame is held (>=1)
 */
export function pickCleanColorIndex(epochSeconds: number, secondsPerFrame: number): number {
  const spf = Math.max(1, Math.floor(secondsPerFrame) || 1);
  const step = Math.floor(epochSeconds / spf);
  const idx = ((step % CLEAN_SEQUENCE.length) + CLEAN_SEQUENCE.length) % CLEAN_SEQUENCE.length;
  return CLEAN_SEQUENCE[idx];
}

/**
 * Resolve a `?c=` override to a palette index, or null if absent/invalid.
 * Accepts a color name ("black"/"white"/"red"/"yellow"/"green"/"blue") or an
 * index 0-5.
 */
export function parseCleanColor(raw: string | null): number | null {
  if (!raw) return null;
  const s = raw.trim().toLowerCase();
  const byName = SPECTRA6_NAMES.indexOf(s as (typeof SPECTRA6_NAMES)[number]);
  if (byName >= 0) return byName;
  const n = Number(s);
  if (Number.isInteger(n) && n >= 0 && n < SPECTRA6_PALETTE.length) return n;
  return null;
}

/** Render a solid full-screen 800x480 PNG of the given Spectra-6 palette index. */
export async function renderCleanPNG(colorIndex: number): Promise<Uint8Array> {
  const indices = new Uint8Array(WIDTH * HEIGHT).fill(colorIndex);
  return encodePNGIndexed(indices, WIDTH, HEIGHT, SPECTRA6_PALETTE);
}
