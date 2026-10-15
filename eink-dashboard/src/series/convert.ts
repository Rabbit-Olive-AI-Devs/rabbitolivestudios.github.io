/**
 * The series' own conversions (round 4 of the mockups). They call shared helpers
 * but change no other pipeline:
 * - desk E1001: Floyd-Steinberg 1-bit after a mild contrast boost (Bayer muddied ink detail);
 * - office E1002: Spectra 6 Floyd-Steinberg where low-saturation pixels may only use
 *   black and white (Pipeline D's method turned greys into coloured speckle).
 */

import { SPECTRA6_PALETTE } from "../spectra6";

const CONTRAST = 1.25;
/** Below this max-min channel spread a pixel counts as grey. */
const GREY_SPREAD = 60;

export function luminance(rgb: Uint8Array): Uint8Array {
  const g = new Uint8Array(rgb.length / 3);
  for (let i = 0; i < g.length; i++) g[i] = Math.round(0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]);
  return g;
}

export function toDeskBits(gray: Uint8Array, w: number, h: number): Uint8Array {
  const f = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) f[i] = Math.min(255, Math.max(0, (gray[i] - 128) * CONTRAST + 128));
  const out = new Uint8Array(gray.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = f[i], n = v < 128 ? 0 : 255, e = v - n;
      out[i] = n ? 1 : 0;
      if (x + 1 < w) f[i + 1] += (e * 7) / 16;
      if (y + 1 < h) {
        if (x > 0) f[i + w - 1] += (e * 3) / 16;
        f[i + w] += (e * 5) / 16;
        if (x + 1 < w) f[i + w + 1] += e / 16;
      }
    }
  }
  return out;
}

export function toOfficeIndices(rgb: Uint8Array, w: number, h: number): Uint8Array {
  const P = SPECTRA6_PALETTE;
  const f = new Float32Array(rgb.length);
  for (let i = 0; i < rgb.length; i++) f[i] = rgb[i];
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const spread = Math.max(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]) - Math.min(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
      const last = spread < GREY_SPREAD ? 1 : P.length - 1;
      let best = 0, bd = Infinity;
      for (let k = 0; k <= last; k++) {
        const d = (f[i * 3] - P[k][0]) ** 2 + (f[i * 3 + 1] - P[k][1]) ** 2 + (f[i * 3 + 2] - P[k][2]) ** 2;
        if (d < bd) { bd = d; best = k; }
      }
      out[i] = best;
      // A grey pixel passes on only its brightness error: its tint was never printable in
      // black and white, and carried forward it surfaces as colour speckle somewhere else.
      const lumErr = last === 1
        ? 0.299 * (f[i * 3] - P[best][0]) + 0.587 * (f[i * 3 + 1] - P[best][1]) + 0.114 * (f[i * 3 + 2] - P[best][2])
        : 0;
      for (let c = 0; c < 3; c++) {
        const e = last === 1 ? lumErr : f[i * 3 + c] - P[best][c];
        if (x + 1 < w) f[(i + 1) * 3 + c] += (e * 7) / 16;
        if (y + 1 < h) {
          if (x > 0) f[(i + w - 1) * 3 + c] += (e * 3) / 16;
          f[(i + w) * 3 + c] += (e * 5) / 16;
          if (x + 1 < w) f[(i + w + 1) * 3 + c] += e / 16;
        }
      }
    }
  }
  return out;
}
