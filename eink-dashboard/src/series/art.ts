/**
 * A writer's picture → a panel: flatten transparency onto paper, crop to fill
 * around a focus point, and shrink by area averaging. Runs offline at import.
 */

export interface RGBImage { w: number; h: number; rgb: Uint8Array }

export function flattenAlpha(rgb: Uint8Array, alpha: Uint8Array | undefined): Uint8Array {
  if (!alpha) return rgb;
  const out = new Uint8Array(rgb.length);
  for (let i = 0; i < alpha.length; i++) {
    const a = alpha[i] / 255;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = Math.round(rgb[i * 3 + c] * a + 255 * (1 - a));
  }
  return out;
}

/**
 * Scale `img` so it covers w×h, then crop around `focus` (default: the centre).
 * `map` takes a point in the original picture's pixels to the panel's pixels.
 * The picture must be at least as large as the panel in both directions.
 */
export function cropToFill(
  img: RGBImage, w: number, h: number, focus?: [number, number],
): { image: RGBImage; map: (p: [number, number]) => [number, number] } {
  const scale = Math.max(w / img.w, h / img.h);
  const cw = w / scale, ch = h / scale;
  const [fx, fy] = focus ?? [img.w / 2, img.h / 2];
  const sx = Math.min(Math.max(fx - cw / 2, 0), img.w - cw);
  const sy = Math.min(Math.max(fy - ch / 2, 0), img.h - ch);
  const rgb = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor(sy + y / scale), y1 = Math.max(y0 + 1, Math.floor(sy + (y + 1) / scale));
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor(sx + x / scale), x1 = Math.max(x0 + 1, Math.floor(sx + (x + 1) / scale));
      let r = 0, g = 0, b = 0, n = 0;
      for (let yy = y0; yy < Math.min(y1, img.h); yy++) {
        for (let xx = x0; xx < Math.min(x1, img.w); xx++) {
          const s = (yy * img.w + xx) * 3;
          r += img.rgb[s]; g += img.rgb[s + 1]; b += img.rgb[s + 2]; n++;
        }
      }
      const d = (y * w + x) * 3;
      rgb[d] = Math.round(r / n); rgb[d + 1] = Math.round(g / n); rgb[d + 2] = Math.round(b / n);
    }
  }
  const map = ([px, py]: [number, number]): [number, number] =>
    [Math.round((px - sx) * scale), Math.round((py - sy) * scale)];
  return { image: { w, h, rgb }, map };
}

/** True when the picture is nearly one flat tone (a failed or empty render). */
export function isNearlyBlank(img: RGBImage): boolean {
  const n = img.w * img.h;
  let sum = 0, sq = 0;
  for (let i = 0; i < n; i++) {
    const v = (img.rgb[i * 3] * 299 + img.rgb[i * 3 + 1] * 587 + img.rgb[i * 3 + 2] * 114) / 1000;
    sum += v; sq += v * v;
  }
  const mean = sum / n;
  return Math.sqrt(Math.max(0, sq / n - mean * mean)) < 6;
}
