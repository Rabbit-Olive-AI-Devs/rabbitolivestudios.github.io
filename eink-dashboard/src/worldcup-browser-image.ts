/**
 * Nuclear option, done right (DECISIONS #48): use Cloudflare Browser Rendering to screenshot
 * the REAL Inter-font /worldcup HTML, then threshold it to pure 1-bit ourselves. SenseCraft's
 * cloud render then only sees pure black/white pixels — nothing left to dither/smudge — so the
 * panel shows the nice proportional typography, crisp. (A Worker has no browser of its own; the
 * BROWSER binding provides headless Chromium.)
 */

import puppeteer from "@cloudflare/puppeteer";
import { decodePNG } from "./png-decode";
import { encodePNGGray8 } from "./png";
import type { Env } from "./types";

/** Average 2x2 blocks: 1600x960 -> 800x480 (supersample for smoother edges pre-threshold). */
function downsample2x(g: Uint8Array, w: number): Uint8Array {
  const tw = w >> 1, th = (g.length / w) >> 1;
  const out = new Uint8Array(tw * th);
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const sx = x << 1, sy = y << 1;
      out[y * tw + x] = (g[sy * w + sx] + g[sy * w + sx + 1] + g[(sy + 1) * w + sx] + g[(sy + 1) * w + sx + 1]) >> 2;
    }
  }
  return out;
}

/**
 * Screenshot `pageUrl` at 2x via Browser Rendering and return a crisp 1-bit 800x480 grayscale PNG.
 * @param threshold gray cutoff (0-255); pixels darker than this become black. ~160 keeps stroke weight.
 */
export async function renderWorldCupBrowserPNG(env: Env, pageUrl: string, threshold = 160): Promise<Uint8Array> {
  if (!env.BROWSER) throw new Error("BROWSER binding not configured");
  const browser = await puppeteer.launch(env.BROWSER);
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 800, height: 480, deviceScaleFactor: 2 });
    await page.goto(pageUrl, { waitUntil: "networkidle0", timeout: 25000 });
    const shot = (await page.screenshot({
      type: "png",
      clip: { x: 0, y: 0, width: 800, height: 480 },
    })) as unknown as Uint8Array;

    const dec = await decodePNG(new Uint8Array(shot));
    let g = dec.gray;
    let w = dec.width;
    if (w === 1600) { g = downsample2x(g, w); w = 800; } // 2x -> native 800x480

    const out = new Uint8Array(800 * 480);
    for (let i = 0; i < out.length; i++) out[i] = (g[i] ?? 255) < threshold ? 0 : 255;
    return await encodePNGGray8(out, 800, 480);
  } finally {
    await browser.close();
  }
}
