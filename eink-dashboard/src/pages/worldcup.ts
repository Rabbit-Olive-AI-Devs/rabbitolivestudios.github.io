/**
 * /worldcup — World Cup 2026 page for reTerminal E1001 (mono).
 *
 * Default response is a server-pre-dithered 1-bit PNG: Browser Rendering screenshots the
 * Inter-styled HTML (`?variant=src`), and we threshold it to pure black/white ourselves so
 * SenseCraft's cloud render has no gray edges to smudge → crisp on the panel (DECISIONS #48).
 * The image is cached with stale-while-revalidate + a cron warm so the device always gets an
 * instant response; a cold cache falls back to the Inter HTML (never blank).
 */

import type { Env, WcPhase } from "../types";
import { getWorldCupData } from "../worldcup";
import { renderWorldCupHTML, type WcTheme } from "../worldcup-ui";
import { testWorldCupData } from "../worldcup-testdata";
import { FONT_INTER_500, FONT_INTER_700 } from "../worldcup-fonts";
import { MONO_STYLE_BASE } from "../worldcup-styles";
import { renderWorldCupBrowserPNG } from "../worldcup-browser-image";
import { pngToBase64 } from "../png";

const WC_ORIGIN = "https://eink-dashboard.thiago-oliveira77.workers.dev";
export const IMG_KEY = "wc:image:v20"; // v20: advance winners into next-round slots before opponent decided (#55); v19: 2-line date on no-team boxes
const IMG_SOFT_TTL_MS = 14 * 60 * 1000; // refresh the rendered image at most ~every 14 min
const IMG_THRESHOLD = 160;              // gray cutoff for the 1-bit threshold

/**
 * B&W stylesheet = its own base (crisp weight-500 / no-clip / content-sized layout) + the inlined
 * Inter web font, appended last so the font-family wins. Fully separate from the color stylesheet
 * (worldcup-styles.ts) — changing one never touches the other. (DECISIONS #48)
 */
const MONO_STYLE = MONO_STYLE_BASE
  + `@font-face{font-family:'WCF';font-weight:500;font-display:block;src:url(${FONT_INTER_500}) format('woff2')}`
  + `@font-face{font-family:'WCF';font-weight:700;font-display:block;src:url(${FONT_INTER_700}) format('woff2')}`
  + `body{font-family:'WCF',sans-serif}`;

const MONO_THEME: WcTheme = { rootCSS: "", styleCSS: MONO_STYLE, fav: "#000", win: "#000", live: "#000" };

/** Full-bleed wrapper that shows an inline base64 PNG at exactly 800x480 (no scaling). */
const imgWrap = (b64: string) =>
  `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=800"><title>World Cup 2026</title>`
  + `<style>*{margin:0;padding:0}html,body{width:800px;height:480px;overflow:hidden;background:#fff}`
  + `img{width:800px;height:480px;display:block;image-rendering:pixelated}</style></head>`
  + `<body><img src="data:image/png;base64,${b64}" width="800" height="480"></body></html>`;

const htmlHeaders = {
  "Content-Type": "text/html; charset=utf-8",
  "Cache-Control": "public, max-age=300",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Referrer-Policy": "no-referrer",
};

function parseTestPhase(raw: string | null): WcPhase | null {
  if (raw === "group" || raw === "r32" || raw === "knockout" || raw === "champion") return raw;
  return null;
}

/** Render the Inter HTML source to a crisp 1-bit PNG and cache it (base64 + timestamp). */
async function renderAndCacheImage(env: Env, origin: string): Promise<string> {
  const png = await renderWorldCupBrowserPNG(env, `${origin}/worldcup?variant=src`, IMG_THRESHOLD);
  const b64 = pngToBase64(png);
  await env.CACHE.put(IMG_KEY, JSON.stringify({ b64, ts: Date.now() }), { expirationTtl: 86400 });
  return b64;
}

/**
 * Stale-while-revalidate image fetch. Returns the cached image instantly (refreshing in the
 * background when stale); returns null only on a cold cache with a request context (so the
 * caller serves the HTML fallback this cycle while the first render happens in the background).
 */
async function getWorldCupImageB64(env: Env, origin: string, ctx?: ExecutionContext): Promise<string | null> {
  const cached = (await env.CACHE.get(IMG_KEY, "json")) as { b64: string; ts: number } | null;
  if (cached?.b64) {
    if (Date.now() - cached.ts >= IMG_SOFT_TTL_MS && ctx) {
      ctx.waitUntil(renderAndCacheImage(env, origin).catch((e) => console.error("WC image refresh:", e)));
    }
    return cached.b64;
  }
  if (ctx) {
    ctx.waitUntil(renderAndCacheImage(env, origin).catch((e) => console.error("WC image cold:", e)));
    return null;
  }
  return await renderAndCacheImage(env, origin); // cron path: render synchronously
}

/** Cron warm: render + cache the image so the device never hits a cold cache. */
export async function warmWorldCupImage(env: Env): Promise<void> {
  await renderAndCacheImage(env, WC_ORIGIN);
}

export async function handleWorldCupPage(env: Env, url: URL, ctx?: ExecutionContext): Promise<Response> {
  try {
    const testPhase = parseTestPhase(url.searchParams.get("test-phase"));
    const variant = url.searchParams.get("variant");
    const data = testPhase ? testWorldCupData(testPhase)
      : url.searchParams.has("test") ? testWorldCupData("group")
      : await getWorldCupData(env, ctx);

    const monoHTML = () => new Response(renderWorldCupHTML(data, MONO_THEME), { headers: htmlHeaders });

    // Internal screenshot source + raw-HTML debug view (Inter font, no image, no label).
    // Canned-phase/test previews also serve HTML (the cached image is always live data).
    if (variant === "src" || variant === "html" || testPhase || url.searchParams.has("test")) {
      return monoHTML();
    }

    // Default (and ?variant=image): the pre-dithered, crisp 1-bit image. Falls back to the
    // Inter HTML on a cold cache so the panel is never blank while the first render runs.
    const b64 = await getWorldCupImageB64(env, url.origin, ctx);
    return b64 ? new Response(imgWrap(b64), { headers: htmlHeaders }) : monoHTML();
  } catch (err) {
    console.error("World Cup page error:", err);
    return new Response("World Cup data temporarily unavailable", {
      status: 503,
      headers: { "Content-Type": "text/plain; charset=utf-8", "Retry-After": "300" },
    });
  }
}
