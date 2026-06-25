/**
 * /worldcup — World Cup 2026 page for reTerminal E1001 (mono).
 * Pure black; favorite/win/live accents collapse to black + glyph markers.
 */

import type { Env, WcPhase } from "../types";
import { getWorldCupData } from "../worldcup";
import { renderWorldCupHTML, type WcTheme } from "../worldcup-ui";
import { testWorldCupData } from "../worldcup-testdata";
import { FONT_ATKINSON_400, FONT_ATKINSON_700, FONT_INTER_500, FONT_INTER_700 } from "../worldcup-fonts";
import { renderWorldCupImagePNG } from "../worldcup-image";
import { renderWorldCupBrowserPNG } from "../worldcup-browser-image";
import { pngToBase64 } from "../png";

/** Full-bleed wrapper that shows an inline base64 PNG at exactly 800x480 (no scaling). */
const imgWrap = (b64: string) =>
  `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=800"><title>World Cup 2026</title>`
  + `<style>*{margin:0;padding:0}html,body{width:800px;height:480px;overflow:hidden;background:#fff}`
  + `img{width:800px;height:480px;display:block;image-rendering:pixelated}</style></head>`
  + `<body><img src="data:image/png;base64,${b64}" width="800" height="480"></body></html>`;

const MONO_THEME: WcTheme = { rootCSS: "", fav: "#000", win: "#000", live: "#000" };

function parseTestPhase(raw: string | null): WcPhase | null {
  if (raw === "group" || raw === "r32" || raw === "knockout" || raw === "champion") return raw;
  return null;
}

/** Tiny corner label so the variant is identifiable when A/B-ing on the panel. */
const tag = (name: string) =>
  `body::after{content:'${name}';position:fixed;bottom:1px;right:3px;font-size:9px;font-weight:400;color:#000;letter-spacing:1px}`;

/**
 * Crispness A/B variants (DECISIONS #48). `?variant=atkinson|inter|small`; default = current page.
 * Returned as pageCSS appended after the base stylesheet so it overrides via specificity/!important.
 */
function buildVariantCSS(raw: string | null): string {
  switch (raw) {
    case "atkinson":
      return `@font-face{font-family:'WCF';font-weight:400;font-display:block;src:url(${FONT_ATKINSON_400}) format('woff2')}`
        + `@font-face{font-family:'WCF';font-weight:700;font-display:block;src:url(${FONT_ATKINSON_700}) format('woff2')}`
        + `body{font-family:'WCF',sans-serif!important}` + tag("atkinson");
    case "inter":
      return `@font-face{font-family:'WCF';font-weight:500;font-display:block;src:url(${FONT_INTER_500}) format('woff2')}`
        + `@font-face{font-family:'WCF';font-weight:700;font-display:block;src:url(${FONT_INTER_700}) format('woff2')}`
        + `body{font-family:'WCF',sans-serif!important}` + tag("inter");
    case "small":
      return `.wc-row{font-size:17px!important;line-height:22px!important}`
        + `.wc-table{font-size:17px!important}.wc-table td{line-height:22px!important}`
        + `.wc-bteam{font-size:16px!important;line-height:21px!important}`
        + `.wc-group-name{font-size:16px!important}.wc-title{font-size:22px!important}.wc-sub{font-size:15px!important}`
        + tag("small");
    default:
      return "";
  }
}

export async function handleWorldCupPage(env: Env, url: URL, ctx?: ExecutionContext): Promise<Response> {
  try {
    const testPhase = parseTestPhase(url.searchParams.get("test-phase"));
    const variant = url.searchParams.get("variant");
    const data = testPhase ? testWorldCupData(testPhase)
      : url.searchParams.has("test") ? testWorldCupData("group")
      : await getWorldCupData(env, ctx);

    const htmlHeaders = {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "public, max-age=900",
      "X-Content-Type-Options": "nosniff",
      "X-Frame-Options": "DENY",
      "Referrer-Policy": "no-referrer",
    };

    // Nuclear option, done right: screenshot the real Inter HTML via Browser Rendering, then
    // threshold to pure 1-bit ourselves so SenseCraft can't fog it. Cached (browser is expensive).
    if (variant === "image") {
      const thr = Math.min(255, Math.max(1, parseInt(url.searchParams.get("thr") || "160", 10) || 160));
      const cacheKey = `wc:image:v2:${thr}`;
      let b64 = url.searchParams.has("fresh") ? null : await env.CACHE.get(cacheKey);
      if (!b64) {
        try {
          // Screenshot the Inter page, but label its corner "image" so it's distinguishable on the panel.
          const png = await renderWorldCupBrowserPNG(env, `${url.origin}/worldcup?variant=inter&tag=image`, thr);
          b64 = pngToBase64(png);
          await env.CACHE.put(cacheKey, b64, { expirationTtl: 900 });
        } catch (e) {
          return new Response("browser render failed: " + String((e as { message?: string })?.message ?? e), {
            status: 500, headers: { "Content-Type": "text/plain; charset=utf-8" },
          });
        }
      }
      return new Response(imgWrap(b64), { headers: htmlHeaders });
    }

    // Bitmap fallback (8x8 pixel font, no anti-aliasing) — kept for comparison.
    if (variant === "bitmap") {
      return new Response(imgWrap(pngToBase64(await renderWorldCupImagePNG(data))), { headers: htmlHeaders });
    }

    // Optional ?tag= overrides the corner label (used so the browser-screenshotted page reads "image").
    let pageCSS = buildVariantCSS(variant);
    const customTag = (url.searchParams.get("tag") || "").replace(/[^a-z0-9]/gi, "").slice(0, 12);
    if (customTag) pageCSS += tag(customTag);
    const html = renderWorldCupHTML(data, MONO_THEME, pageCSS);
    return new Response(html, { headers: htmlHeaders });
  } catch (err) {
    console.error("World Cup page error:", err);
    return new Response("World Cup data temporarily unavailable", {
      status: 503,
      headers: { "Content-Type": "text/plain; charset=utf-8", "Retry-After": "300" },
    });
  }
}
