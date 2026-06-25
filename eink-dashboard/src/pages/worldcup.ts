/**
 * /worldcup — World Cup 2026 page for reTerminal E1001 (mono).
 * Pure black; favorite/win/live accents collapse to black + glyph markers.
 */

import type { Env, WcPhase } from "../types";
import { getWorldCupData } from "../worldcup";
import { renderWorldCupHTML, type WcTheme } from "../worldcup-ui";
import { testWorldCupData } from "../worldcup-testdata";
import { FONT_ATKINSON_400, FONT_ATKINSON_700, FONT_INTER_500, FONT_INTER_700 } from "../worldcup-fonts";

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
    const data = testPhase ? testWorldCupData(testPhase)
      : url.searchParams.has("test") ? testWorldCupData("group")
      : await getWorldCupData(env, ctx);

    const html = renderWorldCupHTML(data, MONO_THEME, buildVariantCSS(url.searchParams.get("variant")));
    return new Response(html, {
      headers: {
        "Content-Type": "text/html; charset=utf-8",
        "Cache-Control": "public, max-age=900",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
      },
    });
  } catch (err) {
    console.error("World Cup page error:", err);
    return new Response("World Cup data temporarily unavailable", {
      status: 503,
      headers: { "Content-Type": "text/plain; charset=utf-8", "Retry-After": "300" },
    });
  }
}
