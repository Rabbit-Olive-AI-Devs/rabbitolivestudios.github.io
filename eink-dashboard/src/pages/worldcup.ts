/**
 * /worldcup — World Cup 2026 page for reTerminal E1001 (mono).
 * Pure black; favorite/win/live accents collapse to black + glyph markers.
 */

import type { Env, WcPhase } from "../types";
import { getWorldCupData } from "../worldcup";
import { renderWorldCupHTML, type WcTheme } from "../worldcup-ui";
import { testWorldCupData } from "../worldcup-testdata";

const MONO_THEME: WcTheme = { rootCSS: "", fav: "#000", win: "#000", live: "#000" };

function parseTestPhase(raw: string | null): WcPhase | null {
  if (raw === "group" || raw === "r32" || raw === "knockout" || raw === "champion") return raw;
  return null;
}

export async function handleWorldCupPage(env: Env, url: URL, ctx?: ExecutionContext): Promise<Response> {
  try {
    const testPhase = parseTestPhase(url.searchParams.get("test-phase"));
    const data = testPhase ? testWorldCupData(testPhase)
      : url.searchParams.has("test") ? testWorldCupData("group")
      : await getWorldCupData(env, ctx);

    const html = renderWorldCupHTML(data, MONO_THEME, "");
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
