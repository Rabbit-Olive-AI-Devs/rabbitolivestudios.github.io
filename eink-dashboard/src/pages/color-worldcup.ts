/**
 * /color/worldcup — World Cup 2026 page for reTerminal E1002 (Spectra 6).
 * Same layout as /worldcup; Spectra-6 accents (green win, red live, blue favorite).
 * Yellow is never used as foreground (DECISIONS.md #40).
 */

import type { Env, WcPhase } from "../types";
import { getWorldCupData } from "../worldcup";
import { renderWorldCupHTML, type WcTheme } from "../worldcup-ui";
import { testWorldCupData } from "../worldcup-testdata";
import { spectra6CSS } from "../spectra6";

const COLOR_THEME: WcTheme = {
  rootCSS: spectra6CSS(),
  fav: "var(--s6-blue)",
  win: "var(--s6-green)",
  live: "var(--s6-red)",
};

function parseTestPhase(raw: string | null): WcPhase | null {
  if (raw === "group" || raw === "r32" || raw === "knockout" || raw === "champion") return raw;
  return null;
}

export async function handleColorWorldCupPage(env: Env, url: URL, ctx?: ExecutionContext): Promise<Response> {
  try {
    const testPhase = parseTestPhase(url.searchParams.get("test-phase"));
    const data = testPhase ? testWorldCupData(testPhase)
      : url.searchParams.has("test") ? testWorldCupData("group")
      : await getWorldCupData(env, ctx);

    const html = renderWorldCupHTML(data, COLOR_THEME, "");
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
    console.error("Color World Cup page error:", err);
    return new Response("World Cup data temporarily unavailable", {
      status: 503,
      headers: { "Content-Type": "text/plain; charset=utf-8", "Retry-After": "300" },
    });
  }
}
