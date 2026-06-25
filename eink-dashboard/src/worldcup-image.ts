/**
 * Nuclear-option crispness variant (DECISIONS #48): render the mono /worldcup group-phase
 * view server-side as a pixel-perfect 800x480 grayscale PNG using the 8x8 bitmap font.
 * Zero anti-aliasing -> zero gray edges -> the SenseCraft cloud render/dither cannot smudge it.
 * Tradeoff: monospace pixel typeface (not Helvetica), names truncated to fit the column.
 */

import { drawText, measureText } from "./font";
import { encodePNGGray8 } from "./png";
import { displayName, matchCell, pickRotatingGroup, FAVORITE_TEAM, teamCode } from "./worldcup-ui";
import type { WorldCupData, WcMatch, WcGroup } from "./types";

const W = 800;
const H = 480;
const PAD = 14;

/** Truncate `text` so it fits within maxW px at the given scale (monospace). */
function fit(text: string, scale: number, maxW: number): string {
  if (measureText(text, scale) <= maxW) return text;
  let t = text;
  while (t.length > 1 && measureText(t + ".", scale) > maxW) t = t.slice(0, -1);
  return t + ".";
}

/** Draw a solid horizontal black rule. */
function hline(buf: Uint8Array, x0: number, x1: number, y: number, thick = 2): void {
  for (let t = 0; t < thick; t++) {
    const yy = y + t;
    if (yy < 0 || yy >= H) continue;
    for (let x = x0; x < x1; x++) buf[yy * W + x] = 0;
  }
}

/** "ECU v GER" style label for a match, full names truncated to fit maxW. */
function matchLabel(m: WcMatch, scale: number, maxW: number): string {
  const cell = matchCell(m); // score or kickoff time
  const home = displayName(m.home) || teamCode(m.home);
  const away = displayName(m.away) || teamCode(m.away);
  // Reserve room for "<cell> " + " v " between names.
  const prefix = cell + "  ";
  const sep = " v ";
  const avail = maxW - measureText(prefix + sep, scale);
  const half = Math.max(24, Math.floor(avail / 2));
  return prefix + fit(home, scale, half) + sep + fit(away, scale, half);
}

/** Render the group-phase view (today + latest results + one standings table) to a PNG. */
export async function renderWorldCupImagePNG(data: WorldCupData): Promise<Uint8Array> {
  const buf = new Uint8Array(W * H).fill(255); // white

  // --- Header ---
  drawText(buf, W, H, PAD, 12, "FIFA WORLD CUP 2026", 2);
  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "America/Chicago" });
  drawText(buf, W, H, W - PAD - measureText(dateStr, 2), 12, dateStr, 2);
  hline(buf, PAD, W - PAD, 40, 3);

  // --- Two columns: TODAY | LATEST RESULTS ---
  const col1 = PAD, col2 = 412;
  const col1W = col2 - col1 - 12, col2W = W - PAD - col2;
  drawText(buf, W, H, col1, 50, "TODAY", 2);
  drawText(buf, W, H, col2, 50, "LATEST RESULTS", 2);

  const rowY0 = 78, rowStep = 23;
  const today = data.todayMatches.slice(0, 6);
  const results = data.recentResults.slice(0, 6);
  for (let i = 0; i < 6; i++) {
    const y = rowY0 + i * rowStep;
    if (today[i]) drawText(buf, W, H, col1, y, matchLabel(today[i], 2, col1W), 2);
    if (results[i]) drawText(buf, W, H, col2, y, matchLabel(results[i], 2, col2W), 2);
  }

  // --- Divider + standings table ---
  hline(buf, PAD, W - PAD, 224, 3);
  const group: WcGroup | null = pickRotatingGroup(data.groups, data.todayMatches, FAVORITE_TEAM, Math.floor(Date.now() / 60000));
  if (group) {
    drawText(buf, W, H, PAD, 234, "GROUP " + group.name, 2);
    // Column x-positions for the stat grid (right edge ~ W-PAD).
    const cols = { P: 430, W: 492, D: 554, L: 616, GD: 678, Pts: 744 };
    const headY = 262;
    drawText(buf, W, H, cols.P, headY, "P", 2);
    drawText(buf, W, H, cols.W, headY, "W", 2);
    drawText(buf, W, H, cols.D, headY, "D", 2);
    drawText(buf, W, H, cols.L, headY, "L", 2);
    drawText(buf, W, H, cols.GD, headY, "GD", 2);
    drawText(buf, W, H, cols.Pts, headY, "Pts", 2);
    const tRowY0 = 288, tStep = 30;
    group.rows.slice(0, 4).forEach((r, i) => {
      const y = tRowY0 + i * tStep;
      const check = r.qualifying ? " *" : "";
      drawText(buf, W, H, PAD, y, String(r.position), 2);
      drawText(buf, W, H, PAD + 28, y, fit(displayName(r.team) + check, 2, cols.P - (PAD + 28) - 8), 2);
      drawText(buf, W, H, cols.P, y, String(r.played), 2);
      drawText(buf, W, H, cols.W, y, String(r.won), 2);
      drawText(buf, W, H, cols.D, y, String(r.drawn), 2);
      drawText(buf, W, H, cols.L, y, String(r.lost), 2);
      drawText(buf, W, H, cols.GD, y, (r.goalDifference >= 0 ? "+" : "") + r.goalDifference, 2);
      drawText(buf, W, H, cols.Pts, y, String(r.points), 2);
    });
  }

  // corner label so the variant is identifiable on the panel
  drawText(buf, W, H, W - PAD - measureText("image", 1), H - 10, "image", 1);

  return encodePNGGray8(buf, W, H);
}
