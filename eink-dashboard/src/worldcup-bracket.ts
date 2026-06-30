/**
 * Hardcoded 2026 FIFA World Cup knockout bracket structure.
 *
 * Why hardcode: football-data exposes the matches and seeds the advancing teams, but its match
 * ordering does NOT encode the bracket tree (its R16/QF id order doesn't line up with R32), and the
 * feed gives no explicit feeder links. So a converging 2-sided bracket can't be positioned from the
 * feed alone — a team ends up on the wrong side. We pin the real tree here (from the official FIFA
 * bracket) and place every match by TEAMS, so each team stays on its correct side. DECISIONS #52.
 *
 * `BRACKET_R32` is the official R32 order, top-to-bottom, LEFT half (8) then RIGHT half (8). The
 * R16/QF/SF/Final tree then follows the standard adjacent pairing within that order (slot i fed by
 * positions 2i and 2i+1), which the rendering splits back into left/right halves.
 */

import type { WcMatch, WcStage } from "./types";
import { teamCode } from "./worldcup-ui";

/** Official 2026 R32 order (left 8 top-to-bottom, then right 8), by FIFA-code team pairs. */
export const BRACKET_R32: [string, string][] = [
  ["GER", "PAR"], ["FRA", "SWE"], ["RSA", "CAN"], ["NED", "MAR"], // left top → bottom
  ["POR", "CRO"], ["ESP", "AUT"], ["USA", "BIH"], ["BEL", "SEN"],
  ["BRA", "JPN"], ["CIV", "NOR"], ["MEX", "ECU"], ["ENG", "COD"], // right top → bottom
  ["ARG", "CPV"], ["AUS", "EGY"], ["SUI", "ALG"], ["COL", "GHA"],
];

export interface OrderedBracket {
  r32: (WcMatch | null)[];   // 16, bracket order (left 8, right 8)
  r16: (WcMatch | null)[];   // 8
  qf: (WcMatch | null)[];    // 4
  sf: (WcMatch | null)[];    // 2
  final: WcMatch | null;
}

/** Winning team's code, or null when not finished / level (fullTime folds penalties, so a shootout winner reads here). */
function winnerCode(m: WcMatch | null | undefined): string | null {
  if (!m || m.status !== "FINISHED" || m.homeScore == null || m.awayScore == null) return null;
  if (m.homeScore > m.awayScore) return teamCode(m.home);
  if (m.awayScore > m.homeScore) return teamCode(m.away);
  return null;
}

const hasCode = (m: WcMatch, c: string): boolean => teamCode(m.home) === c || teamCode(m.away) === c;
const sameTeams = (m: WcMatch, a: string, b: string): boolean => {
  const h = teamCode(m.home), w = teamCode(m.away);
  return (h === a && w === b) || (h === b && w === a);
};

/**
 * Place each later-round source match into its true bracket slot by matching the seeded team(s) to
 * the slot's feeder winners. Slots whose feeders aren't decided yet take the leftover source matches
 * (all-TBD placeholders) in order, so every slot still shows a real date.
 */
function orderRound(feeders: (WcMatch | null)[], pool: WcMatch[], n: number): (WcMatch | null)[] {
  const out: (WcMatch | null)[] = new Array(n).fill(null);
  const used = new Set<number>();
  for (let i = 0; i < n; i++) {
    const known = [winnerCode(feeders[2 * i]), winnerCode(feeders[2 * i + 1])].filter((c): c is string => !!c);
    if (known.length === 0) continue;
    const m = pool.find((x) => !used.has(x.id) && known.every((c) => hasCode(x, c)));
    if (m) { out[i] = m; used.add(m.id); }
  }
  const leftover = pool.filter((x) => !used.has(x.id));
  let li = 0;
  for (let i = 0; i < n; i++) if (!out[i] && li < leftover.length) out[i] = leftover[li++];
  return out;
}

/**
 * Reorder football-data's knockout matches into true bracket positions. R32 is matched to the
 * hardcoded order by teams; each later round is matched to its slot by the feeder winners. The
 * result feeds the converging layout directly (left half then right half, top-to-bottom).
 */
export function orderKnockout(knockout: WcMatch[]): OrderedBracket {
  const byStage = (s: WcStage) => knockout.filter((m) => m.stage === s);
  const r32pool = byStage("R32");
  const r32 = BRACKET_R32.map(([a, b]) => r32pool.find((m) => sameTeams(m, a, b)) ?? null);
  const r16 = orderRound(r32, byStage("R16"), 8);
  const qf = orderRound(r16, byStage("QF"), 4);
  const sf = orderRound(qf, byStage("SF"), 2);
  const final = orderRound(sf, byStage("FINAL"), 1)[0];
  return { r32, r16, qf, sf, final };
}
