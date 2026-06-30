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

/**
 * America/Chicago calendar date of each bracket SLOT (same top-to-bottom / left-then-right order as
 * above), from the official FIFA bracket. Used to place UNPLAYED matches into the right slot: the
 * feed gives each its real date but not its slot, so we match the feed match's `dateChicago` to the
 * slot's date here. (Two slots can share a date — fine, they show the same date.) DECISIONS #53.
 */
const R16_DATES = ["2026-07-04", "2026-07-04", "2026-07-06", "2026-07-06", "2026-07-05", "2026-07-05", "2026-07-07", "2026-07-07"];
const QF_DATES = ["2026-07-09", "2026-07-10", "2026-07-11", "2026-07-11"];
const SF_DATES = ["2026-07-14", "2026-07-15"];
const FINAL_DATES = ["2026-07-19"];

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
 * Place each later-round source match into its true bracket slot.
 *  1. by the seeded team(s) matching the slot's feeder winners (played/seeded matches → real result);
 *  2. otherwise by the match's date matching the slot's official date (unplayed matches → right date);
 *  3. otherwise fill remaining slots in order (last-resort, e.g. a reschedule we haven't pinned).
 */
function orderRound(feeders: (WcMatch | null)[], pool: WcMatch[], slotDates: string[]): (WcMatch | null)[] {
  const n = slotDates.length;
  const out: (WcMatch | null)[] = new Array(n).fill(null);
  const used = new Set<number>();
  const take = (i: number, m: WcMatch | undefined) => { if (m) { out[i] = m; used.add(m.id); } };
  // 1. seeded → by feeder winners
  for (let i = 0; i < n; i++) {
    const known = [winnerCode(feeders[2 * i]), winnerCode(feeders[2 * i + 1])].filter((c): c is string => !!c);
    if (known.length) take(i, pool.find((x) => !used.has(x.id) && known.every((c) => hasCode(x, c))));
  }
  // 2. unplayed → by official slot date
  for (let i = 0; i < n; i++) {
    if (!out[i]) take(i, pool.find((x) => !used.has(x.id) && x.dateChicago === slotDates[i]));
  }
  // 3. leftover → fill in order
  for (let i = 0; i < n; i++) {
    if (!out[i]) take(i, pool.find((x) => !used.has(x.id)));
  }
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
  const r16 = orderRound(r32, byStage("R16"), R16_DATES);
  const qf = orderRound(r16, byStage("QF"), QF_DATES);
  const sf = orderRound(qf, byStage("SF"), SF_DATES);
  const final = orderRound(sf, byStage("FINAL"), FINAL_DATES)[0];
  return { r32, r16, qf, sf, final };
}
