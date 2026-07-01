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

import type { WcMatch, WcStage, WcTeam } from "./types";
import { teamCode } from "./worldcup-ui";

const EMPTY_TEAM: WcTeam = { name: "", code: "" };
const teamKnown = (t: WcTeam | null | undefined): boolean =>
  !!(t && ((t.name && t.name.trim()) || (t.code && t.code.trim())));

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

/** Winning team, or null when not finished / level (fullTime folds penalties, so a shootout winner reads here). */
function winnerTeam(m: WcMatch | null | undefined): WcTeam | null {
  if (!m || m.status !== "FINISHED" || m.homeScore == null || m.awayScore == null) return null;
  if (m.homeScore > m.awayScore) return m.home;
  if (m.awayScore > m.homeScore) return m.away;
  return null;
}

const hasCode = (m: WcMatch, c: string): boolean => teamCode(m.home) === c || teamCode(m.away) === c;
const sameTeams = (m: WcMatch, a: string, b: string): boolean => {
  const h = teamCode(m.home), w = teamCode(m.away);
  return (h === a && w === b) || (h === b && w === a);
};

/**
 * Build each slot of a later round, advancing winners so a team appears in its next-round box the
 * moment it wins — even before its opponent is decided (the feed only seeds a knockout match once
 * BOTH sides are known, so we can't wait for it). For slot i, fed by feeders[2i] (home) and
 * feeders[2i+1] (away):
 *  - If the feed already seeded a real match with both known winners → use it (real teams + score).
 *  - Else if a feeder is decided → overlay the advancing team(s) (unknown side = TBD) onto the
 *    feed's dated-but-unseeded match for this slot (keeps its real id/date/kickoff), else a synth.
 *  - Else (no feeder decided yet) → the feed's unplayed match for this slot (its date placeholder).
 * Slot ↔ match matching: by teams first (order-independent), then by the slot's official date.
 */
function orderRound(feeders: (WcMatch | null)[], pool: WcMatch[], slotDates: string[], stage: WcStage): (WcMatch | null)[] {
  const n = slotDates.length;
  const out: (WcMatch | null)[] = new Array(n).fill(null);
  const used = new Set<number>();
  for (let i = 0; i < n; i++) {
    const hw = winnerTeam(feeders[2 * i]);       // team advancing into this slot's home side (or null)
    const aw = winnerTeam(feeders[2 * i + 1]);   // ... away side
    const knownCodes = [hw, aw].filter((t): t is WcTeam => !!t).map(teamCode);

    // Best source match for this slot: one seeded with all known winners, else the feed's
    // dated (but maybe unseeded) match for this slot — gives us its real id/date/kickoff.
    let src = knownCodes.length
      ? pool.find((x) => !used.has(x.id) && knownCodes.every((c) => hasCode(x, c)))
      : undefined;
    if (!src) src = pool.find((x) => !used.has(x.id) && x.dateChicago === slotDates[i]);
    if (src) used.add(src.id);

    if (!hw && !aw) { out[i] = src ?? null; continue; }              // no feeder decided → placeholder
    if (src && (teamKnown(src.home) || teamKnown(src.away))) { out[i] = src; continue; } // feed seeded it

    // A feeder is decided but the feed hasn't seeded this match → advance the winner(s) ourselves.
    const base: WcMatch = src ?? {
      id: -(1000 + i), stage, group: undefined, status: "SCHEDULED",
      kickoffISO: "", dateChicago: slotDates[i], timeChicago: "",
      home: EMPTY_TEAM, away: EMPTY_TEAM, homeScore: null, awayScore: null,
    };
    out[i] = { ...base, home: hw ?? EMPTY_TEAM, away: aw ?? EMPTY_TEAM };
  }
  // Leftover unmatched matches (e.g. a reschedule) fill any still-empty slots in order.
  for (let i = 0; i < n; i++) {
    if (!out[i]) {
      const m = pool.find((x) => !used.has(x.id));
      if (m) { out[i] = m; used.add(m.id); }
    }
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
  const r16 = orderRound(r32, byStage("R16"), R16_DATES, "R16");
  const qf = orderRound(r16, byStage("QF"), QF_DATES, "QF");
  const sf = orderRound(qf, byStage("SF"), SF_DATES, "SF");
  const final = orderRound(sf, byStage("FINAL"), FINAL_DATES, "FINAL")[0];
  return { r32, r16, qf, sf, final };
}
