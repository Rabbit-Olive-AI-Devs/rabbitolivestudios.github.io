/**
 * football-data.org client for the FIFA World Cup (competition code WC).
 * Primary World Cup data source. Pure `normalizeFootballData` is unit-tested;
 * `fetchFootballData` wraps the two HTTP calls and returns null on any failure.
 */

import { fetchWithTimeout } from "./fetch-timeout";
import { chicagoDateOf, chicagoTimeOf } from "./worldcup-ui";
import type {
  Env, WorldCupData, WcMatch, WcStage, WcStatus, WcGroup, WcStandingRow, WcTeam,
} from "./types";

const BASE = "https://api.football-data.org/v4/competitions/WC";
const TIMEOUT_MS = 5000;

export function mapStage(s: string): WcStage {
  switch (s) {
    case "GROUP_STAGE": return "GROUP";
    case "LAST_32": return "R32";
    case "LAST_16": return "R16";
    case "QUARTER_FINALS": return "QF";
    case "SEMI_FINALS": return "SF";
    case "THIRD_PLACE": return "THIRD";
    case "FINAL": return "FINAL";
    default: return "GROUP";
  }
}

export function mapStatus(s: string): WcStatus {
  switch (s) {
    case "IN_PLAY":
    case "PAUSED": return "LIVE";
    case "FINISHED": return "FINISHED";
    default: return "SCHEDULED"; // SCHEDULED, TIMED, POSTPONED, SUSPENDED, CANCELLED
  }
}

function team(t: any): WcTeam {
  return { name: t?.name ?? "", code: (t?.tla ?? "").toUpperCase() };
}

function groupLetter(g: string | undefined | null): string | undefined {
  if (!g) return undefined;
  const m = g.match(/([A-L])\s*$/i);
  return m ? m[1].toUpperCase() : g;
}

function normalizeMatch(m: any): WcMatch {
  const iso = m.utcDate as string;
  return {
    id: m.id,
    stage: mapStage(m.stage),
    group: groupLetter(m.group),
    status: mapStatus(m.status),
    kickoffISO: iso,
    dateChicago: chicagoDateOf(iso),
    timeChicago: chicagoTimeOf(iso),
    home: team(m.homeTeam),
    away: team(m.awayTeam),
    // fullTime folds the shootout in (e.g. a 0-0 won 4-2 on pens reports 4-2); penalties are
    // exposed separately so the UI can show "0-0 (4-2)". See DECISIONS #50.
    homeScore: m.score?.fullTime?.home ?? null,
    awayScore: m.score?.fullTime?.away ?? null,
    penaltyHome: m.score?.penalties?.home ?? null,
    penaltyAway: m.score?.penalties?.away ?? null,
  };
}

function normalizeGroups(standingsJson: any): WcGroup[] {
  const out: WcGroup[] = [];
  for (const s of standingsJson?.standings ?? []) {
    const name = groupLetter(s.group) ?? "";
    if (!name) continue;
    const rows: WcStandingRow[] = (s.table ?? []).map((r: any) => ({
      position: r.position,
      team: team(r.team),
      played: r.playedGames ?? 0,
      won: r.won ?? 0,
      drawn: r.draw ?? 0,
      lost: r.lost ?? 0,
      goalDifference: r.goalDifference ?? 0,
      points: r.points ?? 0,
      qualifying: (r.position ?? 99) <= 2, // top 2 only; third place never auto-marked
    }));
    out.push({ name, rows });
  }
  out.sort((a, b) => a.name.localeCompare(b.name));
  return out;
}

/** Build WorldCupData from football-data matches + standings JSON (pure). Phase set later. */
export function normalizeFootballData(matchesJson: any, standingsJson: any): WorldCupData {
  const matches: WcMatch[] = (matchesJson?.matches ?? []).map(normalizeMatch);
  const groups = normalizeGroups(standingsJson);
  const knockout = matches.filter((m) => m.stage !== "GROUP");

  // recentResults: finished matches on the most recent finished date.
  const finished = matches.filter((m) => m.status === "FINISHED");
  let recentResults: WcMatch[] = [];
  if (finished.length > 0) {
    const dates = finished.map((m) => m.dateChicago).sort();
    const latestDate = dates[dates.length - 1];
    recentResults = finished.filter((m) => m.dateChicago === latestDate);
  }

  const champion = matches.find((m) => m.stage === "FINAL" && m.status === "FINISHED");
  const championTeam: WcTeam | null = champion && champion.homeScore !== null && champion.awayScore !== null
    ? (champion.homeScore > champion.awayScore ? champion.home : champion.away)
    : null;

  return {
    source: "football-data",
    phase: "group", // overwritten by the data layer via computePhase
    todayMatches: [],
    recentResults,
    groups,
    knockout,
    champion: championTeam,
    generatedAt: 0,
    // full match list is attached by the data layer for phase/today computation
    ...({ _allMatches: matches } as any),
  };
}

/** Fetch matches + standings from football-data. Returns null on any failure. */
export async function fetchFootballData(env: Env): Promise<WorldCupData | null> {
  const key = env.FOOTBALL_DATA_KEY;
  if (!key) return null;
  try {
    const headers = { "X-Auth-Token": key };
    const [mres, sres] = await Promise.all([
      fetchWithTimeout(`${BASE}/matches`, { headers }, TIMEOUT_MS),
      fetchWithTimeout(`${BASE}/standings`, { headers }, TIMEOUT_MS),
    ]);
    if (!mres.ok) throw new Error(`football-data matches ${mres.status}`);
    const matchesJson: any = await mres.json();
    // Standings can 404 in mid-tournament edge cases; tolerate it.
    const standingsJson: any = sres.ok ? await sres.json() : { standings: [] };
    return normalizeFootballData(matchesJson, standingsJson);
  } catch (err) {
    console.error("football-data fetch failed:", err);
    return null;
  }
}
