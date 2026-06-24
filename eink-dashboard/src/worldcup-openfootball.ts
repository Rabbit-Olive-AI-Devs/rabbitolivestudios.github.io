/**
 * openfootball/worldcup.json fallback source (no API key, public domain).
 * Schedule + scores; group standings are DERIVED here from finished matches
 * (football-data computes them server-side, openfootball does not).
 * Updated ~once daily upstream — fallback only.
 */

import { fetchWithTimeout } from "./fetch-timeout";
import { chicagoDateOf, chicagoTimeOf, teamCode } from "./worldcup-ui";
import type { Env, WorldCupData, WcMatch, WcStage, WcGroup, WcStandingRow, WcTeam } from "./types";

const URL_2026 = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json";
const TIMEOUT_MS = 5000;

function mkTeam(name: string): WcTeam {
  return { name: name ?? "", code: teamCode({ name: name ?? "", code: "" }) };
}

function groupLetter(g: string | undefined): string | undefined {
  if (!g) return undefined;
  const m = g.match(/([A-L])\s*$/i);
  return m ? m[1].toUpperCase() : undefined;
}

function isoFrom(date: string, time: string | undefined): string {
  // openfootball times are local to the venue; we only need an ordering key +
  // Chicago conversion. Treat as UTC for stable, deterministic bucketing.
  const t = time && /^\d{1,2}:\d{2}$/.test(time) ? time : "12:00";
  const [h, mn] = t.split(":");
  return `${date}T${h.padStart(2, "0")}:${mn}:00Z`;
}

function normalizeMatch(m: any, idx: number): WcMatch {
  const ft = m.score?.ft;
  const finished = Array.isArray(ft) && ft.length === 2;
  const iso = isoFrom(m.date, m.time);
  const stage: WcStage = "GROUP"; // openfootball group file is group-stage; knockouts via cup_finals not parsed here
  return {
    id: idx + 1,
    stage,
    group: groupLetter(m.group),
    status: finished ? "FINISHED" : "SCHEDULED",
    kickoffISO: iso,
    dateChicago: chicagoDateOf(iso),
    timeChicago: chicagoTimeOf(iso),
    home: mkTeam(m.team1),
    away: mkTeam(m.team2),
    homeScore: finished ? ft[0] : null,
    awayScore: finished ? ft[1] : null,
  };
}

/** Derive group tables from finished group matches (points/GD/played; top 2 qualifying). */
function deriveGroups(matches: WcMatch[]): WcGroup[] {
  const byGroup = new Map<string, Map<string, WcStandingRow>>();
  const ensure = (g: string, t: WcTeam): WcStandingRow => {
    if (!byGroup.has(g)) byGroup.set(g, new Map());
    const tbl = byGroup.get(g)!;
    const code = t.code;
    if (!tbl.has(code)) {
      tbl.set(code, { position: 0, team: t, played: 0, won: 0, drawn: 0, lost: 0, goalDifference: 0, points: 0, qualifying: false });
    }
    return tbl.get(code)!;
  };
  for (const m of matches) {
    if (m.stage !== "GROUP" || !m.group || m.status !== "FINISHED") continue;
    if (m.homeScore === null || m.awayScore === null) continue;
    const h = ensure(m.group, m.home);
    const a = ensure(m.group, m.away);
    h.played++; a.played++;
    h.goalDifference += m.homeScore - m.awayScore;
    a.goalDifference += m.awayScore - m.homeScore;
    if (m.homeScore > m.awayScore) { h.won++; h.points += 3; a.lost++; }
    else if (m.homeScore < m.awayScore) { a.won++; a.points += 3; h.lost++; }
    else { h.drawn++; a.drawn++; h.points++; a.points++; }
  }
  const groups: WcGroup[] = [];
  for (const [name, tbl] of byGroup) {
    const rows = [...tbl.values()].sort(
      (x, y) => y.points - x.points || y.goalDifference - x.goalDifference || teamCode(x.team).localeCompare(teamCode(y.team)),
    );
    rows.forEach((r, i) => { r.position = i + 1; r.qualifying = i < 2; });
    groups.push({ name, rows });
  }
  groups.sort((a, b) => a.name.localeCompare(b.name));
  return groups;
}

/** Build WorldCupData from openfootball JSON (pure). Phase set later by the data layer. */
export function normalizeOpenFootball(json: any): WorldCupData {
  const matches: WcMatch[] = [];
  let i = 0;
  for (const round of json?.rounds ?? []) {
    for (const m of round?.matches ?? []) {
      matches.push(normalizeMatch(m, i++));
    }
  }
  const groups = deriveGroups(matches);
  const finished = matches.filter((m) => m.status === "FINISHED");
  let recentResults: WcMatch[] = [];
  if (finished.length > 0) {
    const dates = finished.map((m) => m.dateChicago).sort();
    const latestDate = dates[dates.length - 1];
    recentResults = finished.filter((m) => m.dateChicago === latestDate);
  }
  return {
    source: "openfootball",
    phase: "group",
    todayMatches: [],
    recentResults,
    groups,
    knockout: matches.filter((m) => m.stage !== "GROUP"),
    champion: null,
    generatedAt: 0,
    ...({ _allMatches: matches } as any),
  };
}

/** Fetch + normalize the openfootball JSON. Returns null on any failure. */
export async function fetchOpenFootball(_env: Env): Promise<WorldCupData | null> {
  try {
    const res = await fetchWithTimeout(URL_2026, undefined, TIMEOUT_MS);
    if (!res.ok) throw new Error(`openfootball ${res.status}`);
    const json: any = await res.json();
    return normalizeOpenFootball(json);
  } catch (err) {
    console.error("openfootball fetch failed:", err);
    return null;
  }
}
