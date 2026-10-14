/**
 * Desk panel Flamengo board: the league table, the next match, the last result
 * and recent form, read from ESPN's public keyless feeds (DECISIONS #70).
 *
 * Everything the page shows lives in one KV entry, `flamengo:v1:state`. A panel
 * request always draws from that entry and refreshes it in the background when
 * it has aged: every six hours on a quiet day, every half hour from kickoff
 * until four hours after, so a result lands soon after the whistle. That is at
 * most about a dozen writes a day (DECISIONS #62).
 *
 * ESPN's table does not say which places qualify for what, so the zone lines
 * come from FotMob's table legend, the one feed here that is not ESPN's.
 *
 * Club crests are fetched at refresh time and kept in the same entry as 1-bit
 * bitmaps. They are never committed to the repository.
 */

import type { Env } from "./types";
import { flamengoCacheKey } from "./cache-keys";
import { fetchWithTimeout } from "./fetch-timeout";
import { crestBits } from "./hud/crest";
import { withBudget } from "./with-budget";

export const FLAMENGO_ID = "819";

const ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer";
const RESULTS_URL = `${ESPN}/all/teams/${FLAMENGO_ID}/schedule`;
const FIXTURES_URL = `${RESULTS_URL}?fixture=true`;
const TABLE_URL = "https://site.api.espn.com/apis/v2/sports/soccer/bra.1/standings";
/** FotMob's Serie A table; only its legend (which places lead where) is used. */
const ZONES_URL = "https://www.fotmob.com/api/data/tltable?leagueId=268";
const teamUrl = (id: string) => `${ESPN}/all/teams/${encodeURIComponent(id)}`;
const crestUrl = (id: string) =>
  `https://a.espncdn.com/combiner/i?img=/i/teamlogos/soccer/500/${encodeURIComponent(id)}.png&w=96&h=96`;

const USER_AGENT = "eink-dashboard (personal project)";
const KV_TTL_SECONDS = 604800;
const HOUR = 3_600_000;
/** How old the entry may get on a quiet day. */
const QUIET_TTL_MS = 6 * HOUR;
/** How old it may get while a match is on or has just ended. */
const MATCH_TTL_MS = HOUR / 2;
/** From kickoff, how long a match keeps the short refresh and counts as the one being played. */
const MATCH_WINDOW_MS = 4 * HOUR;
const COLD_BUDGET_MS = 6000;
const KEPT_RESULTS = 12;
const KEPT_FIXTURES = 4;

export interface FlaTeam { id: string; name: string; abbr: string }

export interface FlaMatch {
  id: string;
  /** Kickoff, milliseconds since the epoch. */
  at: number;
  league: string;
  /** Round of a knockout tournament, such as "Semifinals". */
  stage?: string;
  home: FlaTeam;
  away: FlaTeam;
  /** Both set once the match is over. */
  homeScore: number | null;
  awayScore: number | null;
  done: boolean;
  venue?: string;
}

export interface FlaRow {
  pos: number; id: string; name: string;
  p: number; w: number; d: number; l: number; gd: string; pts: number;
}

export interface FlaState {
  fetchedAt: number;
  table: FlaRow[];
  /** Finished matches, oldest first. */
  results: FlaMatch[];
  /** Matches still to come, soonest first. */
  fixtures: FlaMatch[];
  /** Places after which one zone of the table ends and another begins, e.g. [4, 5, 11, 16]. */
  zones: number[];
  /** A club's place in its own league, as ESPN words it, by team id. */
  standing: Record<string, string>;
  /** 96x96 1-bit crests by team id (see hud/crest.ts). */
  crests: Record<string, string>;
}

// --- Reading ESPN's answers ---------------------------------------------------

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => (v && typeof v === "object" ? (v as Obj) : {});
const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const str = (v: unknown): string => (typeof v === "string" ? v.trim() : typeof v === "number" ? String(v) : "");

/** A score arrives as a number, a string, or an object holding either. */
function score(v: unknown): number | null {
  const raw = v && typeof v === "object" ? obj(v).value ?? obj(v).displayValue : v;
  if (raw === null || raw === undefined || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? Math.round(n) : null;
}

function team(v: unknown): FlaTeam | null {
  const t = obj(obj(v).team);
  const id = str(t.id);
  const name = str(t.shortDisplayName) || str(t.displayName);
  if (!id || !name) return null;
  return { id, name, abbr: str(t.abbreviation) || name.slice(0, 3) };
}

/** Matches out of a team schedule answer. Anything malformed is left out. */
export function parseSchedule(json: unknown): FlaMatch[] {
  const out: FlaMatch[] = [];
  for (const raw of list(obj(json).events)) {
    const e = obj(raw);
    const comp = obj(list(e.competitions)[0]);
    const sides = list(comp.competitors);
    const homeSide = sides.find((s) => str(obj(s).homeAway) === "home");
    const awaySide = sides.find((s) => str(obj(s).homeAway) === "away");
    const home = team(homeSide);
    const away = team(awaySide);
    const at = Date.parse(str(e.date));
    const id = str(e.id);
    if (!id || !home || !away || !Number.isFinite(at)) continue;

    const league = obj(e.league);
    const status = obj(obj(comp.status).type);
    const homeScore = score(obj(homeSide).score);
    const awayScore = score(obj(awaySide).score);
    // A postponed match is also "post"; only a completed one with a score is a result.
    const done = status.completed === true && homeScore !== null && awayScore !== null;
    const stage = league.isTournament === true ? str(obj(e.seasonType).name) : "";
    const venue = str(obj(comp.venue).fullName).replace(/\s*\(.*?\)\s*/g, " ").trim();
    out.push({
      id, at, league: str(league.name), home, away,
      homeScore: done ? homeScore : null, awayScore: done ? awayScore : null, done,
      ...(stage ? { stage } : {}), ...(venue ? { venue } : {}),
    });
  }
  return out.sort((a, b) => a.at - b.at);
}

/** The league table, top to bottom. */
export function parseTable(json: unknown): FlaRow[] {
  const entries = list(obj(obj(list(obj(json).children)[0]).standings).entries);
  const rows: FlaRow[] = [];
  for (const raw of entries) {
    const e = obj(raw);
    const t = obj(e.team);
    const stats = new Map(list(e.stats).map((s) => [str(obj(s).name), obj(s)]));
    const n = (k: string) => Number(stats.get(k)?.value ?? stats.get(k)?.displayValue);
    const id = str(t.id);
    const name = str(t.shortDisplayName) || str(t.displayName);
    const row = { pos: n("rank"), p: n("gamesPlayed"), w: n("wins"), d: n("ties"), l: n("losses"), pts: n("points") };
    if (!id || !name || Object.values(row).some((v) => !Number.isFinite(v))) continue;
    const diff = n("pointDifferential");
    const gd = Number.isFinite(diff) ? (diff > 0 ? `+${diff}` : String(diff)) : str(stats.get("pointDifferential")?.displayValue);
    rows.push({ ...row, id, name, gd });
  }
  return rows.sort((a, b) => a.pos - b.pos);
}

/** "1st in Argentine Liga Profesional de Fútbol", from a team answer. */
export function parseStanding(json: unknown): string | null {
  return str(obj(obj(json).team).standingSummary) || null;
}

/**
 * Where the table's zones change, from FotMob's legend: each entry lists the
 * row indices that share a fate (Libertadores, relegation and so on). Returns
 * the places a zone ends at, or null when the legend is missing or malformed.
 */
export function parseZones(json: unknown, clubs: number): number[] | null {
  const legend = list(obj(obj(list(json)[0]).data).legend);
  if (legend.length === 0 || clubs < 2) return null;
  const zoneOf = new Array<number>(clubs).fill(-1);
  for (let z = 0; z < legend.length; z++) {
    for (const i of list(obj(legend[z]).indices)) {
      if (typeof i !== "number" || !Number.isInteger(i) || i < 0 || i >= clubs) return null;
      zoneOf[i] = z;
    }
  }
  const ends: number[] = [];
  for (let i = 0; i < clubs - 1; i++) if (zoneOf[i] !== zoneOf[i + 1]) ends.push(i + 1);
  return ends.length ? ends : null;
}

// --- Loading ------------------------------------------------------------------

export interface FlaSources {
  /** Parsed JSON, or null when the feed could not be read. */
  json: (url: string) => Promise<unknown>;
  /** A club's crest as a packed 1-bit bitmap, or null. */
  crest: (teamId: string) => Promise<string | null>;
}

const live: FlaSources = {
  async json(url) {
    try {
      // FotMob answers 403 to a request that names no client at all.
      const res = await fetchWithTimeout(url, { headers: { Accept: "application/json", "User-Agent": USER_AGENT } }, 8000);
      if (!res.ok) console.error(`Flamengo: ${url} answered ${res.status}`);
      return res.ok ? await res.json() : null;
    } catch (err) {
      console.error(`Flamengo: ${url} failed:`, err);
      return null;
    }
  },
  async crest(teamId) {
    try {
      const res = await fetchWithTimeout(crestUrl(teamId), undefined, 8000);
      return res.ok ? await crestBits(new Uint8Array(await res.arrayBuffer())) : null;
    } catch (err) {
      console.error(`Flamengo: crest ${teamId} failed:`, err);
      return null;
    }
  },
};

/**
 * Read everything afresh. A feed that fails keeps what `previous` had for it;
 * returns null only when there is nothing at all to show. Never throws.
 */
export async function loadFlamengo(
  now: number, previous: FlaState | null, src: FlaSources = live,
): Promise<FlaState | null> {
  try {
    const [resultsJson, fixturesJson, tableJson, zonesJson] = await Promise.all([
      src.json(RESULTS_URL), src.json(FIXTURES_URL), src.json(TABLE_URL), src.json(ZONES_URL),
    ]);
    if (!resultsJson && !fixturesJson && !tableJson) return null;

    // A match can sit in either list while it is being played, so pool them.
    const byId = new Map<string, FlaMatch>();
    const pooled = [
      ...(resultsJson ? parseSchedule(resultsJson) : previous?.results ?? []),
      ...(fixturesJson ? parseSchedule(fixturesJson) : previous?.fixtures ?? []),
    ];
    for (const m of pooled) if (!byId.get(m.id)?.done) byId.set(m.id, m);
    const all = [...byId.values()].sort((a, b) => a.at - b.at);
    const results = all.filter((m) => m.done).slice(-KEPT_RESULTS);
    const fixtures = all.filter((m) => !m.done && m.at > now - MATCH_WINDOW_MS).slice(0, KEPT_FIXTURES);

    const parsedTable = tableJson ? parseTable(tableJson) : [];
    const table = parsedTable.length ? parsedTable : previous?.table ?? [];
    // Unknown zones draw no lines; they are never guessed.
    const zones = parseZones(zonesJson, table.length) ?? previous?.zones ?? [];

    // Only the two clubs of the next match need a crest and a league position.
    const next = fixtures[0];
    const standing: Record<string, string> = {};
    const crests: Record<string, string> = {};
    if (next) {
      await Promise.all([next.home, next.away].map(async (t) => {
        const crest = previous?.crests[t.id] ?? (await src.crest(t.id));
        if (crest) crests[t.id] = crest;
        if (table.some((r) => r.id === t.id)) return;
        const summary = parseStanding(await src.json(teamUrl(t.id))) ?? previous?.standing[t.id];
        if (summary) standing[t.id] = summary;
      }));
    }
    return { fetchedAt: now, table, zones, results, fixtures, standing, crests };
  } catch (err) {
    console.error("Flamengo: load failed:", err);
    return null;
  }
}

// --- Cache --------------------------------------------------------------------

async function readCached(env: Env): Promise<FlaState | null> {
  try {
    const raw = await env.CACHE.get(flamengoCacheKey());
    if (!raw) return null;
    const s = JSON.parse(raw) as FlaState;
    const ok = Number.isFinite(s?.fetchedAt) && [s.table, s.zones, s.results, s.fixtures].every(Array.isArray)
      && s.standing && s.crests;
    return ok ? s : null;
  } catch {
    return null;
  }
}

/** True when the entry should be read again: sooner around a match than on a quiet day. */
export function needsRefresh(s: FlaState, now: number): boolean {
  const playing = s.fixtures.some((m) => now >= m.at && now < m.at + MATCH_WINDOW_MS);
  return now - s.fetchedAt >= (playing ? MATCH_TTL_MS : QUIET_TTL_MS);
}

export interface FlaOptions {
  ctx?: ExecutionContext;
  now?: number;
  /** Replaces ESPN, for tests. */
  sources?: FlaSources;
  /** How long to wait when nothing is cached. The cron can afford longer than a panel. */
  budgetMs?: number;
}

/**
 * What the page should show now: the cached entry, refreshed in the background
 * when it has aged. Null only when nothing has ever been cached and ESPN did
 * not answer in time.
 */
export async function getFlamengo(env: Env, opts: FlaOptions = {}): Promise<FlaState | null> {
  const now = opts.now ?? Date.now();
  const cached = await readCached(env);
  if (cached && !needsRefresh(cached, now)) return cached;

  const refresh = async (): Promise<FlaState | null> => {
    const fresh = await loadFlamengo(now, cached, opts.sources);
    if (fresh) {
      try {
        await env.CACHE.put(flamengoCacheKey(), JSON.stringify(fresh), { expirationTtl: KV_TTL_SECONDS });
      } catch (err) {
        console.error("Flamengo: cache write failed:", err);
      }
    }
    return fresh;
  };

  if (cached && opts.ctx) {
    opts.ctx.waitUntil(refresh());
    return cached;
  }
  const pending = refresh();
  opts.ctx?.waitUntil(pending);
  return (await withBudget(pending, opts.budgetMs ?? COLD_BUDGET_MS)) ?? cached;
}

// --- What the page shows --------------------------------------------------------

export type Outcome = "W" | "D" | "L";

export interface FlaSide { id: string; name: string; abbr: string; rank: string; crest?: string }

export interface FlaView {
  /** Flamengo's league place and points, for the top bar. */
  pill: string;
  table: FlaRow[];
  /** Places with a zone line under them. */
  zones: number[];
  /** League matches played and in a season. */
  played: { done: number; of: number } | null;
  next: {
    home: FlaSide; away: FlaSide; at: number; venue: string;
    league: string; stage: string; where: "HOME" | "AWAY"; live: boolean;
  } | null;
  /** "1D 22H", "5H 12M", "45M" or "LIVE"; empty with no next match. */
  countdown: string;
  then: { opponent: string; where: "HOME" | "AWAY"; at: number } | null;
  /** `score` reads home-away, as scores are printed. */
  last: { score: string; outcome: Outcome; opponent: string; where: "HOME" | "AWAY"; league: string; at: number } | null;
  /** Up to five, oldest first. */
  form: { outcome: Outcome; opponent: string }[];
  /** Matches in a row without losing, counting back from the latest. */
  unbeaten: number;
  /** Whole days since the data was read, when that is a day or more. */
  staleDays: number;
}

const LEAGUE_NAMES: Record<string, string> = {
  "Brazilian Serie A": "SERIE A",
  "CONMEBOL Libertadores": "COPA LIBERTADORES",
  "CONMEBOL Sudamericana": "COPA SUDAMERICANA",
};

export function leagueName(name: string): string {
  return LEAGUE_NAMES[name] ?? name.toUpperCase();
}

export function ordinal(n: number): string {
  const tens = n % 100;
  const suffix = tens >= 11 && tens <= 13 ? "TH" : ["TH", "ST", "ND", "RD"][n % 10] ?? "TH";
  return `${n}${suffix}`;
}

function countdown(ms: number): string {
  const minutes = Math.max(1, Math.round(ms / 60_000));
  const d = Math.floor(minutes / 1440);
  const h = Math.floor((minutes % 1440) / 60);
  const m = minutes % 60;
  return d > 0 ? `${d}D ${h}H` : h > 0 ? `${h}H ${m}M` : `${m}M`;
}

function outcomeOf(m: FlaMatch): Outcome {
  const ours = m.home.id === FLAMENGO_ID ? m.homeScore! : m.awayScore!;
  const theirs = m.home.id === FLAMENGO_ID ? m.awayScore! : m.homeScore!;
  return ours > theirs ? "W" : ours < theirs ? "L" : "D";
}

const opponentOf = (m: FlaMatch): FlaTeam => (m.home.id === FLAMENGO_ID ? m.away : m.home);
const whereOf = (m: FlaMatch): "HOME" | "AWAY" => (m.home.id === FLAMENGO_ID ? "HOME" : "AWAY");

export function flamengoView(s: FlaState | null, now: number): FlaView {
  const table = s?.table ?? [];
  const us = table.find((r) => r.id === FLAMENGO_ID);
  const upcoming = (s?.fixtures ?? []).filter((m) => m.at > now - MATCH_WINDOW_MS);
  const results = s?.results ?? [];
  const nextMatch = upcoming[0];
  const thenMatch = upcoming[1];
  const lastMatch = results[results.length - 1];

  const side = (t: FlaTeam): FlaSide => {
    const row = table.find((r) => r.id === t.id);
    const rank = row ? `${ordinal(row.pos)} IN SERIE A` : (s?.standing[t.id] ?? "").toUpperCase();
    const crest = s?.crests[t.id];
    return { ...t, rank, ...(crest ? { crest } : {}) };
  };

  let unbeaten = 0;
  for (let i = results.length - 1; i >= 0 && outcomeOf(results[i]) !== "L"; i--) unbeaten++;

  return {
    pill: us ? `${ordinal(us.pos)}  /  ${us.pts} PTS` : "FLAMENGO",
    table,
    zones: s?.zones ?? [],
    played: us ? { done: us.p, of: Math.max(us.p, (table.length - 1) * 2) } : null,
    next: nextMatch ? {
      home: side(nextMatch.home), away: side(nextMatch.away), at: nextMatch.at, venue: nextMatch.venue ?? "",
      league: leagueName(nextMatch.league), stage: (nextMatch.stage ?? "").toUpperCase(),
      where: whereOf(nextMatch), live: now >= nextMatch.at,
    } : null,
    countdown: !nextMatch ? "" : now >= nextMatch.at ? "LIVE" : countdown(nextMatch.at - now),
    then: thenMatch ? { opponent: opponentOf(thenMatch).name, where: whereOf(thenMatch), at: thenMatch.at } : null,
    last: lastMatch ? {
      score: `${lastMatch.homeScore}-${lastMatch.awayScore}`, outcome: outcomeOf(lastMatch),
      opponent: opponentOf(lastMatch).name, where: whereOf(lastMatch), league: leagueName(lastMatch.league), at: lastMatch.at,
    } : null,
    form: results.slice(-5).map((m) => ({ outcome: outcomeOf(m), opponent: opponentOf(m).abbr })),
    unbeaten,
    staleDays: s ? Math.floor((now - s.fetchedAt) / (24 * HOUR)) : 0,
  };
}
