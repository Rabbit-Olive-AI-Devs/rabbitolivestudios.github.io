/**
 * Planet of the day for the desk panel: one real exoplanet from NASA's
 * Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu), no API key.
 *
 * The archive is asked a few small questions once a day and the answer is kept
 * in KV as `planet:v1:DATE` (one write a day, DECISIONS #62). A panel request
 * never waits on the archive when an older planet exists: it shows yesterday's
 * and refreshes in the background (the principle of DECISIONS #42 and #64).
 * With no cache and no archive it shows a built-in planet rather than fail.
 */

import type { Env } from "./types";
import { getBirthdayToday } from "./birthday";
import { planetCacheKey } from "./cache-keys";
import { shiftDateStr } from "./date-utils";
import { fetchWithTimeout } from "./fetch-timeout";
import { withBudget } from "./with-budget";

export { planetCacheKey };

export interface Planet {
  name: string;
  host: string;
  distanceLy: number | null;
  /** Radius in Earth radii. */
  radius: number;
  /** Mass in Earth masses. */
  mass: number | null;
  periodDays: number | null;
  /** Equilibrium temperature in kelvin. */
  tempK: number | null;
  year: number | null;
  method: string;
  stars: number;
  planets: number;
  starTempK: number | null;
  /** Sky position in degrees. */
  ra: number | null;
  dec: number | null;
}

export interface PlanetOfDay {
  planet: Planet;
  /** Position in the archive's list, and the list's length. `total` is 0 for a built-in planet. */
  index: number;
  total: number;
  /** Which planet out from its star this is, counting from 0; -1 when unknown. */
  slot: number;
  /** Planets discovered per year, as [year, count]. */
  years: [number, number][];
  /** A short line for the corner of the view. */
  tag?: string;
}

/** Runs one ADQL query against the archive and returns its JSON rows. */
export type ArchiveQuery = (adql: string) => Promise<unknown>;

const TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync";
const PARSEC_IN_LIGHT_YEARS = 3.26156;
const KV_TTL_SECONDS = 604800;
const ARCHIVE_TIMEOUT_MS = 8000;
/** How long a panel request may wait for the archive when nothing older can be shown. */
const COLD_BUDGET_MS = 6000;

/** Only planets with enough known about them to fill the page. */
const USABLE = "pl_rade is not null and sy_dist is not null and pl_orbper is not null";
const COLUMNS = "pl_name,hostname,sy_dist,pl_rade,pl_bmasse,pl_orbper,pl_eqt,disc_year,discoverymethod,sy_snum,sy_pnum,st_teff,ra,dec";

const NAMES_QUERY = `select pl_name from pscomppars where ${USABLE} order by pl_name`;
const YEARS_QUERY = "select disc_year,count(*) as n from pscomppars group by disc_year order by disc_year";

const quote = (s: string) => `'${s.replace(/'/g, "''")}'`;

export function planetQuery(name: string): string {
  return `select ${COLUMNS} from pscomppars where pl_name=${quote(name)}`;
}

const siblingsQuery = (host: string) =>
  `select pl_name,pl_orbsmax from pscomppars where hostname=${quote(host)} order by pl_orbsmax`;

const yearQuery = (year: number) =>
  `select top 1 pl_name from pscomppars where disc_year=${Math.trunc(year)} and ${USABLE}`;

async function queryArchive(adql: string): Promise<unknown> {
  const url = `${TAP_URL}?format=json&query=${encodeURIComponent(adql)}`;
  const res = await fetchWithTimeout(url, { headers: { "User-Agent": "eink-dashboard (personal project)" } }, ARCHIVE_TIMEOUT_MS);
  if (!res.ok) throw new Error(`Exoplanet Archive answered ${res.status}`);
  return res.json();
}

type Row = Record<string, unknown>;
const rows = (v: unknown): Row[] => (Array.isArray(v) ? v.filter((r): r is Row => typeof r === "object" && r !== null) : []);
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const str = (v: unknown): string => (typeof v === "string" ? v : "");

/** An archive row as a Planet, or null when it lacks a name or a radius. */
export function toPlanet(row: unknown): Planet | null {
  if (typeof row !== "object" || row === null) return null;
  const r = row as Row;
  const name = str(r.pl_name);
  const radius = num(r.pl_rade);
  if (!name || radius === null || radius <= 0) return null;
  const parsecs = num(r.sy_dist);
  return {
    name,
    host: str(r.hostname) || name,
    distanceLy: parsecs === null ? null : parsecs * PARSEC_IN_LIGHT_YEARS,
    radius,
    mass: num(r.pl_bmasse),
    periodDays: num(r.pl_orbper),
    tempK: num(r.pl_eqt),
    year: num(r.disc_year),
    method: str(r.discoverymethod),
    stars: Math.max(1, num(r.sy_snum) ?? 1),
    planets: Math.max(1, num(r.sy_pnum) ?? 1),
    starTempK: num(r.st_teff),
    ra: num(r.ra),
    dec: num(r.dec),
  };
}

function hash(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return h;
}

// --- Days and planets that get something extra -----------------------------------

/** A reason to show a particular planet, or a planet from a particular year, with a tag. */
export interface PlanetOccasion {
  name?: string;
  year?: number;
  tag?: string;
}

const NICKNAMES: Readonly<Record<string, string>> = {
  "Kepler-16 b": "REAL-LIFE TATOOINE",
  "OGLE-2005-BLG-390L b": "NICKNAMED HOTH",
  "Kepler-10 b": "A REAL MUSTAFAR",
  "Kepler-22 b": "A REAL KAMINO",
  "Kepler-452 b": "EARTH'S OLDER COUSIN",
  "Kepler-186 f": "A REAL ENDOR?",
  "K2-18 b": "MILLER'S PLANET?",
  "51 Peg b": "THE FIRST ONE, 1995",
  "Proxima Cen b": "OUR NEAREST NEIGHBOUR",
  "TRAPPIST-1 e": "ONE OF SEVEN SISTERS",
};

/** Fixed dates, as MM-DD. */
const DATE_PLANETS: Readonly<Record<string, string>> = {
  "01-01": "Proxima Cen b",
  "05-04": "Kepler-16 b",
  "05-25": "OGLE-2005-BLG-390L b",
  "07-20": "TRAPPIST-1 e",
  "10-06": "51 Peg b",
  "11-07": "K2-18 b",
  "11-17": "Kepler-186 f",
  "12-25": "Kepler-22 b",
};

/** The first exoplanet around a sun-like star was found in 1995; birthdays before that have no planet. */
const FIRST_DISCOVERY_YEAR = 1995;

/** What the date calls for: a birthday planet, a fixed date's planet, or nothing special. */
export function planetOccasion(dateStr: string): PlanetOccasion | null {
  const [, month, day] = dateStr.split("-").map(Number);
  const person = getBirthdayToday(month, day);
  if (person && person.birthYear >= FIRST_DISCOVERY_YEAR) {
    return { year: person.birthYear, tag: `HAPPY BIRTHDAY ${person.key.toUpperCase().slice(0, 9)}` };
  }
  const name = DATE_PLANETS[dateStr.slice(5)];
  return name ? { name } : null;
}

// --- Loading -----------------------------------------------------------------------

/**
 * Ask the archive for the date's planet. Returns null on any failure; never
 * throws. The same date always picks the same planet.
 */
export async function loadPlanet(
  dateStr: string, query: ArchiveQuery = queryArchive, occasion: PlanetOccasion | null = planetOccasion(dateStr),
): Promise<PlanetOfDay | null> {
  try {
    const names = rows(await query(NAMES_QUERY)).map((r) => str(r.pl_name)).filter(Boolean);
    if (names.length === 0) return null;

    let name = occasion?.name;
    if (!name && occasion?.year) name = str(rows(await query(yearQuery(occasion.year)))[0]?.pl_name) || undefined;
    if (!name) name = names[hash(dateStr) % names.length];

    const planet = toPlanet(rows(await query(planetQuery(name)))[0]);
    if (!planet) return null;

    const [yearRows, siblingRows] = await Promise.all([query(YEARS_QUERY), query(siblingsQuery(planet.host))]);
    const years = rows(yearRows)
      .map((r): [number | null, number | null] => [num(r.disc_year), num(r.n)])
      .filter((p): p is [number, number] => p[0] !== null && p[1] !== null);
    const squash = (s: string) => s.replace(/\s+/g, "");
    const slot = rows(siblingRows).findIndex((r) => squash(str(r.pl_name)) === squash(planet.name));

    const tag = occasion?.tag ?? NICKNAMES[planet.name];
    return { planet, index: Math.max(0, names.indexOf(planet.name)), total: names.length, slot, years, ...(tag ? { tag } : {}) };
  } catch (err) {
    console.error(`Planet: archive lookup failed for ${dateStr}:`, err);
    return null;
  }
}

/** Real planets, as the archive had them on 2026-10-12, for when it cannot be reached and nothing is cached. */
const BUILT_IN: readonly Planet[] = [
  { name: "TRAPPIST-1 e", host: "TRAPPIST-1", distanceLy: 40.54, radius: 0.92, mass: 0.692, periodDays: 6.101, tempK: 249.7, year: 2017, method: "Transit", stars: 1, planets: 7, starTempK: 2566, ra: 346.6264, dec: -5.0435 },
  { name: "51 Peg b", host: "51 Peg", distanceLy: 50.43, radius: 14.1, mass: 193.9, periodDays: 4.2308, tempK: 1328.9, year: 1995, method: "Radial Velocity", stars: 1, planets: 1, starTempK: 5761, ra: 344.3675, dec: 20.7691 },
  { name: "Kepler-16 b", host: "Kepler-16", distanceLy: 244.9, radius: 8.449, mass: 105.8, periodDays: 228.776, tempK: 205.9, year: 2011, method: "Transit", stars: 2, planets: 1, starTempK: 4450, ra: 289.0758, dec: 51.7572 },
  { name: "Proxima Cen b", host: "Proxima Cen", distanceLy: 4.244, radius: 1.02, mass: 1.055, periodDays: 11.1847, tempK: 218, year: 2016, method: "Radial Velocity", stars: 3, planets: 2, starTempK: 2900, ra: 217.3935, dec: -62.6762 },
];

function builtIn(dateStr: string): PlanetOfDay {
  const planet = BUILT_IN[hash(dateStr) % BUILT_IN.length];
  const tag = NICKNAMES[planet.name];
  return { planet, index: 0, total: 0, slot: -1, years: [], ...(tag ? { tag } : {}) };
}

async function readCached(env: Env, dateStr: string): Promise<PlanetOfDay | null> {
  try {
    const raw = await env.CACHE.get(planetCacheKey(dateStr));
    if (!raw) return null;
    const day = JSON.parse(raw) as PlanetOfDay;
    return day?.planet?.name && day.planet.radius > 0 ? day : null;
  } catch {
    return null;
  }
}

export interface PlanetOptions {
  /** Replaces the archive, for tests. */
  query?: ArchiveQuery;
  ctx?: ExecutionContext;
  /** False for a date other than today, so a visitor's ?date= cannot spend KV writes. */
  store?: boolean;
  /** How long to wait for the archive when nothing older can be shown. The cron can afford longer than a panel. */
  budgetMs?: number;
}

/** The date's planet: from cache, else from the archive, else yesterday's, else a built-in one. */
export async function getPlanetOfDay(env: Env, dateStr: string, opts: PlanetOptions = {}): Promise<PlanetOfDay> {
  const store = opts.store !== false;
  if (store) {
    const cached = await readCached(env, dateStr);
    if (cached) return cached;
  }

  const refresh = async (): Promise<PlanetOfDay | null> => {
    const day = await loadPlanet(dateStr, opts.query);
    if (day && store) {
      try {
        await env.CACHE.put(planetCacheKey(dateStr), JSON.stringify(day), { expirationTtl: KV_TTL_SECONDS });
      } catch (err) {
        console.error("Planet: cache write failed:", err);
      }
    }
    return day;
  };

  const older = store
    ? (await readCached(env, shiftDateStr(dateStr, -1))) ?? (await readCached(env, shiftDateStr(dateStr, -2)))
    : null;
  if (older && opts.ctx) {
    opts.ctx.waitUntil(refresh());
    return older;
  }
  // Nothing older to show: wait for the archive, but only so long. If the budget
  // runs out the lookup still finishes in the background and fills the cache.
  const pending = refresh();
  opts.ctx?.waitUntil(pending);
  return (await withBudget(pending, opts.budgetMs ?? COLD_BUDGET_MS)) ?? older ?? builtIn(dateStr);
}
