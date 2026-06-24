# World Cup 2026 Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an adaptive FIFA World Cup 2026 page to both e-ink displays (`/worldcup` mono, `/color/worldcup` color) showing today's matches + results, rotating group standings, and a knockout bracket — auto-transitioning by tournament phase.

**Architecture:** A source-agnostic data layer (`worldcup.ts`) fetches from football-data.org (primary) with an openfootball static-JSON fallback, reusing the existing weather stale-while-revalidate + `withBudget` + KV degradation pattern. Pure render/logic helpers live in `worldcup-ui.ts` (unit-tested). Two thin page handlers share those helpers via a theme object, differing only in palette (pure black vs Spectra-6).

**Tech Stack:** TypeScript, Cloudflare Workers, KV cache, `node:test` (via `npm run test:utils`), Wrangler.

**Reference spec:** `docs/superpowers/specs/2026-06-23-worldcup-2026-dashboard-design.md`

**Conventions to follow (read before starting):**
- Data layer template: `src/weather.ts` (`getWeatherForLocation`) + `src/weather-nws.ts` (fallback client + pure `normalize*`).
- Shared UI helpers template: `src/weather-ui.ts`. Page templates: `src/pages/color-weather.ts`, `src/pages/weather2.ts`.
- Cache keys: `src/cache-keys.ts`. KV TTL rule (DECISIONS.md #24): soft TTL drives freshness; KV `expirationTtl` ≫ soft TTL (use 86400).
- All external fetches use `fetchWithTimeout()`. All dynamic HTML escaped via `escapeHTML()`. E-ink: 800×480, pure black on mono, Spectra-6 only on color, no emoji, no JS, no scrolling (flex column, DECISIONS.md #28).
- Tests import **compiled** JS from the build dir (see `tests/utils.test.js` header). Keep tested logic in exported pure functions.

**Validation step 0 (do once, before Task 5 relies on live data):** With the secret set (`npx wrangler secret put FOOTBALL_DATA_KEY`), run `curl -H "X-Auth-Token: $KEY" https://api.football-data.org/v4/competitions/WC/matches | head` and confirm it returns 2026 fixtures. If 2026 isn't covered yet, the openfootball fallback (Task 6) carries the feature; note it in the PR.

---

## File Structure

| File | Create/Modify | Responsibility |
|------|---------------|----------------|
| `src/types.ts` | Modify | Add `FOOTBALL_DATA_KEY?` to `Env`; add World Cup interfaces |
| `src/cache-keys.ts` | Modify | Add `worldCupCacheKey()` |
| `src/worldcup-ui.ts` | Create | Pure logic + HTML section builders (phase, team code, kickoff fmt, group rotation, bracket builder, render dispatch) |
| `src/worldcup-football-data.ts` | Create | football-data.org client + pure normalize → `WorldCupData` |
| `src/worldcup-openfootball.ts` | Create | openfootball JSON fallback + pure normalize → partial `WorldCupData` |
| `src/worldcup-testdata.ts` | Create | Canned fixtures for `?test` / `?test-phase` |
| `src/worldcup.ts` | Create | Data layer: KV cache + SWR + budget + degradation chain; computes `phase` |
| `src/pages/worldcup.ts` | Create | `/worldcup` mono handler (pure-black theme) |
| `src/pages/color-worldcup.ts` | Create | `/color/worldcup` Spectra-6 handler |
| `src/index.ts` | Modify | Routes, cron warm, VERSION bump, 404 list |
| `tests/worldcup.test.js` | Create | Unit tests for pure helpers + cache key + normalizers |
| `package.json` | Modify | Version bump |
| `README.md`, `DECISIONS.md`, `MEMORY.md` | Modify | Doc sweep |

---

## Task 1: Types, Env, and cache key (foundation)

**Files:**
- Modify: `src/types.ts`
- Modify: `src/cache-keys.ts`
- Test: `tests/worldcup.test.js`

- [ ] **Step 1: Add the World Cup types and Env field**

In `src/types.ts`, add `FOOTBALL_DATA_KEY?: string;` to the `Env` interface (after `TEST_AUTH_KEY?`), and append this block at the end of the file:

```ts
// --- World Cup 2026 types ---

export type WcStage = "GROUP" | "R32" | "R16" | "QF" | "SF" | "THIRD" | "FINAL";
export type WcStatus = "SCHEDULED" | "LIVE" | "FINISHED";
export type WcPhase = "group" | "r32" | "knockout" | "champion";

export interface WcTeam {
  name: string;
  code: string; // 3-letter (BRA, USA); "TBD" when undecided
}

export interface WcMatch {
  id: number;
  stage: WcStage;
  group?: string;            // "A".."L" for group stage
  status: WcStatus;
  kickoffISO: string;        // UTC ISO from source
  dateChicago: string;       // YYYY-MM-DD in America/Chicago
  timeChicago: string;       // "1:00 PM" in America/Chicago
  home: WcTeam;
  away: WcTeam;
  homeScore: number | null;
  awayScore: number | null;
}

export interface WcStandingRow {
  position: number;
  team: WcTeam;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goalDifference: number;
  points: number;
  qualifying: boolean;       // top 2 only; third place never auto-marked
}

export interface WcGroup {
  name: string;              // "A".."L"
  rows: WcStandingRow[];
}

export interface WorldCupData {
  source: "football-data" | "openfootball";
  phase: WcPhase;
  todayMatches: WcMatch[];
  recentResults: WcMatch[];  // most recent finished matchday
  groups: WcGroup[];         // [] once group stage is over
  knockout: WcMatch[];       // R32..Final matches
  champion: WcTeam | null;
  generatedAt: number;       // epoch ms
}
```

- [ ] **Step 2: Add the cache key**

In `src/cache-keys.ts`, add a version constant near the others and a helper:

```ts
export const WORLDCUP_CACHE_VERSION = "v1";

export function worldCupCacheKey(): string {
  return `wc:data:${WORLDCUP_CACHE_VERSION}`;
}
```

- [ ] **Step 3: Write the failing test for the cache key**

Create `tests/worldcup.test.js`:

```js
const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { worldCupCacheKey } = fromBuild("src/cache-keys.js");

test("worldCupCacheKey is stable", () => {
  assert.equal(worldCupCacheKey(), "wc:data:v1");
});
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npm run test:utils`
Expected: PASS (includes the new `worldcup.test.js`). If it fails to compile, fix types first.

- [ ] **Step 5: Commit**

```bash
git add src/types.ts src/cache-keys.ts tests/worldcup.test.js
git commit -m "Add World Cup types, Env key, and cache key"
```

---

## Task 2: Phase detection, team code, and match-cell helpers

**Files:**
- Create: `src/worldcup-ui.ts`
- Test: `tests/worldcup.test.js`

- [ ] **Step 1: Write the failing tests**

Append to `tests/worldcup.test.js`:

```js
const {
  computePhase,
  teamCode,
  matchCell,
  STAGE_LABELS,
} = fromBuild("src/worldcup-ui.js");

const gm = (status) => ({ stage: "GROUP", status });
const km = (stage, status) => ({ stage, status });

test("computePhase: group while any group match unfinished", () => {
  assert.equal(computePhase([gm("FINISHED"), gm("SCHEDULED")]), "group");
});

test("computePhase: r32 when groups done and latest active round is R32", () => {
  const matches = [gm("FINISHED"), km("R32", "SCHEDULED")];
  assert.equal(computePhase(matches), "r32");
});

test("computePhase: knockout from R16 onward", () => {
  const matches = [gm("FINISHED"), km("R32", "FINISHED"), km("R16", "SCHEDULED")];
  assert.equal(computePhase(matches), "knockout");
});

test("computePhase: champion when final finished", () => {
  const matches = [gm("FINISHED"), km("FINAL", "FINISHED")];
  assert.equal(computePhase(matches), "champion");
});

test("teamCode prefers code, falls back to first 3 letters", () => {
  assert.equal(teamCode({ name: "Brazil", code: "BRA" }), "BRA");
  assert.equal(teamCode({ name: "Brazil", code: "" }), "BRA");
  assert.equal(teamCode({ name: "", code: "" }), "TBD");
});

test("matchCell shows score when finished, time otherwise", () => {
  assert.equal(matchCell({ status: "FINISHED", homeScore: 2, awayScore: 1, timeChicago: "1 PM" }), "2-1");
  assert.equal(matchCell({ status: "LIVE", homeScore: 0, awayScore: 0, timeChicago: "1 PM" }), "0-0");
  assert.equal(matchCell({ status: "SCHEDULED", homeScore: null, awayScore: null, timeChicago: "1 PM" }), "1 PM");
});

test("STAGE_LABELS covers all knockout stages", () => {
  for (const s of ["GROUP", "R32", "R16", "QF", "SF", "THIRD", "FINAL"]) {
    assert.ok(typeof STAGE_LABELS[s] === "string" && STAGE_LABELS[s].length > 0);
  }
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npm run test:utils`
Expected: FAIL — cannot find module `src/worldcup-ui.js`.

- [ ] **Step 3: Create `src/worldcup-ui.ts` with the pure helpers**

```ts
/**
 * World Cup 2026 — pure logic + HTML section builders.
 *
 * Shared by pages/worldcup.ts (mono) and pages/color-worldcup.ts (color).
 * Logic functions are exported and unit-tested; render functions take a
 * `WcTheme` so the two displays differ only in palette.
 */

import type {
  WcMatch, WcStage, WcStatus, WcPhase, WcTeam, WcGroup, WorldCupData,
} from "./types";

export const STAGE_LABELS: Record<WcStage, string> = {
  GROUP: "Group Stage",
  R32: "Round of 32",
  R16: "Round of 16",
  QF: "Quarter-finals",
  SF: "Semi-finals",
  THIRD: "Third Place",
  FINAL: "Final",
};

// Knockout rounds ordered earliest → latest, for "latest active round" logic.
const KO_ORDER: WcStage[] = ["R32", "R16", "QF", "SF", "FINAL"];

/** 3-letter code, falling back to the first 3 letters of the name, then "TBD". */
export function teamCode(team: { name: string; code: string }): string {
  if (team.code && team.code.trim()) return team.code.trim().toUpperCase().slice(0, 3);
  if (team.name && team.name.trim()) return team.name.trim().toUpperCase().slice(0, 3);
  return "TBD";
}

/** Score ("2-1") when finished/live, else the Chicago kickoff time. */
export function matchCell(m: { status: WcStatus; homeScore: number | null; awayScore: number | null; timeChicago: string }): string {
  if ((m.status === "FINISHED" || m.status === "LIVE") && m.homeScore !== null && m.awayScore !== null) {
    return `${m.homeScore}-${m.awayScore}`;
  }
  return m.timeChicago;
}

/**
 * Derive the tournament phase from the full match list.
 * - champion: the FINAL is finished
 * - group: any GROUP match is not yet FINISHED
 * - else knockout sub-phase by the latest round with any non-scheduled-only activity:
 *   r32 if the furthest active round is R32, otherwise knockout (R16+).
 */
export function computePhase(matches: { stage: WcStage; status: WcStatus }[]): WcPhase {
  if (matches.some((m) => m.stage === "FINAL" && m.status === "FINISHED")) return "champion";

  const groupMatches = matches.filter((m) => m.stage === "GROUP");
  const groupUnfinished = groupMatches.some((m) => m.status !== "FINISHED");
  if (groupMatches.length === 0 || groupUnfinished) return "group";

  // Group stage complete → find the furthest knockout round that has any match.
  let furthest = -1;
  for (const m of matches) {
    const idx = KO_ORDER.indexOf(m.stage);
    if (idx > furthest) furthest = idx;
  }
  // furthest === 0 means only R32 exists so far.
  return furthest <= 0 ? "r32" : "knockout";
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npm run test:utils`
Expected: PASS for all Task 2 tests.

- [ ] **Step 5: Commit**

```bash
git add src/worldcup-ui.ts tests/worldcup.test.js
git commit -m "Add World Cup phase/team/match-cell pure helpers"
```

---

## Task 3: Group rotation picker

**Files:**
- Modify: `src/worldcup-ui.ts`
- Test: `tests/worldcup.test.js`

- [ ] **Step 1: Write the failing test**

Append to `tests/worldcup.test.js`:

```js
const { pickRotatingGroup } = fromBuild("src/worldcup-ui.js");

const mkGroup = (name) => ({ name, rows: [] });

test("pickRotatingGroup prefers a group with a match today", () => {
  const groups = [mkGroup("A"), mkGroup("B"), mkGroup("C")];
  const todayMatches = [{ group: "C", home: { code: "X" }, away: { code: "Y" } }];
  const g = pickRotatingGroup(groups, todayMatches, "ZZZ", 0);
  assert.equal(g.name, "C");
});

test("pickRotatingGroup round-robins by time bucket when no match today", () => {
  const groups = [mkGroup("A"), mkGroup("B"), mkGroup("C")];
  assert.equal(pickRotatingGroup(groups, [], "ZZZ", 0).name, "A");
  assert.equal(pickRotatingGroup(groups, [], "ZZZ", 15).name, "B");
  assert.equal(pickRotatingGroup(groups, [], "ZZZ", 30).name, "C");
  assert.equal(pickRotatingGroup(groups, [], "ZZZ", 45).name, "A");
});

test("pickRotatingGroup returns null for empty groups", () => {
  assert.equal(pickRotatingGroup([], [], "BRA", 0), null);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npm run test:utils`
Expected: FAIL — `pickRotatingGroup is not a function`.

- [ ] **Step 3: Implement `pickRotatingGroup` in `src/worldcup-ui.ts`**

Append:

```ts
/**
 * Choose one group to show in the bottom panel.
 * Candidate order: groups with a match today first (in array order), then the
 * rest. Within the candidate list, advance by a 15-minute time bucket so the
 * panel cycles as the device refreshes. Deterministic (no randomness).
 *
 * @param epochMinChicago minutes-since-epoch in Chicago (or any monotonic min counter)
 */
export function pickRotatingGroup(
  groups: WcGroup[],
  todayMatches: { group?: string }[],
  _favCode: string,
  epochMinChicago: number,
): WcGroup | null {
  if (groups.length === 0) return null;
  const todayGroupNames = new Set(
    todayMatches.map((m) => m.group).filter((g): g is string => !!g),
  );
  const withToday = groups.filter((g) => todayGroupNames.has(g.name));
  const rest = groups.filter((g) => !todayGroupNames.has(g.name));
  const candidates = withToday.length > 0 ? withToday : rest.length > 0 ? rest : groups;
  const bucket = Math.floor(epochMinChicago / 15);
  const idx = ((bucket % candidates.length) + candidates.length) % candidates.length;
  return candidates[idx];
}
```

> Note: `_favCode` is reserved for future favorite-group biasing; kept in the signature so callers are stable. (If a linter forbids unused params, prefix retains intent.)

- [ ] **Step 4: Run to verify pass**

Run: `npm run test:utils`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worldcup-ui.ts tests/worldcup.test.js
git commit -m "Add World Cup group rotation picker"
```

---

## Task 4: Bracket column builder

**Files:**
- Modify: `src/worldcup-ui.ts`
- Test: `tests/worldcup.test.js`

- [ ] **Step 1: Write the failing test**

Append to `tests/worldcup.test.js`:

```js
const { buildBracket } = fromBuild("src/worldcup-ui.js");

const ko = (stage, h, a, hs = null, as = null, status = "SCHEDULED") => ({
  stage, status, home: { name: h, code: h }, away: { name: a, code: a },
  homeScore: hs, awayScore: as,
});

test("buildBracket groups knockout matches into ordered rounds", () => {
  const matches = [
    ko("R16", "BRA", "SUI", 2, 0, "FINISHED"),
    ko("R16", "ARG", "NED"),
    ko("QF", "BRA", "ARG"),
    ko("FINAL", "BRA", "ENG"),
    ko("SF", "BRA", "FRA"),
  ];
  const b = buildBracket(matches);
  // Rounds present, ordered R16 -> QF -> SF -> FINAL
  assert.deepEqual(b.rounds.map((r) => r.stage), ["R16", "QF", "SF", "FINAL"]);
  assert.equal(b.rounds[0].matches.length, 2);
  assert.equal(b.rounds[3].stage, "FINAL");
});

test("buildBracket excludes R32 and THIRD from the tree", () => {
  const matches = [ko("R32", "BRA", "SUI"), ko("THIRD", "X", "Y"), ko("R16", "BRA", "ARG")];
  const b = buildBracket(matches);
  assert.deepEqual(b.rounds.map((r) => r.stage), ["R16"]);
  assert.equal(b.third && b.third.stage, "THIRD");
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npm run test:utils`
Expected: FAIL — `buildBracket is not a function`.

- [ ] **Step 3: Implement `buildBracket` in `src/worldcup-ui.ts`**

Append:

```ts
export interface WcBracketRound {
  stage: WcStage;
  label: string;
  matches: WcMatch[];
}
export interface WcBracket {
  rounds: WcBracketRound[];   // R16 -> QF -> SF -> FINAL (only those present)
  third: WcMatch | null;      // third-place playoff, shown as a footer
}

const TREE_ORDER: WcStage[] = ["R16", "QF", "SF", "FINAL"];

/** Group knockout matches into the legible R16→Final tree; R32 + third excluded from the tree. */
export function buildBracket(knockout: WcMatch[]): WcBracket {
  const rounds: WcBracketRound[] = [];
  for (const stage of TREE_ORDER) {
    const matches = knockout.filter((m) => m.stage === stage);
    if (matches.length > 0) {
      rounds.push({ stage, label: STAGE_LABELS[stage], matches });
    }
  }
  const third = knockout.find((m) => m.stage === "THIRD") ?? null;
  return { rounds, third };
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npm run test:utils`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worldcup-ui.ts tests/worldcup.test.js
git commit -m "Add World Cup bracket column builder"
```

---

## Task 5: football-data.org client + normalizer

**Files:**
- Create: `src/worldcup-football-data.ts`
- Test: `tests/worldcup.test.js`

- [ ] **Step 1: Write the failing test for the pure normalizer**

Append to `tests/worldcup.test.js`:

```js
const { normalizeFootballData, mapStage, mapStatus } = fromBuild("src/worldcup-football-data.js");

test("mapStage/mapStatus translate football-data enums", () => {
  assert.equal(mapStage("GROUP_STAGE"), "GROUP");
  assert.equal(mapStage("LAST_16"), "R16");
  assert.equal(mapStage("QUARTER_FINALS"), "QF");
  assert.equal(mapStage("FINAL"), "FINAL");
  assert.equal(mapStatus("IN_PLAY"), "LIVE");
  assert.equal(mapStatus("PAUSED"), "LIVE");
  assert.equal(mapStatus("FINISHED"), "FINISHED");
  assert.equal(mapStatus("TIMED"), "SCHEDULED");
});

test("normalizeFootballData maps matches + standings to WorldCupData", () => {
  const matchesJson = {
    matches: [
      {
        id: 1, stage: "GROUP_STAGE", group: "Group A", status: "FINISHED",
        utcDate: "2026-06-23T18:00:00Z",
        homeTeam: { name: "Brazil", tla: "BRA" }, awayTeam: { name: "Serbia", tla: "SRB" },
        score: { fullTime: { home: 3, away: 0 } },
      },
    ],
  };
  const standingsJson = {
    standings: [
      {
        group: "Group A",
        table: [
          { position: 1, team: { name: "Brazil", tla: "BRA" }, playedGames: 1, won: 1, draw: 0, lost: 0, goalDifference: 3, points: 3 },
          { position: 4, team: { name: "Serbia", tla: "SRB" }, playedGames: 1, won: 0, draw: 0, lost: 1, goalDifference: -3, points: 0 },
        ],
      },
    ],
  };
  const data = normalizeFootballData(matchesJson, standingsJson);
  assert.equal(data.source, "football-data");
  assert.equal(data.knockout.length, 0);
  assert.equal(data.groups.length, 1);
  assert.equal(data.groups[0].name, "A");
  assert.equal(data.groups[0].rows[0].qualifying, true);   // position 1 -> top 2
  assert.equal(data.groups[0].rows[1].qualifying, false);  // position 4
  // recentResults includes the finished match
  assert.equal(data.recentResults.length, 1);
  assert.equal(data.recentResults[0].home.code, "BRA");
  assert.equal(data.recentResults[0].homeScore, 3);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npm run test:utils`
Expected: FAIL — cannot find `src/worldcup-football-data.js`.

- [ ] **Step 3: Implement `src/worldcup-football-data.ts`**

```ts
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
    homeScore: m.score?.fullTime?.home ?? null,
    awayScore: m.score?.fullTime?.away ?? null,
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
    const latestDate = finished.map((m) => m.dateChicago).sort().at(-1)!;
    recentResults = finished.filter((m) => m.dateChicago === latestDate);
  }

  const champion = matches.find((m) => m.stage === "FINAL" && m.status === "FINISHED");
  const championTeam: WcTeam | null = champion
    ? (champion.homeScore! > champion.awayScore! ? champion.home : champion.away)
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
    ...( { _allMatches: matches } as any ),
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
    // Standings can 404 mid-tournament edge cases; tolerate it.
    const standingsJson: any = sres.ok ? await sres.json() : { standings: [] };
    return normalizeFootballData(matchesJson, standingsJson);
  } catch (err) {
    console.error("football-data fetch failed:", err);
    return null;
  }
}
```

> The `_allMatches` carry-along lets the data layer compute `phase` + `todayMatches` from the full list without re-fetching. The data layer reads it then deletes it before caching (Task 7).

- [ ] **Step 4: Add the Chicago date/time helpers used above to `src/worldcup-ui.ts`**

Append to `src/worldcup-ui.ts`:

```ts
/** YYYY-MM-DD for a UTC ISO timestamp, in America/Chicago. */
export function chicagoDateOf(iso: string): string {
  const fmt = new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit",
  });
  return fmt.format(new Date(iso)); // en-CA yields YYYY-MM-DD
}

/** "1:00 PM" for a UTC ISO timestamp, in America/Chicago. */
export function chicagoTimeOf(iso: string): string {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago", hour: "numeric", minute: "2-digit", hour12: true,
  });
  return fmt.format(new Date(iso));
}
```

- [ ] **Step 5: Add tests for the Chicago helpers**

Append to `tests/worldcup.test.js`:

```js
const { chicagoDateOf, chicagoTimeOf } = fromBuild("src/worldcup-ui.js");

test("chicagoDateOf/chicagoTimeOf convert UTC to Chicago", () => {
  // 2026-06-23T18:00Z is 1:00 PM CDT (UTC-5)
  assert.equal(chicagoDateOf("2026-06-23T18:00:00Z"), "2026-06-23");
  assert.equal(chicagoTimeOf("2026-06-23T18:00:00Z"), "1:00 PM");
  // 2026-06-24T02:00Z is 9:00 PM CDT the previous day
  assert.equal(chicagoDateOf("2026-06-24T02:00:00Z"), "2026-06-23");
  assert.equal(chicagoTimeOf("2026-06-24T02:00:00Z"), "9:00 PM");
});
```

- [ ] **Step 6: Run to verify pass**

Run: `npm run test:utils`
Expected: PASS for all Task 5 tests.

- [ ] **Step 7: Commit**

```bash
git add src/worldcup-football-data.ts src/worldcup-ui.ts tests/worldcup.test.js
git commit -m "Add football-data.org World Cup client + normalizer"
```

---

## Task 6: openfootball fallback client + normalizer

**Files:**
- Create: `src/worldcup-openfootball.ts`
- Test: `tests/worldcup.test.js`

- [ ] **Step 1: Write the failing test**

Append to `tests/worldcup.test.js`:

```js
const { normalizeOpenFootball } = fromBuild("src/worldcup-openfootball.js");

test("normalizeOpenFootball maps rounds/matches and derives simple standings", () => {
  const json = {
    rounds: [
      {
        name: "Matchday 1",
        matches: [
          {
            date: "2026-06-23", time: "18:00",
            team1: "Brazil", team2: "Serbia", group: "Group A",
            score: { ft: [3, 0] },
          },
          {
            date: "2026-06-23", time: "21:00",
            team1: "Switzerland", team2: "Cameroon", group: "Group A",
          },
        ],
      },
    ],
  };
  const data = normalizeOpenFootball(json);
  assert.equal(data.source, "openfootball");
  // finished match becomes a result
  assert.equal(data.recentResults.length, 1);
  assert.equal(data.recentResults[0].homeScore, 3);
  // derived standings: Brazil leads Group A
  const groupA = data.groups.find((g) => g.name === "A");
  assert.ok(groupA);
  assert.equal(groupA.rows[0].team.code, "BRA");
  assert.equal(groupA.rows[0].points, 3);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npm run test:utils`
Expected: FAIL — cannot find `src/worldcup-openfootball.js`.

- [ ] **Step 3: Implement `src/worldcup-openfootball.ts`**

```ts
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
    const h = ensure(m.group, m.home);
    const a = ensure(m.group, m.away);
    h.played++; a.played++;
    h.goalDifference += m.homeScore! - m.awayScore!;
    a.goalDifference += m.awayScore! - m.homeScore!;
    if (m.homeScore! > m.awayScore!) { h.won++; h.points += 3; a.lost++; }
    else if (m.homeScore! < m.awayScore!) { a.won++; a.points += 3; h.lost++; }
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
    const latestDate = finished.map((m) => m.dateChicago).sort().at(-1)!;
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
    ...( { _allMatches: matches } as any ),
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
```

- [ ] **Step 4: Run to verify pass**

Run: `npm run test:utils`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/worldcup-openfootball.ts tests/worldcup.test.js
git commit -m "Add openfootball World Cup fallback client + derived standings"
```

---

## Task 7: Data layer (cache + SWR + budget + degradation + phase)

**Files:**
- Create: `src/worldcup.ts`

> This mirrors `getWeatherForLocation` (read it first). No new unit test — it's I/O orchestration; verified end-to-end in Task 12. Keep the pure pieces (already tested) doing the logic.

- [ ] **Step 1: Implement `src/worldcup.ts`**

```ts
/**
 * World Cup 2026 data layer.
 *
 * Stale-while-revalidate over KV (mirrors weather.ts):
 *  - fresh cache (<SOFT_TTL): return it
 *  - stale cache: return immediately, refresh in background (ctx.waitUntil)
 *  - cold cache: bounded refresh (withBudget) on the request path
 *
 * Refresh chain: football-data.org (primary) -> openfootball (fallback).
 * Phase + todayMatches + generatedAt are computed here from the full match list.
 */

import { withBudget } from "./with-budget";
import { worldCupCacheKey } from "./cache-keys";
import { fetchFootballData } from "./worldcup-football-data";
import { fetchOpenFootball } from "./worldcup-openfootball";
import { computePhase, chicagoDateOf } from "./worldcup-ui";
import { getChicagoDateISO } from "./date-utils";
import type { Env, WorldCupData, WcMatch, CachedValue } from "./types";

const SOFT_TTL_MS = 12 * 60 * 1000;   // 12 min freshness (drives SWR)
const HARD_TTL_S = 86400;             // 24h KV survival for stale fallback (DECISIONS.md #24)
const REFRESH_BUDGET_MS = 5000;       // cold-path wall-clock bound (SenseCraft renderer)

/** Finalize a normalized blob: compute phase, todayMatches, generatedAt; strip carry-along. */
function finalize(data: WorldCupData): WorldCupData {
  const all: WcMatch[] = (data as any)._allMatches ?? [];
  const today = getChicagoDateISO();
  const phase = computePhase(all);
  const todayMatches = all
    .filter((m) => m.dateChicago === today)
    .sort((a, b) => a.kickoffISO.localeCompare(b.kickoffISO));
  const out: WorldCupData = {
    ...data,
    phase,
    todayMatches,
    generatedAt: Date.now(),
  };
  delete (out as any)._allMatches;
  return out;
}

async function refresh(env: Env, cacheKey: string): Promise<WorldCupData | null> {
  let data = await fetchFootballData(env);
  if (!data) {
    console.warn("WorldCup: football-data unavailable, trying openfootball");
    data = await fetchOpenFootball(env);
  }
  if (!data) {
    console.error("WorldCup: all sources failed");
    return null;
  }
  const finalized = finalize(data);
  await env.CACHE.put(
    cacheKey,
    JSON.stringify({ data: finalized, timestamp: Date.now() }),
    { expirationTtl: HARD_TTL_S },
  );
  console.log(`WorldCup: refreshed from ${finalized.source} (phase=${finalized.phase})`);
  return finalized;
}

/** Get World Cup data with stale-while-revalidate + degradation. */
export async function getWorldCupData(env: Env, ctx?: ExecutionContext): Promise<WorldCupData> {
  const cacheKey = worldCupCacheKey();
  const cached = await env.CACHE.get<CachedValue<WorldCupData>>(cacheKey, "json");
  const fresh = cached && Date.now() - cached.timestamp < SOFT_TTL_MS;

  if (cached && fresh) {
    console.log("WorldCup: cache hit");
    return cached.data;
  }

  const doRefresh = () => refresh(env, cacheKey);

  if (cached) {
    console.log("WorldCup: serving stale, revalidating");
    if (ctx) {
      ctx.waitUntil(doRefresh().catch((e) => console.error("WorldCup bg refresh:", e)));
    } else {
      const r = await doRefresh();
      if (r) return r;
    }
    return cached.data;
  }

  // Cold start: bound the wait on the request path.
  const refreshPromise = doRefresh();
  let result: WorldCupData | null;
  if (ctx) {
    result = await withBudget(refreshPromise, REFRESH_BUDGET_MS);
    if (result === null) {
      console.warn(`WorldCup: cold refresh exceeded ${REFRESH_BUDGET_MS}ms budget`);
      ctx.waitUntil(refreshPromise.catch(() => {}));
    }
  } else {
    result = await refreshPromise;
  }
  if (result) return result;
  if (cached) return cached!.data;
  throw new Error("WorldCup: no data from any source and no cache");
}
```

- [ ] **Step 2: Typecheck**

Run: `npm run typecheck`
Expected: PASS (no type errors). If `CachedValue<WorldCupData>` complains about `source`, recall `source` is optional — we don't set it; that's fine.

- [ ] **Step 3: Commit**

```bash
git add src/worldcup.ts
git commit -m "Add World Cup data layer with SWR + football-data/openfootball degradation"
```

---

## Task 8: Canned test fixtures

**Files:**
- Create: `src/worldcup-testdata.ts`

> Lets `?test` / `?test-phase` render every layout at 800×480 before the real data reaches that phase. No upstream calls.

- [ ] **Step 1: Implement `src/worldcup-testdata.ts`**

```ts
/**
 * Canned WorldCupData fixtures for ?test / ?test-phase visual testing.
 * Cheap (no upstream) so these routes need no auth (like ?test-device).
 */

import type { WorldCupData, WcMatch, WcGroup, WcTeam, WcPhase, WcStage, WcStatus } from "./types";

const T = (name: string, code: string): WcTeam => ({ name, code });

function m(
  id: number, stage: WcStage, status: WcStatus, time: string,
  home: WcTeam, away: WcTeam, hs: number | null, as: number | null,
  group?: string, dateChicago = "2026-06-23",
): WcMatch {
  return {
    id, stage, group, status,
    kickoffISO: `${dateChicago}T${id.toString().padStart(2, "0")}:00:00Z`,
    dateChicago, timeChicago: time,
    home, away, homeScore: hs, awayScore: as,
  };
}

const BRA = T("Brazil", "BRA"), SRB = T("Serbia", "SRB"), SUI = T("Switzerland", "SUI"), CMR = T("Cameroon", "CMR");
const ARG = T("Argentina", "ARG"), NED = T("Netherlands", "NED"), ENG = T("England", "ENG"), FRA = T("France", "FRA");
const ESP = T("Spain", "ESP"), POR = T("Portugal", "POR"), GER = T("Germany", "GER"), USA = T("USA", "USA");

function groupA(): WcGroup {
  return {
    name: "A",
    rows: [
      { position: 1, team: BRA, played: 2, won: 2, drawn: 0, lost: 0, goalDifference: 5, points: 6, qualifying: true },
      { position: 2, team: SUI, played: 2, won: 1, drawn: 1, lost: 0, goalDifference: 2, points: 4, qualifying: true },
      { position: 3, team: CMR, played: 2, won: 0, drawn: 1, lost: 1, goalDifference: -2, points: 1, qualifying: false },
      { position: 4, team: SRB, played: 2, won: 0, drawn: 0, lost: 2, goalDifference: -5, points: 0, qualifying: false },
    ],
  };
}

export function testWorldCupData(phase: WcPhase): WorldCupData {
  const base: WorldCupData = {
    source: "football-data", phase,
    todayMatches: [], recentResults: [], groups: [], knockout: [], champion: null,
    generatedAt: Date.now(),
  };

  if (phase === "group" || phase === "r32") {
    const today = [
      m(13, "GROUP", "SCHEDULED", "1:00 PM", USA, SRB, null, null, "C"),
      m(16, "GROUP", "SCHEDULED", "4:00 PM", ARG, NED, null, null, "B"),
      m(19, "GROUP", "LIVE", "7:00 PM", BRA, SUI, 1, 0, "A"),
    ];
    const results = [
      m(20, "GROUP", "FINISHED", "—", ESP, GER, 2, 1, "D", "2026-06-22"),
      m(21, "GROUP", "FINISHED", "—", FRA, POR, 0, 0, "E", "2026-06-22"),
      m(22, "GROUP", "FINISHED", "—", ENG, USA, 3, 0, "C", "2026-06-22"),
    ];
    const knockout = phase === "r32"
      ? [m(30, "R32", "FINISHED", "—", BRA, ESP, 2, 1), m(31, "R32", "SCHEDULED", "3:00 PM", ARG, POR, null, null)]
      : [];
    return { ...base, todayMatches: today, recentResults: results, groups: phase === "r32" ? [] : [groupA()], knockout, phase };
  }

  if (phase === "knockout") {
    const r16 = [
      m(40, "R16", "FINISHED", "—", BRA, SUI, 2, 0), m(41, "R16", "FINISHED", "—", ARG, NED, 1, 0),
      m(42, "R16", "SCHEDULED", "1:00 PM", ENG, FRA, null, null), m(43, "R16", "SCHEDULED", "4:00 PM", ESP, POR, null, null),
    ];
    const qf = [m(50, "QF", "SCHEDULED", "—", BRA, ARG, null, null)];
    return { ...base, knockout: [...r16, ...qf], todayMatches: [r16[2], r16[3]], phase };
  }

  // champion
  const final = m(60, "FINAL", "FINISHED", "—", BRA, ENG, 2, 1);
  return { ...base, knockout: [final], champion: BRA, phase };
}
```

- [ ] **Step 2: Typecheck**

Run: `npm run typecheck`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/worldcup-testdata.ts
git commit -m "Add canned World Cup test fixtures for all phases"
```

---

## Task 9: Shared render builders (theme-driven HTML)

**Files:**
- Modify: `src/worldcup-ui.ts`

> Both pages share these builders; the page files (Tasks 10–11) only supply a `WcTheme` + page CSS. This keeps mono/color DRY. After implementing, the page tasks let you visually verify at 800×480.

- [ ] **Step 1: Add the theme type and render functions to `src/worldcup-ui.ts`**

Append:

```ts
import { escapeHTML } from "./escape";

export interface WcTheme {
  rootCSS: string;   // extra :root vars (spectra6CSS() for color, "" for mono)
  fav: string;       // favorite-team accent color (CSS color)
  win: string;       // win/qualified accent
  live: string;      // live / today accent
}

export const FAVORITE_TEAM = "BRA"; // 3-letter code; "" disables the highlight

const isFav = (code: string) => FAVORITE_TEAM !== "" && code === FAVORITE_TEAM;

/** One match row: "TIME/SCORE  HOME v AWAY", with favorite + live accents. */
function matchRow(mm: WcMatch, theme: WcTheme): string {
  const h = teamCode(mm.home), a = teamCode(mm.away);
  const cell = matchCell(mm);
  const cellColor = mm.status === "LIVE" ? theme.live : "#000";
  const favMark = (c: string) => (isFav(c) ? ` style="color:${theme.fav};font-weight:800"` : "");
  const star = (isFav(h) || isFav(a)) ? ` <span style="color:${theme.fav}">&#9654;</span>` : "";
  return `<div class="wc-row">
    <span class="wc-cell" style="color:${cellColor}">${escapeHTML(cell)}</span>
    <span${favMark(h)}>${escapeHTML(h)}</span> <span class="wc-v">v</span> <span${favMark(a)}>${escapeHTML(a)}</span>${star}
  </div>`;
}

function groupTable(group: WcGroup, theme: WcTheme): string {
  const rows = group.rows.map((r) => {
    const code = teamCode(r.team);
    const fav = isFav(code);
    const mark = r.qualifying ? ` <span style="color:${theme.win}">&#10003;</span>` : "";
    const nameStyle = fav ? ` style="color:${theme.fav};font-weight:800"` : "";
    return `<tr>
      <td class="wc-pos">${r.position}</td>
      <td${nameStyle}>${escapeHTML(code)}${mark}</td>
      <td>${r.played}</td><td>${r.won}</td><td>${r.drawn}</td><td>${r.lost}</td>
      <td>${r.goalDifference >= 0 ? "+" : ""}${r.goalDifference}</td>
      <td class="wc-pts">${r.points}</td>
    </tr>`;
  }).join("");
  return `<div class="wc-group">
    <div class="wc-group-name">GROUP ${escapeHTML(group.name)}</div>
    <table class="wc-table">
      <thead><tr><th></th><th></th><th>P</th><th>W</th><th>D</th><th>L</th><th>GD</th><th>Pts</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}

function bracketColumn(round: WcBracketRound, theme: WcTheme): string {
  const ties = round.matches.map((mm) => {
    const h = teamCode(mm.home), a = teamCode(mm.away);
    const finished = mm.status === "FINISHED" && mm.homeScore !== null && mm.awayScore !== null;
    const homeWon = finished && mm.homeScore! > mm.awayScore!;
    const awayWon = finished && mm.awayScore! > mm.homeScore!;
    const side = (code: string, won: boolean) => {
      const fav = isFav(code) ? `color:${theme.fav};` : "";
      const w = won ? "font-weight:800;" : "";
      return `<div class="wc-bteam" style="${fav}${w}">${escapeHTML(code)}</div>`;
    };
    const sc = finished ? `<div class="wc-bscore">${mm.homeScore}-${mm.awayScore}</div>` : "";
    const liveCls = mm.status === "LIVE" ? ` style="border-color:${theme.live}"` : "";
    return `<div class="wc-tie"${liveCls}>${side(h, homeWon)}${side(a, awayWon)}${sc}</div>`;
  }).join("");
  return `<div class="wc-bcol"><div class="wc-bcol-label">${escapeHTML(round.label)}</div>${ties}</div>`;
}

function header(data: WorldCupData): string {
  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "America/Chicago" });
  const phaseLabel = data.phase === "group" ? "Group Stage"
    : data.phase === "r32" ? "Round of 32"
    : data.phase === "champion" ? "Champions" : "Knockouts";
  return `<div class="wc-header">
    <div class="wc-title">FIFA WORLD CUP 2026</div>
    <div class="wc-sub">${escapeHTML(dateStr)} &middot; ${escapeHTML(phaseLabel)}</div>
  </div>`;
}

/** Split layout (group + r32): today + results on top, group table / R32 list below. */
function splitLayout(data: WorldCupData, theme: WcTheme): string {
  const today = data.todayMatches.length > 0
    ? data.todayMatches.map((mm) => matchRow(mm, theme)).join("")
    : `<div class="wc-empty">No matches today</div>`;
  const results = data.recentResults.length > 0
    ? data.recentResults.map((mm) => matchRow(mm, theme)).join("")
    : `<div class="wc-empty">—</div>`;

  let bottom = "";
  if (data.phase === "r32") {
    const list = data.knockout.filter((m) => m.stage === "R32").map((mm) => matchRow(mm, theme)).join("");
    bottom = `<div class="wc-bottom"><div class="wc-panel-label">ROUND OF 32</div><div class="wc-r32">${list || `<div class="wc-empty">—</div>`}</div></div>`;
  } else {
    const epochMin = Math.floor(Date.now() / 60000);
    const group = pickRotatingGroup(data.groups, data.todayMatches, FAVORITE_TEAM, epochMin);
    bottom = `<div class="wc-bottom">${group ? groupTable(group, theme) : `<div class="wc-empty">Standings unavailable</div>`}</div>`;
  }

  return `${header(data)}
  <div class="wc-split">
    <div class="wc-col"><div class="wc-panel-label">TODAY</div>${today}</div>
    <div class="wc-col"><div class="wc-panel-label">LATEST RESULTS</div>${results}</div>
  </div>
  ${bottom}`;
}

/** Bracket layout (R16 → Final). */
function bracketLayout(data: WorldCupData, theme: WcTheme): string {
  const b = buildBracket(data.knockout);
  const cols = b.rounds.map((r) => bracketColumn(r, theme)).join("");
  const third = b.third
    ? `<div class="wc-third">3rd place: ${escapeHTML(teamCode(b.third.home))} ${b.third.status === "FINISHED" ? `${b.third.homeScore}-${b.third.awayScore}` : "v"} ${escapeHTML(teamCode(b.third.away))}</div>`
    : "";
  return `${header(data)}
  <div class="wc-bracket">${cols || `<div class="wc-empty">Bracket not available yet</div>`}</div>
  ${third}`;
}

function championLayout(data: WorldCupData, theme: WcTheme): string {
  const final = data.knockout.find((m) => m.stage === "FINAL");
  const champ = data.champion ? teamCode(data.champion) : "—";
  const score = final ? `${final.homeScore}-${final.awayScore}` : "";
  const matchup = final ? `${teamCode(final.home)} ${score} ${teamCode(final.away)}` : "";
  return `${header(data)}
  <div class="wc-champion">
    <div class="wc-champ-label">CHAMPIONS</div>
    <div class="wc-champ-team" style="color:${theme.win}">${escapeHTML(champ)}</div>
    <div class="wc-champ-final">${escapeHTML(matchup)}</div>
  </div>`;
}

/** Build the full HTML document for a given theme + page CSS. */
export function renderWorldCupHTML(data: WorldCupData, theme: WcTheme, pageCSS: string): string {
  let body: string;
  switch (data.phase) {
    case "knockout": body = bracketLayout(data, theme); break;
    case "champion": body = championLayout(data, theme); break;
    default: body = splitLayout(data, theme); break; // group + r32
  }
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>World Cup 2026</title>
<style>
  :root { ${theme.rootCSS} }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { width: 800px; height: 480px; overflow: hidden; display: flex; flex-direction: column;
    background: #fff; color: #000; font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; padding: 14px 22px; }
  .wc-header { display: flex; justify-content: space-between; align-items: baseline; border-bottom: 3px solid #000; padding-bottom: 6px; margin-bottom: 8px; }
  .wc-title { font-size: 24px; font-weight: 800; letter-spacing: 1px; }
  .wc-sub { font-size: 17px; font-weight: 600; }
  .wc-panel-label { font-size: 13px; font-weight: 800; letter-spacing: 1.5px; margin-bottom: 4px; }
  .wc-split { display: flex; gap: 24px; margin-bottom: 10px; }
  .wc-col { flex: 1; }
  .wc-row { font-size: 19px; font-weight: 600; padding: 3px 0; border-bottom: 1px solid #ccc; display: flex; align-items: center; gap: 8px; }
  .wc-cell { display: inline-block; min-width: 64px; font-weight: 800; }
  .wc-v { font-size: 14px; font-weight: 500; }
  .wc-bottom { flex: 1; min-height: 0; overflow: hidden; border-top: 3px solid #000; padding-top: 6px; }
  .wc-group-name { font-size: 15px; font-weight: 800; letter-spacing: 1px; margin-bottom: 2px; }
  .wc-table { width: 100%; border-collapse: collapse; font-size: 18px; }
  .wc-table th { font-size: 12px; font-weight: 700; text-align: center; padding: 1px 4px; }
  .wc-table td { text-align: center; padding: 2px 4px; font-weight: 600; }
  .wc-table td:nth-child(2) { text-align: left; font-weight: 700; }
  .wc-pos { color: #555; } .wc-pts { font-weight: 800; }
  .wc-r32 { columns: 2; font-size: 18px; }
  .wc-bracket { flex: 1; display: flex; gap: 4px; min-height: 0; overflow: hidden; align-items: stretch; }
  .wc-bcol { flex: 1; display: flex; flex-direction: column; justify-content: space-around; }
  .wc-bcol-label { font-size: 11px; font-weight: 800; text-align: center; letter-spacing: 1px; margin-bottom: 4px; }
  .wc-tie { border: 2px solid #000; border-radius: 4px; padding: 2px 4px; margin: 3px 0; text-align: center; }
  .wc-bteam { font-size: 16px; font-weight: 600; }
  .wc-bscore { font-size: 13px; font-weight: 800; }
  .wc-third { font-size: 15px; font-weight: 700; text-align: center; margin-top: 6px; }
  .wc-champion { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; }
  .wc-champ-label { font-size: 20px; font-weight: 700; letter-spacing: 4px; }
  .wc-champ-team { font-size: 96px; font-weight: 800; }
  .wc-champ-final { font-size: 28px; font-weight: 700; }
  .wc-empty { font-size: 16px; color: #555; padding: 6px 0; }
  ${pageCSS}
</style>
</head>
<body>${body}</body>
</html>`;
}
```

- [ ] **Step 2: Typecheck**

Run: `npm run typecheck`
Expected: PASS. (Remove the dead `liveBorder` line if the compiler flags an always-empty expression — replace its use in `wc-tie` with `""`.)

- [ ] **Step 3: Commit**

```bash
git add src/worldcup-ui.ts
git commit -m "Add theme-driven World Cup HTML render builders"
```

---

## Task 10: Mono + color page handlers

**Files:**
- Create: `src/pages/worldcup.ts`
- Create: `src/pages/color-worldcup.ts`

- [ ] **Step 1: Implement `src/pages/worldcup.ts` (E1001 mono)**

```ts
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
```

- [ ] **Step 2: Implement `src/pages/color-worldcup.ts` (E1002 color)**

```ts
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
```

- [ ] **Step 3: Typecheck**

Run: `npm run typecheck`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/pages/worldcup.ts src/pages/color-worldcup.ts
git commit -m "Add World Cup mono + color page handlers"
```

---

## Task 11: Wire routes, cron, and version

**Files:**
- Modify: `src/index.ts`
- Modify: `package.json`

- [ ] **Step 1: Add imports near the other page imports in `src/index.ts`**

After the `handleColorWeatherPage` import line, add:

```ts
import { handleWorldCupPage } from "./pages/worldcup";
import { handleColorWorldCupPage } from "./pages/color-worldcup";
import { getWorldCupData } from "./worldcup";
```

- [ ] **Step 2: Bump `VERSION`**

Change `const VERSION = "3.12.0";` to `const VERSION = "3.13.0";`.

- [ ] **Step 3: Add the routes**

In the `switch (path)` block, after the `case "/color/weather":` handler, add:

```ts
      case "/worldcup":
        return handleWorldCupPage(env, url, ctx);
      case "/color/worldcup":
        return handleColorWorldCupPage(env, url, ctx);
```

- [ ] **Step 4: Add to the 404 endpoints list**

In the `default:` response `endpoints` array, add `"/worldcup", "/color/worldcup",` to the list (after the color entries).

- [ ] **Step 5: Warm the World Cup cache in cron**

In `handleScheduled`, add `getWorldCupData(env)` to the every-6h `Promise.allSettled([...])` array (last element) and add `"worldcup"` to the matching `labels` array:

```ts
    const sixHourResults = await Promise.allSettled([
      getHeadlines(env, dateStr, getCurrentPeriod()),
      getWeather(env),
      getWeatherForLocation(env, 41.8781, -87.6298, "60606", "Chicago, IL"),
      fetchDeviceData(env, E1001_DEVICE_ID),
      fetchDeviceData(env, E1002_DEVICE_ID),
      getWorldCupData(env),
    ]);
    const labels = ["headlines", "weather-60540", "weather-60606", "device-E1001", "device-E1002", "worldcup"] as const;
```

- [ ] **Step 6: Bump `package.json` version**

Change `"version": "3.12.0",` to `"version": "3.13.0",`.

- [ ] **Step 7: Typecheck + dry-run**

Run: `npm run typecheck && npm run dry-run`
Expected: both PASS (build succeeds).

- [ ] **Step 8: Commit**

```bash
git add src/index.ts package.json
git commit -m "Wire World Cup routes + cron warm; bump to v3.13.0"
```

---

## Task 12: End-to-end verification (local dev + visual at 800×480)

**Files:** none (verification only)

- [ ] **Step 1: Start the dev server**

Run: `lsof -ti:8790` (ensure free), then `npx wrangler dev --port 8790` in the background.

- [ ] **Step 2: Verify all phases render at 800×480 (mono)**

In a browser at exactly 800×480 viewport, load each and confirm no scrolling/clipping, pure black, no emoji:
- `http://localhost:8790/worldcup?test-phase=group`
- `http://localhost:8790/worldcup?test-phase=r32`
- `http://localhost:8790/worldcup?test-phase=knockout`
- `http://localhost:8790/worldcup?test-phase=champion`

Adjust CSS spacing in `renderWorldCupHTML` (Task 9) if anything overflows 480px. Re-test after edits.

- [ ] **Step 3: Verify all phases render (color)**

Same four URLs under `/color/worldcup`. Confirm only Spectra-6 colors appear: green `✓`/champion, red live score, blue favorite (BRA). Confirm no yellow foreground text.

- [ ] **Step 4: Verify live data path (if `FOOTBALL_DATA_KEY` is set locally)**

Set the key for local dev (e.g. `.dev.vars` with `FOOTBALL_DATA_KEY=...`, gitignored — never commit it), restart dev, load `http://localhost:8790/worldcup` (no params). Confirm real fixtures/standings render. Check `wrangler tail`/console for `WorldCup: refreshed from football-data`.

- [ ] **Step 5: Verify fallback + degradation**

Temporarily unset the key (rename `.dev.vars`), restart, load `/worldcup`. Confirm console shows `football-data unavailable, trying openfootball` and the page still renders from openfootball.

- [ ] **Step 6: Full test suite + build**

Run: `npm run typecheck && npm run test:utils && npm run dry-run`
Expected: all PASS.

- [ ] **Step 7: Stop the dev server**

Kill the background `wrangler dev` process. (CLAUDE.md: never leave dev servers running.)

- [ ] **Step 8: Commit any CSS refinements from Steps 2–3**

```bash
git add src/worldcup-ui.ts
git commit -m "Refine World Cup layout spacing for 800x480"
```

---

## Task 13: Documentation sweep

**Files:**
- Modify: `README.md`, `DECISIONS.md`, `MEMORY.md`

- [ ] **Step 1: README — endpoints + pipeline note**

Add `/worldcup` and `/color/worldcup` rows to the Endpoints table (with the `?test-phase=group|r32|knockout|champion` test param noted). Add a short "World Cup 2026" subsection describing the adaptive phases + data sources (football-data primary, openfootball fallback). Update the version reference if present.

- [ ] **Step 2: DECISIONS.md — new entry #44**

Add `## 44. World Cup 2026 Adaptive Dashboard` covering: the single adaptive page per display + phase model; data-source choice (football-data primary + openfootball fallback, reusing the weather SWR/budget pattern); the bracket-legibility decision (R32 as a list, R16→Final as a tree); third-place `✓` suppression; favorite-team constant; seasonal/removable nature.

- [ ] **Step 3: MEMORY.md — concise pointers**

Add a `NEXT SESSION` / project line: routes `/worldcup` + `/color/worldcup`, cache key `wc:data:v1`, sources football-data+openfootball, phase model, `FAVORITE_TEAM` constant, needs `FOOTBALL_DATA_KEY` secret, seasonal (remove from pagelists after 2026-07-19). Keep MEMORY.md within its line budget (it's already near the limit — move detail into a topic file if needed).

- [ ] **Step 4: Verify the doc sweep**

Confirm: README endpoints table accurate; DECISIONS #44 present; MEMORY pointer present; `package.json` + `src/index.ts` VERSION both `3.13.0`.

- [ ] **Step 5: Commit**

```bash
git add README.md DECISIONS.md MEMORY.md
git commit -m "Document World Cup 2026 dashboard (v3.13.0)"
```

---

## Post-implementation (manual / user-side, not code)

- `npx wrangler secret put FOOTBALL_DATA_KEY` (interactive; never echoed/committed).
- Run validation step 0 (top of plan) to confirm football-data carries 2026 data.
- `npx wrangler deploy` (only on user approval).
- Add `/worldcup` to the E1001 pagelist and `/color/worldcup` to the E1002 pagelist in SenseCraft HMI.
- After 2026-07-19, remove from pagelists (champion card is safe to leave up otherwise).

---

## Self-Review Notes

- **Spec coverage:** today+results+rotating standings (Task 9 splitLayout + Task 3), bracket (Tasks 4, 9), adaptive phases (Task 2 computePhase + Task 9 dispatch), data sources + degradation (Tasks 5–7), favorite team (Task 9 `FAVORITE_TEAM`), test params (Tasks 8, 10), caching/cron/TTL (Tasks 1, 7, 11), mono+color (Tasks 9–10), docs+version (Tasks 11, 13). All spec sections map to a task.
- **Type consistency:** `WorldCupData`/`WcMatch`/`WcGroup`/`WcTheme` names used identically across tasks; `matchCell`, `teamCode`, `computePhase`, `buildBracket`, `pickRotatingGroup`, `chicagoDateOf/TimeOf`, `renderWorldCupHTML`, `getWorldCupData` signatures match their definitions and call sites.
- **Known executor watch-points:** (1) the `_allMatches` carry-along is intentional and stripped in `finalize()` before caching; (2) the dead `liveBorder` expression in `bracketColumn` should be simplified to `""` if the compiler complains; (3) `.at(-1)` requires the ES2022 lib — the project already uses modern TS (`tsc` 5.7), but if `.at` errors, replace with `[arr.length - 1]`.
