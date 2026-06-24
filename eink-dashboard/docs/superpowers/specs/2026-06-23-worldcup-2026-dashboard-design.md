# World Cup 2026 Dashboard — Design Spec

**Date:** 2026-06-23
**Status:** Draft for review
**Author:** Claude Code session (brainstormed with Thiago)

---

## 1. Overview & Goals

Add a seasonal **FIFA World Cup 2026** view to both e-ink displays for the duration of
the tournament (now → final on 2026-07-19). The view is a single **adaptive page per
display** that auto-transitions as the tournament moves through its phases, always showing
the most relevant thing at a glance.

Three content views, requested by the user:

1. **Today's matches + latest results** — kickoff times (Chicago local) + final/live scores.
2. **Group standings** — the 12 group tables.
3. **Knockout bracket** — the elimination tree.

These are *not* three separate pages. They are folded into one adaptive page that changes
shape by tournament phase (see §3).

### Success criteria

- A glance at the page tells you: what's on today, the latest scores, and where we are in
  the tournament (group table now, bracket later).
- Fits 800×480 with no scrolling, legible on e-ink (pure black on E1001, Spectra-6 on E1002).
- Never blocks or shows a blank panel: serves instantly from cache, degrades gracefully.
- Both phase layouts are testable at 800×480 *today*, before knockouts exist.

### Non-goals (YAGNI)

- **No true live in-match scores.** SenseCraft screenshots every ~15 min; "live" means
  "refreshed within ~15 min," nothing finer.
- **No team-focus page, no AI match illustration** (both were considered and dropped).
- **No player stats, lineups, xG, odds, or commentary.**
- **No user configuration UI.** Favorite team is a code constant.
- **No persistence beyond KV cache.** The page is stateless between requests.

---

## 2. Routes & Endpoints

| Route | Display | Description |
|-------|---------|-------------|
| `GET /worldcup` | E1001 mono | Adaptive World Cup page, pure black |
| `GET /color/worldcup` | E1002 color | Adaptive World Cup page, Spectra-6 palette |

Both are HTML pages (SenseCraft "Web Function" screenshots them), following the
`/weather` ↔ `/color/weather` pattern. No image pipeline, no dithering — the color page
uses only the 6 Spectra palette colors via CSS (same approach as `/color/weather`).

### Test parameters (cheap, no AI cost → open, no `TEST_AUTH_KEY`)

| Param | Effect |
|-------|--------|
| `?test` | Inject canned fixture data (no upstream fetch) |
| `?test-phase=group` | Force the group-stage split layout |
| `?test-phase=r32` | Force the Round-of-32 split layout |
| `?test-phase=knockout` | Force the bracket tree (R16→Final) |
| `?test-phase=champion` | Force the post-final champion card |

These let us validate every layout at 800×480 before the real data reaches that phase.

---

## 3. Adaptive Behavior (phases)

Phase is derived from the data, not the calendar (calendar is only a fallback hint):

| Phase | Condition | Layout |
|-------|-----------|--------|
| **group** | Not all 72 group-stage matches finished | Split: today/results + rotating group table |
| **r32** | Group stage done; latest active round is Round of 32 | Split: today/results + R32 results list |
| **knockout** | Latest active round ∈ {R16, QF, SF, Final} | Full-screen converging bracket tree |
| **champion** | Final finished | Champion card (final score + winner) |

"Latest active round" = the furthest round that has any scheduled / live / finished match.

### 3a. Split layout (group + r32)

```
┌────────────────────────────────────────────────────────────┐
│ FIFA WORLD CUP 2026             Tue Jun 23 · Group Stage    │  header
├──────────────────────────────────┬─────────────────────────┤
│ TODAY                            │ LATEST RESULTS           │
│ 13:00  MEX  v  KSA               │ ESP 2-1 CHI              │
│ 16:00  POL  v  ARG               │ GER 0-0 JPN              │
│ 19:00  USA  v  ITA               │ BRA 3-0 SRB   ▶          │  ▶ = favorite
├──────────────────────────────────┴─────────────────────────┤
│ GROUP D                          (cycles groups each refresh)│  bottom panel
│  1  Brazil       P2 W2 D0 L0  +5  6  ✓                       │
│  2  Switzerland  P2 W1 D1 L0  +2  4                          │
│  3  Cameroon     P2 W0 D1 L1  -2  1                          │
│  4  Serbia       P2 W0 D0 L2  -5  0                          │
└────────────────────────────────────────────────────────────┘
```

- **Top-left "TODAY":** all of today's matches (Chicago date). Each row: kickoff time
  (12h Chicago), home, away. If a match is live/finished, show the score instead of the
  time. Up to ~6 rows; if today has no matches, the panel reads "No matches today —
  next: <date>" and the results panel widens.
- **Top-right "LATEST RESULTS":** the most recent matchday's finished scores (up to ~6).
- **Bottom panel (group phase):** one group table, **cycling** through the 12 groups (see
  §7). `✓` marks teams currently in a qualifying position (top 2; third place is ambiguous
  so it is *not* marked — see §8).
- **Bottom panel (r32 phase):** replaces the group table with a "ROUND OF 32" list —
  finished R32 results + today's R32 ties — since a 16-tie tree is not legible.

### 3b. Bracket layout (R16 → Final)

```
┌────────────────────────────────────────────────────────────┐
│ FIFA WORLD CUP 2026             Sat Jul 11 · Quarter-finals │
├────────────────────────────────────────────────────────────┤
│  R16     QF      SF      FINAL      SF      QF      R16      │
│ BRA┐                                              ┌ENG      │
│    ├BRA┐                                      ┌POR┤         │
│ SUI┘   │                                      │   └POR      │
│        ├ ?  ┐                          ┌ ?  ┤             │
│ ARG┐   │    │                          │    │   ┌FRA      │
│    ├ARG┘    ├ <CHAMPION> ┤             └ ?  ┤             │
│ NED┘        │                          │        └ESP      │
│             └ ...        ...        ... ┘                   │
│  3rd place: <T1> v <T2>                                     │
└────────────────────────────────────────────────────────────┘
```

- 7 columns across 800px (~114px each): R16(L), QF(L), SF(L), Final(center), SF(R),
  QF(R), R16(R). R16 has 4 ties per side (8 total) → ~105px per tie block, legible.
- Each slot shows the 3-letter team code + score; the winner advances bold.
- Today's tie(s) highlighted (mono: bold + ▶; color: red accent). Favorite team highlighted
  throughout (see §6).
- Undecided slots render `?` / `TBD`.
- 3rd-place playoff as a one-line footer.

### 3c. Champion card (after final)

Full-screen: "FIFA WORLD CUP 2026 CHAMPIONS" + winner name/flag-code + final score + runner-up.
Stays until the page is removed from the device pagelist.

---

## 4. Data Sources & Resilience

Chosen: **football-data.org primary + openfootball static JSON fallback** (user decision).
This mirrors the existing weather resilience pattern (DECISIONS.md #42/#43).

### Primary: football-data.org

- `GET https://api.football-data.org/v4/competitions/WC/matches` — all matches with
  `stage` (GROUP_STAGE, LAST_32, LAST_16, QUARTER_FINALS, SEMI_FINALS, THIRD_PLACE, FINAL),
  `status`, `score`, `homeTeam`/`awayTeam`, `utcDate`, `group`.
- `GET https://api.football-data.org/v4/competitions/WC/standings` — the 12 group tables,
  computed server-side by football-data (points, GD, played, position) — **we do not compute
  standings ourselves**, avoiding the error-prone tiebreaker / third-place math.
- Auth: `X-Auth-Token: <env.FOOTBALL_DATA_KEY>` (Worker **secret**, never hardcoded).
- Free tier: 10 req/min — trivially within budget (we cache; ≤ a few calls per refresh).
- All fetches via `fetchWithTimeout()` (4s, matching the weather Open-Meteo timeout).

### Fallback: openfootball/worldcup.json

- `GET https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json`
  — public domain, no key, schedule + groups + scores + knockouts. Updated ~once daily
  (acceptable as a fallback only).
- Used when football-data is down, rate-limited, or (validation step 0) doesn't yet carry
  2026 data. Provides matches + results. **Standings are derived minimally** from finished
  matches when only the fallback is available (points/GD/played; qualifying `✓` suppressed
  if tiebreakers are ambiguous).

### Degradation chain (never crash, never block)

1. Fresh KV cache → serve instantly.
2. Stale KV cache → serve instantly, refresh in background (`ctx.waitUntil`, SWR).
3. Cold cache → bounded refresh via `withBudget` (reuse `src/with-budget.ts`):
   football-data → openfootball. If it exceeds the budget, serve a minimal
   "World Cup — data updating" card and fill cache in the background.
4. Total failure with empty cache → minimal "World Cup — data unavailable" card (HTTP 200,
   not a crash; page handler wrapped in try/catch → 503 with `Retry-After` only as last
   resort, matching weather handlers).

---

## 5. Data Model (normalized)

A single normalized shape, source-agnostic, cached in KV and consumed by both pages.

```ts
type WcStage = "GROUP" | "R32" | "R16" | "QF" | "SF" | "THIRD" | "FINAL";
type WcStatus = "SCHEDULED" | "LIVE" | "FINISHED";

interface WcTeam { name: string; code: string; }          // code = 3-letter (BRA, USA)
interface WcMatch {
  id: number;
  stage: WcStage;
  group?: string;                  // "A".."L" for group stage
  status: WcStatus;
  kickoffChicago: string;          // ISO or "13:00"-formatted, Chicago local
  dateChicago: string;             // YYYY-MM-DD (Chicago) for today/results bucketing
  home: WcTeam; away: WcTeam;
  homeScore?: number; awayScore?: number;
}
interface WcStandingRow {
  position: number; team: WcTeam;
  played: number; won: number; drawn: number; lost: number;
  goalDifference: number; points: number;
  qualifying?: boolean;            // top 2; omitted when ambiguous
}
interface WcGroup { name: string; rows: WcStandingRow[]; }
interface WorldCupData {
  source: "football-data" | "openfootball";
  phase: "group" | "r32" | "knockout" | "champion";
  todayMatches: WcMatch[];
  recentResults: WcMatch[];        // most recent finished matchday
  groups: WcGroup[];               // [] once group stage is over
  knockout: WcMatch[];             // R32..Final matches (for bracket / r32 list)
  champion?: WcTeam;
  generatedAt: number;             // epoch ms, for staleness
}
```

`phase` is computed in the data layer (single source of truth) so both pages just render.

---

## 6. Favorite Team Highlight

- Single constant `FAVORITE_TEAM = "BRA"` (3-letter code) in the worldcup module.
- Applied everywhere the team appears: today/results rows, group table, bracket slots.
- **Mono (E1001):** `▶` prefix marker + bold weight.
- **Color (E1002):** blue text accent or yellow background fill (yellow only as background
  per DECISIONS.md #40 — never as foreground).
- Trivial to change or disable (set to `""`).

---

## 7. Group Standings Rotation

The bottom panel shows **one** group at a time (12 groups can't fit legibly).

- Selection: prefer groups with a match **today** (in kickoff order); otherwise round-robin
  over all 12 by a 15-min time bucket: `Math.floor(chicagoEpochMin / 15) % candidates.length`.
- Advances each device refresh (~15 min), cycling all groups over ~3 hours.
- The favorite team's group is sorted to appear first in the candidate order so it shows
  most often.
- Deterministic (time-bucketed, no randomness) → cache-friendly, matches project conventions.

---

## 8. Edge Cases & Constraints

- **Third-place qualification ambiguity:** 2026 advances the 8 best third-placed teams to
  the Round of 32. That ranking is cross-group and unstable mid-stage. We mark only the
  **top 2** of each group with `✓`; third place is never auto-marked. (football-data's
  `position` is within-group, so this is honest and avoids implying a wrong outcome.)
- **No matches today:** "TODAY" panel shows next match date; results panel widens.
- **Many matches in a day:** group-stage final days have simultaneous kickoffs (up to ~6).
  Rows shrink to fit; cap at what fits 800×480, prefer favorite/earliest.
- **Phase overlap (R32 days that also still show a 'group' label):** decided strictly by
  "all 72 group matches finished," so there's a clean switch.
- **Team codes:** use FIFA 3-letter codes; map from football-data `tla` field; fallback to
  first 3 letters of name uppercased.
- **Timezone:** America/Chicago for both devices (Central). Reuse existing date-utils +
  weather-ui `formatTime` (12h).
- **Tournament over but page still deployed:** champion card persists; safe indefinitely.

---

## 9. Module Structure

Follows existing templates (`weather.ts` for cached fetch + SWR; `weather-ui.ts` for shared
render helpers; `pages/weather2.ts` + `pages/color-weather.ts` for the two displays).

| File | Purpose | Template |
|------|---------|----------|
| `src/worldcup.ts` | Data layer: `getWorldCupData(env, ctx)` — KV cache + SWR + budget + degradation chain; computes `phase` | `weather.ts` |
| `src/worldcup-football-data.ts` | football-data.org client → normalized `WorldCupData` (matches + standings) | `weather-nws.ts` |
| `src/worldcup-openfootball.ts` | openfootball JSON fallback → normalized data (+ minimal standings derivation) | `weather-nws.ts` |
| `src/worldcup-ui.ts` | Shared render helpers: match row, group table, bracket tree builder, team code, phase→layout, favorite highlight | `weather-ui.ts` |
| `src/pages/worldcup.ts` | `/worldcup` mono HTML (pure black) | `pages/weather2.ts` |
| `src/pages/color-worldcup.ts` | `/color/worldcup` Spectra-6 HTML | `pages/color-weather.ts` |
| `src/types.ts` | Add the §5 interfaces | — |
| `src/index.ts` | Route `/worldcup` + `/color/worldcup`; thread `ctx`; add WC warm to cron; bump `VERSION` | — |

The bracket tree builder is the one genuinely new piece of logic; it lives in
`worldcup-ui.ts` and is pure (data → positioned rows/columns), independently testable.

---

## 10. Caching, Cron, TTL

- **Cache keys:** `wc:data:v1` (single normalized blob serves both displays — same
  tournament data). Per-display rendering differs only at the HTML layer, so one cache entry.
- **TTL policy (per DECISIONS.md #24):** soft TTL ~10–15 min (freshness, drives SWR refresh);
  KV `expirationTtl: 86400` (24h hard, generous stale-fallback window). Hard TTL ≫ soft TTL.
- **Cron:** add one `getWorldCupData(env)` warm call to the existing every-6h
  `Promise.allSettled` block (one fetch serves both devices). Request-path SWR handles the
  finer ~15-min freshness as devices poll.
- Reuse `src/with-budget.ts` for the cold-cache request-path bound (REFRESH_BUDGET_MS ~5s),
  exactly as the recent weather work.

---

## 11. Display Styling

- **E1001 mono (`/worldcup`):** pure black `#000` only. Highlights via bold, `✓`/`▶`
  glyphs (inline SVG or ASCII-safe chars, **no emoji**), and inverse bars (black bg/white
  text) for live/today.
- **E1002 color (`/color/worldcup`):** Spectra-6 CSS variables (DECISIONS.md #40 legibility
  rules): green = win/qualified, red = live/today's tie accent, blue = favorite accent,
  yellow only as background fill. Black for normal text.
- Both: flex-column body, `overflow: hidden`, sections sized to fit 480px (reuse the weather
  flex-layout lesson, DECISIONS.md #28). Security headers via `htmlResponse()`; escape all
  dynamic text via `escapeHTML()` (team names from upstream).

---

## 12. Documentation & Versioning

Per CLAUDE.md mandatory doc sweep, update in the same commit(s):

- `README.md` — endpoints table (+ `/worldcup`, `/color/worldcup`, test params), brief
  pipeline/architecture note, version.
- `DECISIONS.md` — new decision entry (#44): World Cup adaptive page, data-source choice +
  fallback, bracket-legibility solution, third-place ambiguity, phase model.
- `MEMORY.md` — concise pointers (routes, cache key, data sources, phase model, seasonal
  removal note).
- `package.json` + `src/index.ts` `VERSION` — bump to **v3.13.0** (feature release).
  *Pending user approval per CLAUDE.md git practices.*

### Deployment / ops notes (manual, user-side)

- Set the secret: `npx wrangler secret put FOOTBALL_DATA_KEY` (interactive, not echoed).
- **Validation step 0 (before relying on primary):** with the key set, confirm
  `GET /v4/competitions/WC/matches` returns 2026 fixtures (not historical). If 2026 isn't
  covered yet, the openfootball fallback carries the feature until it is.
- Add `/worldcup` to the E1001 pagelist and `/color/worldcup` to the E1002 pagelist in
  SenseCraft HMI.
- Seasonal: remove from pagelists after 2026-07-19 (champion card is safe to leave up).

---

## 13. Definition of Done

- [ ] `/worldcup` and `/color/worldcup` render all four phases at exactly 800×480 (verified
      via `?test-phase=group|r32|knockout|champion`).
- [ ] Mono = pure black; color = Spectra-6 only, passing the legibility rules.
- [ ] Live football-data path works with the secret; openfootball fallback works with the
      key removed/forced; stale + empty-cache degradation verified.
- [ ] Favorite team highlighted in every view; togglable via constant.
- [ ] Group rotation cycles all 12 groups; bracket tree legible R16→Final; R32 list legible.
- [ ] No emoji, no JS, no grays (mono); no scrolling; flex layout holds with max-density days.
- [ ] `npm run typecheck && npm run test:utils && npm run dry-run` clean.
- [ ] Docs swept (README, DECISIONS, MEMORY, version) in the same commit.

---

## 14. Risks / Open Items

- **football-data 2026 coverage** unconfirmed without the key (validation step 0). Mitigated
  by the openfootball fallback.
- **Max-density group days** (6 simultaneous matches + results + table) are the tightest
  layout; may need row-count caps. Test with `?test-phase=group` using a packed fixture.
- **Bracket during R32** is deliberately a list, not a tree — a conscious legibility
  tradeoff, documented for DECISIONS.md.
- **Seasonal feature:** ~4-week useful life; keep the surface small and self-removing-friendly.
