/**
 * World Cup 2026 — pure logic + HTML section builders.
 *
 * Shared by pages/worldcup.ts (mono) and pages/color-worldcup.ts (color).
 * Logic functions are exported and unit-tested; render functions take a
 * `WcTheme` so the two displays differ only in palette.
 */

import { escapeHTML } from "./escape";
import type {
  WcMatch, WcStage, WcStatus, WcPhase, WcGroup, WcTeam, WorldCupData,
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

/** Curated short names for teams whose full name overflows the panels. Keyed by FIFA code. */
const NAME_OVERRIDES: Record<string, string> = {
  RSA: "S. Africa",
  KOR: "S. Korea",
  BIH: "Bosnia",
  CPV: "Cape Verde",
  USA: "USA",
};

/** Display name: curated override when present, else the source name. */
export function displayName(team: { name: string; code: string }): string {
  const code = (team.code ?? "").trim().toUpperCase();
  if (NAME_OVERRIDES[code]) return NAME_OVERRIDES[code];
  return (team.name ?? "").trim();
}

/** Full team name, escaped, word-aware-truncated to maxChars with an ellipsis. "TBD" if empty. */
export function teamLabel(team: { name: string; code: string }, maxChars: number): string {
  const name = displayName(team);
  if (!name) return "TBD";
  let shown = name;
  if (name.length > maxChars) {
    // Trim a trailing space/hyphen left by the cut so we don't get "Bosnia-…".
    shown = name.slice(0, Math.max(1, maxChars - 1)).replace(/[\s\-–]+$/, "") + "…";
  }
  return escapeHTML(shown);
}

/**
 * Per-row "mathematically guaranteed a top-2 group finish" flags. Fixture-aware:
 * brute-forces every win/draw/loss outcome of the group's remaining matches and marks
 * a team only if it finishes top-2 in ALL of them. This is the only correct way — the
 * points-only heuristic counts rivals independently and so misses the common case where
 * two chasers play EACH OTHER (only one can win), e.g. USA in Group D (DECISIONS #47).
 *
 * Conservative on tiebreakers: a team is "safe" only if at most one OTHER team can have
 * points >= its points in a given scenario (so a points tie never counts as safe, since
 * GD/head-to-head could rank the tied rival above). Third-place best-of qualification is
 * never marked (cross-group / unstable — see DECISIONS #44).
 *
 * @param rows     standings rows (need team identity + current points + position)
 * @param remaining non-finished group matches, as { home, away } team objects
 */
export function qualifiedFlags(
  rows: { team: { name: string; code: string }; position: number; points: number }[],
  remaining: { home: { name: string; code: string }; away: { name: string; code: string } }[],
): boolean[] {
  // Group complete: positions are final (tiebreakers already resolved by the source).
  if (remaining.length === 0) return rows.map((r) => r.position <= 2);

  // Resolve a match team to its standings row by NAME first, then code. football-data is
  // internally inconsistent (e.g. Curaçao is CUW in standings but CUR in matches), so keying
  // on the 3-letter code alone silently drops fixtures; the full name is stable across feeds.
  const norm = (s: string) => (s ?? "").trim().toLowerCase();
  const byName = new Map<string, number>();
  const byCode = new Map<string, number>();
  rows.forEach((r, i) => {
    if (r.team.name) byName.set(norm(r.team.name), i);
    if (r.team.code) byCode.set(norm(r.team.code), i);
  });
  const resolve = (t: { name: string; code: string }): number | undefined => {
    const n = byName.get(norm(t.name));
    return n !== undefined ? n : byCode.get(norm(t.code));
  };

  const base = rows.map((r) => r.points);
  const safe = rows.map(() => true);
  const total = 3 ** remaining.length;     // group of 4 -> at most 3^6 = 729 scenarios

  for (let scenario = 0; scenario < total; scenario++) {
    const pts = base.slice();
    let s = scenario;
    for (const g of remaining) {
      const outcome = s % 3;
      s = (s - outcome) / 3;
      const h = resolve(g.home);
      const a = resolve(g.away);
      if (h === undefined || a === undefined) continue;     // match references a team outside this group
      if (outcome === 0) pts[h] += 3;                       // home win
      else if (outcome === 1) { pts[h] += 1; pts[a] += 1; } // draw
      else pts[a] += 3;                                     // away win
    }
    for (let i = 0; i < rows.length; i++) {
      let atOrAbove = 0;
      for (let j = 0; j < rows.length; j++) {
        if (j !== i && pts[j] >= pts[i]) atOrAbove++;
      }
      if (atOrAbove >= 2) safe[i] = false;                  // could be pushed out of top 2 in this scenario
    }
  }
  return safe;
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
 * - else knockout sub-phase by the CURRENT active round (earliest knockout round with
 *   an unfinished match): r32 if R32 is still active, otherwise knockout (R16+).
 *
 * The source publishes all 104 fixtures up front, so R16/QF/SF/FINAL exist as scheduled
 * TBD placeholders from the start. "A later round exists" therefore does NOT mean it is
 * active — keying on the furthest *existing* round wrongly jumps straight to the (all-TBD)
 * R16 bracket the moment the group stage ends. Use the earliest round that still has an
 * unfinished (SCHEDULED/LIVE) match: that is the round actually being played.
 */
export function computePhase(matches: { stage: WcStage; status: WcStatus }[]): WcPhase {
  if (matches.some((m) => m.stage === "FINAL" && m.status === "FINISHED")) return "champion";

  const groupMatches = matches.filter((m) => m.stage === "GROUP");
  const groupUnfinished = groupMatches.some((m) => m.status !== "FINISHED");
  if (groupMatches.length === 0 || groupUnfinished) return "group";

  // Group stage complete → current round = earliest knockout round with an unfinished match.
  let current = -1;
  for (let i = 0; i < KO_ORDER.length; i++) {
    if (matches.some((m) => m.stage === KO_ORDER[i] && m.status !== "FINISHED")) {
      current = i;
      break;
    }
  }
  // No unfinished knockout match left (FINAL-finished already returned champion above) or no
  // knockout fixtures yet → fall back to the R32 split layout, which degrades gracefully.
  if (current < 0) return "r32";
  return current === 0 ? "r32" : "knockout";
}

/**
 * Choose one group to show in the bottom panel.
 * Candidate order: groups with a match today first (in array order), then the
 * rest. Within the candidate list, advance by a 15-minute time bucket so the
 * panel cycles as the device refreshes. Deterministic (no randomness).
 *
 * @param epochMinChicago minutes-since-epoch (or any monotonic minute counter)
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

// --- Rendering ---

export interface WcTheme {
  rootCSS: string;   // extra :root vars (spectra6CSS() for color, "" for mono)
  styleCSS: string;  // the FULL per-display stylesheet (reset + body + .wc-* classes) — separate per display
  fav: string;       // favorite-team accent color (CSS color)
  win: string;       // win/qualified accent
  live: string;      // live / today accent
  flag?: (code: string) => string;  // FIFA code -> <img> flag chip (color page only)
}

export const FAVORITE_TEAM: string = "BRA"; // 3-letter code; "" disables the highlight

const isFav = (code: string) => FAVORITE_TEAM !== "" && code === FAVORITE_TEAM;

/** Flag chip HTML for a FIFA code, or "" when the theme has no flags (mono). */
const flagImg = (theme: WcTheme, code: string) => (theme.flag ? theme.flag(code) : "");

/** One match row: "TIME/SCORE  HOME v AWAY", with favorite + live accents. */
function matchRow(mm: WcMatch, theme: WcTheme): string {
  const h = teamCode(mm.home), a = teamCode(mm.away);
  const cell = matchCell(mm);
  const cellColor = mm.status === "LIVE" ? theme.live : "#000";
  const favMark = (c: string) => (isFav(c) ? ` style="color:${theme.fav};font-weight:800"` : "");
  const star = (isFav(h) || isFav(a)) ? ` <span style="color:${theme.fav}">&#9654;</span>` : "";
  const side = (team: WcMatch["home"], code: string) =>
    `<span class="wc-team"${favMark(code)}>${flagImg(theme, code)}${teamLabel(team, 14)}</span>`;
  return `<div class="wc-row">
    <span class="wc-cell" style="color:${cellColor}">${escapeHTML(cell)}</span>
    ${side(mm.home, h)} <span class="wc-v">v</span> ${side(mm.away, a)}${star}
  </div>`;
}

function groupTable(group: WcGroup, theme: WcTheme): string {
  const rows = group.rows.map((r) => {
    const code = teamCode(r.team);
    const fav = isFav(code);
    // r.qualifying is the fixture-aware "guaranteed top-2" flag computed in finalize().
    const mark = r.qualifying ? ` <span style="color:${theme.win}">&#10003;</span>` : "";
    const nameStyle = fav ? ` style="color:${theme.fav};font-weight:800"` : "";
    return `<tr>
      <td class="wc-pos">${r.position}</td>
      <td${nameStyle}>${flagImg(theme, code)}${teamLabel(r.team, 12)}${mark}</td>
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
    const side = (team: WcMatch["home"], code: string, won: boolean) => {
      const fav = isFav(code) ? `color:${theme.fav};` : "";
      const w = won ? "font-weight:800;" : "";
      return `<div class="wc-bteam" style="${fav}${w}">${flagImg(theme, code)}${teamLabel(team, 12)}</div>`;
    };
    const sc = finished ? `<div class="wc-bscore">${mm.homeScore}-${mm.awayScore}</div>` : "";
    const liveCls = mm.status === "LIVE" ? ` style="border-color:${theme.live}"` : "";
    return `<div class="wc-tie"${liveCls}>${side(mm.home, h, homeWon)}${side(mm.away, a, awayWon)}${sc}</div>`;
  }).join("");
  return `<div class="wc-bcol"><div class="wc-bcol-label">${escapeHTML(round.label)}</div>${ties}</div>`;
}

function header(data: WorldCupData, subtitleOverride?: string): string {
  const now = new Date();
  const dateStr = now.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric", timeZone: "America/Chicago" });
  const phaseLabel = data.phase === "group" ? "Group Stage"
    : data.phase === "r32" ? "Round of 32"
    : data.phase === "champion" ? "Champions" : "Knockouts";
  const sub = subtitleOverride ?? `${dateStr} · ${phaseLabel}`;
  return `<div class="wc-header">
    <div class="wc-title">FIFA WORLD CUP 2026</div>
    <div class="wc-sub">${escapeHTML(sub)}</div>
  </div>`;
}

const SHORT_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** "2026-06-29" -> "Jun 29" (the date is already Chicago-local). */
function shortChicagoDate(dateChicago: string): string {
  const [, m, d] = dateChicago.split("-").map((n) => parseInt(n, 10));
  if (!m || !d) return "";
  return `${SHORT_MONTHS[m - 1]} ${d}`;
}

/** Group-stage layout: today + results on top, a rotating group table below. */
function splitLayout(data: WorldCupData, theme: WcTheme): string {
  const today = data.todayMatches.length > 0
    ? data.todayMatches.map((mm) => matchRow(mm, theme)).join("")
    : `<div class="wc-empty">No matches today</div>`;
  const results = data.recentResults.length > 0
    ? data.recentResults.map((mm) => matchRow(mm, theme)).join("")
    : `<div class="wc-empty">—</div>`;

  const epochMin = Math.floor(Date.now() / 60000);
  const group = pickRotatingGroup(data.groups, data.todayMatches, FAVORITE_TEAM, epochMin);
  const bottom = `<div class="wc-bottom">${group ? groupTable(group, theme) : `<div class="wc-empty">Standings unavailable</div>`}</div>`;

  return `${header(data)}
  <div class="wc-split">
    <div class="wc-col"><div class="wc-panel-label">TODAY</div><div class="wc-rowlist">${today}</div></div>
    <div class="wc-col"><div class="wc-panel-label">LATEST RESULTS</div><div class="wc-rowlist">${results}</div></div>
  </div>
  ${bottom}`;
}

const teamKnown = (t: { name: string; code: string } | undefined): boolean =>
  !!(t && ((t.name && t.name.trim()) || (t.code && t.code.trim())));

const EMPTY_TEAM: WcTeam = { name: "", code: "" };

/** The winning team of a finished knockout match, or null (not finished / level on our fullTime score). */
export function winnerTeam(m: WcMatch | undefined): WcTeam | null {
  if (m && m.status === "FINISHED" && m.homeScore !== null && m.awayScore !== null) {
    if (m.homeScore > m.awayScore) return m.home;
    if (m.awayScore > m.homeScore) return m.away;
  }
  return null;
}

/**
 * Build the next knockout round (bracket order) from the previous round's winners.
 * The data source publishes later-round matches with EMPTY teams until it seeds them, so we
 * advance winners ourselves: slot i is fed by prev[2i] (home) and prev[2i+1] (away). When the
 * source HAS already seeded a match's teams (i.e. once it's been played/drawn), we prefer it
 * (it carries the real teams + score). Dates/ids come from the source match at that position.
 */
export function advanceRound(prev: WcMatch[], source: WcMatch[]): WcMatch[] {
  const out: WcMatch[] = [];
  const n = Math.floor(prev.length / 2);
  for (let i = 0; i < n; i++) {
    const src = source[i];
    if (src && (teamKnown(src.home) || teamKnown(src.away))) { out.push(src); continue; }
    out.push({
      id: src?.id ?? -(1000 + i),
      stage: src?.stage ?? "R16",
      group: undefined,
      status: src?.status ?? "SCHEDULED",
      kickoffISO: src?.kickoffISO ?? "",
      dateChicago: src?.dateChicago ?? "",
      timeChicago: src?.timeChicago ?? "",
      home: winnerTeam(prev[2 * i]) ?? EMPTY_TEAM,
      away: winnerTeam(prev[2 * i + 1]) ?? EMPTY_TEAM,
      homeScore: null,
      awayScore: null,
    });
  }
  return out;
}

/**
 * One bracket box. If neither team is decided yet, it shows just the round date as a faint
 * placeholder, leaving room for the eventual teams. Once a team is known it renders two team
 * lines, the favorite accent, a per-team score + bold winner on finished ties, otherwise a
 * date·time line. `compact` (inner rounds, where boxes are narrow) shows the 3-letter code
 * instead of the full name (the color theme still draws the flag); R32 uses full names.
 */
function bracketBox(mm: WcMatch, theme: WcTheme, compact = false, extraClass = ""): string {
  const cls = `wc-ktie${extraClass ? ` ${extraClass}` : ""}`;
  if (!teamKnown(mm.home) && !teamKnown(mm.away)) {
    const d = shortChicagoDate(mm.dateChicago);
    return `<div class="${cls} wc-kempty">${d ? `<span class="wc-ktbd">${escapeHTML(d)}</span>` : ""}</div>`;
  }
  const h = teamCode(mm.home), a = teamCode(mm.away);
  const finished = mm.status === "FINISHED" && mm.homeScore !== null && mm.awayScore !== null;
  const live = mm.status === "LIVE";
  const homeWon = finished && mm.homeScore! > mm.awayScore!;
  const awayWon = finished && mm.awayScore! > mm.homeScore!;
  // Compact (inner rounds, narrow boxes): color shows the flag only (no text — full names and
  // even 3-letter codes overflow), mono shows the 3-letter code. R32 (wide) shows full names.
  const label = (team: WcMatch["home"], code: string) => {
    if (!compact) return teamLabel(team, 13);
    if (theme.flag) return "";                                   // color: flag carries identity
    return escapeHTML(teamKnown(team) ? code : "TBD");           // mono: 3-letter code
  };
  const line = (team: WcMatch["home"], code: string, won: boolean, score: number | null) => {
    const fav = isFav(code) ? `color:${theme.fav};` : "";
    const w = won ? "font-weight:700;" : "";
    const sc = finished ? `<span class="wc-kscore">${score}</span>` : "";
    return `<div class="wc-kteam" style="${fav}${w}">${flagImg(theme, code)}<span class="wc-kname">${label(team, code)}</span>${sc}</div>`;
  };
  const when = live ? "LIVE"
    : finished ? "Full time"
    : compact ? shortChicagoDate(mm.dateChicago)
    : `${shortChicagoDate(mm.dateChicago)} · ${mm.timeChicago}`;
  // In compact (inner) boxes, only render a side once it's decided — so a single advanced
  // team shows as one prominent centered flag/code rather than a blank second line.
  const homeLine = (!compact || teamKnown(mm.home)) ? line(mm.home, h, homeWon, mm.homeScore) : "";
  const awayLine = (!compact || teamKnown(mm.away)) ? line(mm.away, a, awayWon, mm.awayScore) : "";
  return `<div class="${cls}${live ? " wc-klive" : ""}">
    ${homeLine}
    ${awayLine}
    <div class="wc-kwhen">${escapeHTML(when)}</div>
  </div>`;
}

/**
 * Round-of-32 layout: the full knockout bracket, data-driven and self-advancing. The 16 R32
 * ties sit on the two outer edges; the R16→SF columns converge toward a center Final box. Each
 * round past R32 is computed from the previous round's winners (`advanceRound`) so a team that
 * has won its tie (e.g. Canada) immediately appears in its next-round box, even though the
 * source leaves those matches' teams empty until later. Inner rounds use compact 3-letter codes
 * (+ flags on color) since their boxes are narrow; R32 uses full names. Bracket order = match id.
 */
function r32Layout(data: WorldCupData, theme: WcTheme): string {
  const stage = (s: WcStage) => data.knockout.filter((m) => m.stage === s).slice().sort((a, b) => a.id - b.id);
  const r32 = stage("R32");

  if (r32.length === 0) {
    return `${header(data, "Round of 32")}
    <div class="wc-empty">Round of 32 fixtures not available yet</div>`;
  }

  const r16 = advanceRound(r32, stage("R16"));
  const qf = advanceRound(r16, stage("QF"));
  const sf = advanceRound(qf, stage("SF"));
  const final = advanceRound(sf, stage("FINAL"))[0];

  const dates = r32.map((m) => m.dateChicago).filter(Boolean).sort();
  const range = dates.length > 0 && dates[0] !== dates[dates.length - 1]
    ? `${shortChicagoDate(dates[0])} – ${shortChicagoDate(dates[dates.length - 1])}`
    : dates.length > 0 ? shortChicagoDate(dates[0]) : "";
  const subtitle = `Round of 32${range ? ` · ${range}` : ""} · times CT`;

  const lh = <T,>(arr: T[]): T[] => arr.slice(0, Math.ceil(arr.length / 2));
  const rh = <T,>(arr: T[]): T[] => arr.slice(Math.ceil(arr.length / 2));
  const col = (matches: WcMatch[], cls = "", compact = false) =>
    `<div class="wc-kcol ${cls}">${matches.map((mm) => bracketBox(mm, theme, compact)).join("")}</div>`;

  return `${header(data, subtitle)}
  <div class="wc-kbracket">
    <div class="wc-kside">
      ${col(lh(r32), "wc-kr32")}
      ${col(lh(r16), "", true)}
      ${col(lh(qf), "", true)}
      ${col(lh(sf), "", true)}
    </div>
    <div class="wc-kcenter">
      <div class="wc-kcenter-label">FINAL</div>
      ${final ? bracketBox(final, theme, true, "wc-kfinal") : ""}
    </div>
    <div class="wc-kside wc-kside-right">
      ${col(rh(sf), "", true)}
      ${col(rh(qf), "", true)}
      ${col(rh(r16), "", true)}
      ${col(rh(r32), "wc-kr32")}
    </div>
  </div>`;
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
  const champCode = data.champion ? teamCode(data.champion) : "";
  const champName = data.champion ? teamLabel(data.champion, 20) : "—";
  const bigFlag = data.champion && theme.flag
    ? theme.flag(champCode).replace('class="wc-flag"', 'class="wc-flag-big"')
    : "";
  const score = final ? `${final.homeScore}-${final.awayScore}` : "";
  const matchup = final ? `${teamLabel(final.home, 20)} ${escapeHTML(score)} ${teamLabel(final.away, 20)}` : "";
  return `${header(data)}
  <div class="wc-champion">
    <div class="wc-champ-label">CHAMPIONS</div>
    ${bigFlag}
    <div class="wc-champ-team" style="color:${theme.win}">${champName}</div>
    <div class="wc-champ-final">${matchup}</div>
  </div>`;
}

/** Build the full HTML document for a given theme + page CSS. */
/**
 * Build the full HTML document. The stylesheet is supplied entirely by `theme.styleCSS`
 * (separate per display — see worldcup-styles.ts) so the two displays never share CSS;
 * only the structure (this function + the section builders) and the logic are shared.
 */
export function renderWorldCupHTML(data: WorldCupData, theme: WcTheme): string {
  let body: string;
  switch (data.phase) {
    case "r32": body = r32Layout(data, theme); break;
    case "knockout": body = bracketLayout(data, theme); break;
    case "champion": body = championLayout(data, theme); break;
    default: body = splitLayout(data, theme); break; // group
  }
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=800">
<title>World Cup 2026</title>
<style>
  :root { ${theme.rootCSS} }
  ${theme.styleCSS}
</style>
</head>
<body>${body}</body>
</html>`;
}
