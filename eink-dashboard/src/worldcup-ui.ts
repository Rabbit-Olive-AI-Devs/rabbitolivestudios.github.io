/**
 * World Cup 2026 — pure logic + HTML section builders.
 *
 * Shared by pages/worldcup.ts (mono) and pages/color-worldcup.ts (color).
 * Logic functions are exported and unit-tested; render functions take a
 * `WcTheme` so the two displays differ only in palette.
 */

import { escapeHTML } from "./escape";
import type {
  WcMatch, WcStage, WcStatus, WcPhase, WcGroup, WorldCupData,
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

/** Full team name, escaped, truncated to maxChars with an ellipsis. "TBD" if empty. */
export function teamLabel(team: { name: string; code: string }, maxChars: number): string {
  const name = (team.name ?? "").trim();
  if (!name) return "TBD";
  const shown = name.length > maxChars ? name.slice(0, Math.max(1, maxChars - 1)) + "…" : name;
  return escapeHTML(shown);
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
 * - else knockout sub-phase by the latest round with any match:
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
  fav: string;       // favorite-team accent color (CSS color)
  win: string;       // win/qualified accent
  live: string;      // live / today accent
}

export const FAVORITE_TEAM: string = "BRA"; // 3-letter code; "" disables the highlight

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
