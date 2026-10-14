/**
 * Mechanical checks of a season package (docs/series/SEASON-FORMAT.md section 7).
 * Every problem names its chapter and scene, so a writer can fix a whole season in
 * one pass. Lettering fit is checked later, when the pages are drawn (render.ts).
 */

import { BANNED_NAMES } from "./banned";
import { TEMPLATES } from "./templates";
import type { Chapter, SeasonPackage, TemplateId } from "./types";
import { plannedChapters, SCENES_PER_CHAPTER } from "./types";

const DAYS = ["MON", "TUE", "WED", "THU", "FRI"];

function badChars(s: string): string[] {
  const out: string[] = [];
  for (const ch of s) {
    const c = ch.codePointAt(0)!;
    if (c < 32 || c > 126) out.push(`U+${c.toString(16).toUpperCase().padStart(4, "0")} "${ch}"`);
  }
  return [...new Set(out)];
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function bannedIn(text: string, names: string[]): string[] {
  return names.filter((n) => new RegExp(`(^|[^A-Za-z0-9])${escapeRe(n)}([^A-Za-z0-9]|$)`, "i").test(text));
}

function isMonday(date: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(date) && new Date(date + "T12:00:00Z").getUTCDay() === 1;
}

export function validatePackage(
  raw: unknown, artFiles: Set<string>, extraBanned: string[],
): { pkg: SeasonPackage | null; problems: string[] } {
  const problems: string[] = [];
  const p = raw as SeasonPackage;
  if (!p || p.format !== 1 || !p.season || !Array.isArray(p.chapters)) {
    return { pkg: null, problems: ["season.json: not a format 1 season package"] };
  }
  const banned = [...BANNED_NAMES, ...extraBanned];
  const s = p.season;
  for (const f of ["show", "title", "writer"] as const) if (!s[f]) problems.push(`season: missing ${f}`);
  if (!Number.isInteger(s.number) || s.number < 0) problems.push("season: number must be 0 or more");
  if (!isMonday(s.premiere)) problems.push(`season: premiere ${s.premiere} is not a Monday`);
  const planned = plannedChapters(s.number);
  if (p.chapters.length > planned) problems.push(`season: ${p.chapters.length} chapters, season ${s.number} has ${planned}`);

  const text = (where: string, value: string | undefined) => {
    if (value === undefined) return;
    const bad = badChars(value);
    if (bad.length) problems.push(`${where}: characters the panel font does not have: ${bad.join(", ")}`);
    for (const n of bannedIn(value, banned)) problems.push(`${where}: banned name "${n}"`);
  };

  p.chapters.forEach((ch: Chapter, ci) => {
    const C = `chapter ${ci + 1}`;
    if (ch.number !== ci + 1) problems.push(`${C}: numbered ${ch.number}, expected ${ci + 1}`);
    if (!ch.title) problems.push(`${C}: missing title`);
    if (!ch.teaser) problems.push(`${C}: missing teaser`);
    text(`${C} title`, ch.title);
    text(`${C} teaser`, ch.teaser);
    if (!Array.isArray(ch.scenes) || ch.scenes.length !== SCENES_PER_CHAPTER) {
      problems.push(`${C}: ${ch.scenes?.length ?? 0} scenes, needs ${SCENES_PER_CHAPTER}`);
    }
    const newArtByDay = [0, 0, 0, 0, 0];
    (ch.scenes ?? []).forEach((sc, si) => {
      const S = `${C} scene ${si + 1}`;
      if (sc.number !== si + 1) problems.push(`${S}: numbered ${sc.number}, expected ${si + 1}`);
      const t = TEMPLATES[sc.template as TemplateId];
      if (!t) { problems.push(`${S}: unknown template "${sc.template}"`); return; }
      if (si === 0 && sc.template !== "splash") problems.push(`${S}: must be a splash (the chapter's title card)`);
      text(`${S} caption`, sc.caption);
      text(`${S} narration`, sc.narration);
      if (t.strip && !sc.narration) problems.push(`${S}: missing narration`);
      if (!Array.isArray(sc.panels) || sc.panels.length !== t.panels.length) {
        problems.push(`${S}: ${sc.panels?.length ?? 0} panels, template ${t.id} has ${t.panels.length}`);
        return;
      }
      const prev = si > 0 ? ch.scenes[si - 1] : null;
      sc.panels.forEach((pn, pi) => {
        if (pn.art === "hold") {
          if (!prev || prev.template !== sc.template) problems.push(`${S}: "hold" after a ${prev ? prev.template : "chapter start"}; hold needs the same template`);
        } else {
          if (!artFiles.has(pn.art)) problems.push(`${S}: art file "${pn.art}" not in art/`);
          newArtByDay[Math.min(4, Math.floor(si / 5))]++;
        }
        text(`${S} panel ${pi + 1} caption`, pn.caption);
        const balloons = pn.balloons ?? [];
        if (balloons.length > t.balloonsPerPanel) problems.push(`${S}: ${balloons.length} balloons in panel ${pi + 1}, room for ${t.balloonsPerPanel}`);
        balloons.forEach((b, bi) => {
          if (!b.speaker || !b.text) problems.push(`${S}: balloon ${bi + 1} needs a speaker and text`);
          if (b.anchor !== null && !(Array.isArray(b.anchor) && b.anchor.length === 2)) problems.push(`${S}: balloon ${bi + 1} anchor must be [x, y] or null`);
          text(`${S} balloon ${bi + 1}`, `${b.speaker} ${b.text}`);
        });
      });
    });
    if ((ch.scenes ?? []).length === SCENES_PER_CHAPTER) {
      newArtByDay.forEach((n, d) => { if (n === 0) problems.push(`${C} ${DAYS[d]}: no new picture`); });
    }
  });
  return { pkg: problems.length ? null : p, problems };
}

/** One valid chapter: a splash then wide scenes, one new picture per scene. */
export function samplePackage(): SeasonPackage {
  const pad = (n: number) => String(n).padStart(2, "0");
  return {
    format: 1,
    season: { number: 0, show: "THE LONG DRIFT", title: "PROLOGUE", writer: "Claude", premiere: "2026-10-26" },
    chapters: [{
      number: 1, title: "THE SALT MARKET", teaser: "Next week: the crate starts talking.",
      scenes: Array.from({ length: 25 }, (_, i) => i === 0
        ? { number: 1, template: "splash" as const, panels: [{ art: "c01-s01.png" }] }
        : {
          number: i + 1, template: "wide" as const, caption: "MERREK PRIME. DUSK.",
          narration: "Vess talks the harbour master down to half the docking fee.",
          panels: [{ art: `c01-s${pad(i + 1)}.png`, balloons: [{ speaker: "BOLT", text: "I will not be sharing the odds.", anchor: [600, 200] as [number, number] }] }],
        }),
    }],
  };
}
