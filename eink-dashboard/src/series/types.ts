/**
 * The series (DECISIONS #71): a graphic novel written and drawn ahead of time by
 * AI writers, one season per quarter, shown on all three panels. These types
 * describe the season package the writers deliver (docs/series/SEASON-FORMAT.md),
 * the schedule kept in R2, and the slot the clock picks.
 */


export type TemplateId = "wide" | "split" | "strip" | "splash" | "panel-text";

export interface Balloon {
  speaker: string;
  text: string;
  /** The speaker's head in the art file's own pixels; null = a voice from off the picture. */
  anchor: [number, number] | null;
}

export interface ScenePanel {
  /** A file in the package's art/ folder, or "hold" to keep the previous scene's picture. */
  art: string;
  focus?: [number, number];
  caption?: string;
  balloons?: Balloon[];
  /** Boxes no balloon may cover, [x, y, w, h] in the art file's own pixels. */
  clear?: [number, number, number, number][];
}

export interface Scene {
  number: number;
  template: TemplateId;
  caption?: string;
  narration?: string;
  panels: ScenePanel[];
}

export interface Chapter {
  number: number;
  title: string;
  /** Optional chapter-only credit; omitted chapters inherit the season writer. */
  writer?: string;
  teaser: string;
  scenes: Scene[];
}

export interface SeasonInfo {
  /** 0 is the Prologue. */
  number: number;
  /** The show's title, as printed in the top bar. */
  show: string;
  title: string;
  writer: string;
  /** A Monday, YYYY-MM-DD. */
  premiere: string;
}

export interface SeasonPackage {
  format: 1;
  season: SeasonInfo;
  chapters: Chapter[];
}

export interface ScheduleSeason extends SeasonInfo {
  /** Chapters planned for the season. */
  chapters: number;
  /** Chapters the showrunner approved: only these air. */
  approved: number[];
  /** Chapters uploaded and waiting for approval: previews only. */
  pending: number[];
}

export interface Schedule {
  seasons: ScheduleSeason[];
}

export type Slot =
  | { kind: "scene"; season: number; chapter: number; scene: number; replay: boolean }
  | { kind: "teaser"; season: number; chapter: number }
  | { kind: "between"; next: number | null; prev: number | null };

export const SCENES_PER_CHAPTER = 25;

export function plannedChapters(season: number): number {
  return season === 0 ? 10 : 13;
}
