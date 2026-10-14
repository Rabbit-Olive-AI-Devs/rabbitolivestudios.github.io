/**
 * The five page templates (docs/series/SEASON-FORMAT.md section 3). Panels are
 * page rectangles; the top bar ends at y=36, the narration strip starts at 400.
 */

import type { TemplateId } from "./types";

export interface Rect { x: number; y: number; w: number; h: number }

export interface Template {
  id: TemplateId;
  panels: Rect[];
  /** Narration strip and credits under the art. */
  strip: boolean;
  /** Text column beside the art (panel-text). */
  column: Rect | null;
  balloonsPerPanel: number;
  /** A short caption at the top of each panel (strip). */
  panelCaptions: boolean;
}

export const STRIP_TOP = 400;
/** Balloons prefer the top 40% of a panel: the calm zone writers keep plain. */
export const CALM_FRACTION = 0.4;

export const TEMPLATES: Record<TemplateId, Template> = {
  wide: { id: "wide", panels: [{ x: 12, y: 46, w: 776, h: 346 }], strip: true, column: null, balloonsPerPanel: 2, panelCaptions: false },
  split: {
    id: "split", panels: [{ x: 12, y: 46, w: 382, h: 346 }, { x: 406, y: 46, w: 382, h: 346 }],
    strip: true, column: null, balloonsPerPanel: 2, panelCaptions: false,
  },
  strip: {
    id: "strip", panels: [{ x: 12, y: 46, w: 252, h: 346 }, { x: 274, y: 46, w: 252, h: 346 }, { x: 536, y: 46, w: 252, h: 346 }],
    strip: true, column: null, balloonsPerPanel: 1, panelCaptions: true,
  },
  splash: { id: "splash", panels: [{ x: 0, y: 37, w: 800, h: 443 }], strip: false, column: null, balloonsPerPanel: 0, panelCaptions: false },
  "panel-text": {
    id: "panel-text", panels: [{ x: 12, y: 46, w: 476, h: 426 }], strip: false,
    column: { x: 500, y: 46, w: 288, h: 426 }, balloonsPerPanel: 1, panelCaptions: false,
  },
};
