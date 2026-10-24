/**
 * Build one chapter's finished pages from a validated package. Pictures come in
 * through `art` (decoded by the import CLI); this file resolves "hold", crops each
 * picture to its panel around its focus, maps anchors and keep-clear boxes into
 * page coordinates, and renders. Everything is collected before anything is
 * returned, so one run lists every problem in the chapter.
 */

import { hudPageHTML } from "../hud/page";
import { pngToBase64 } from "../png";
import type { RGBImage } from "./art";
import { cropToFill, isNearlyBlank } from "./art";
import { cardKey, sceneKey, teaserKey } from "./keys";
import type { PanelArt } from "./render";
import { renderCard, renderScene, renderTeaser } from "./render";
import type { Rect } from "./templates";
import { TEMPLATES } from "./templates";
import type { SeasonInfo, SeasonPackage } from "./types";
import { plannedChapters } from "./types";

/**
 * The finished one-request document the panels load, stored as-is in R2 so the Worker only
 * passes it through: encoding a full-colour page per request would cost too much CPU.
 */
function toDocument(png: Uint8Array): Uint8Array {
  return new TextEncoder().encode(hudPageHTML("The Series", pngToBase64(png)));
}

export type ArtSource = (name: string) => RGBImage | null;

export async function buildChapter(
  pkg: SeasonPackage, chapterNo: number, art: ArtSource,
): Promise<{ files: Map<string, Uint8Array>; problems: string[] }> {
  const files = new Map<string, Uint8Array>();
  const problems: string[] = [];
  const ch = pkg.chapters[chapterNo - 1];
  if (!ch) return { files, problems: [`chapter ${chapterNo}: not in the package`] };
  // Override only the visible scene credit; preserve season metadata and cards.
  const sceneInfo = ch.writer === undefined ? pkg.season : { ...pkg.season, writer: ch.writer };
  const season = pkg.season.number;
  let prevPanels: PanelArt[] = [];

  for (const scene of ch.scenes) {
    const S = `chapter ${chapterNo} scene ${scene.number}`;
    const t = TEMPLATES[scene.template];
    const panels: PanelArt[] = [];
    let ok = true;
    scene.panels.forEach((pn, pi) => {
      const rect = t.panels[pi];
      if (pn.art === "hold") {
        const held = prevPanels[pi];
        if (!held) { problems.push(`${S}: nothing to hold`); ok = false; return; }
        // The held picture is already cropped: this scene's anchors and keep-clear boxes are
        // in panel pixels. The previous scene's keep-clear boxes still apply (same picture).
        const anchors = (pn.balloons ?? []).map((b, bi): [number, number] | null => {
          if (b.anchor === null) return null;
          const [x, y] = b.anchor;
          if (x < 0 || y < 0 || x >= rect.w || y >= rect.h) {
            problems.push(`${S}: balloon ${bi + 1} anchor [${x}, ${y}] is outside the held panel (${rect.w}x${rect.h})`);
            ok = false;
          }
          return [rect.x + x, rect.y + y];
        });
        const own: Rect[] = (pn.clear ?? []).map(([x, y, w, h]) => ({ x: rect.x + x, y: rect.y + y, w, h }));
        panels.push({ image: held.image, anchors, clear: [...held.clear, ...own] });
        return;
      }
      const img = art(pn.art);
      if (!img) { problems.push(`${S}: picture ${pn.art} could not be read`); ok = false; return; }
      if (img.w < rect.w || img.h < rect.h) {
        problems.push(`${S}: picture ${pn.art} is ${img.w}x${img.h}, smaller than the ${rect.w}x${rect.h} panel`);
        ok = false; return;
      }
      const { image, map } = cropToFill(img, rect.w, rect.h, pn.focus);
      if (isNearlyBlank(image)) { problems.push(`${S}: picture ${pn.art} is nearly blank`); ok = false; return; }
      const anchors = (pn.balloons ?? []).map((b, bi): [number, number] | null => {
        if (b.anchor === null) return null;
        const [x, y] = map(b.anchor);
        if (x < 0 || y < 0 || x >= rect.w || y >= rect.h) {
          problems.push(`${S}: balloon ${bi + 1} anchor [${b.anchor[0]}, ${b.anchor[1]}] is cropped out of the panel`);
          ok = false;
        }
        return [rect.x + x, rect.y + y];
      });
      const clear: Rect[] = (pn.clear ?? []).map(([x, y, w, h]) => {
        const [x0, y0] = map([x, y]);
        const [x1, y1] = map([x + w, y + h]);
        return { x: rect.x + x0, y: rect.y + y0, w: x1 - x0, h: y1 - y0 };
      });
      panels.push({ image, anchors, clear });
    });
    if (!ok) continue;
    const { pages, problems: p } = await renderScene(sceneInfo, ch, scene, panels);
    problems.push(...p);
    if (pages.s) files.set(sceneKey(season, chapterNo, scene.number, false), toDocument(pages.s));
    if (pages.r) files.set(sceneKey(season, chapterNo, scene.number, true), toDocument(pages.r));
    prevPanels = panels;
  }

  const finale = chapterNo === plannedChapters(season);
  // The existing teaser has no writer credit. Keep its layout and pixels unchanged.
  const teaser = await renderTeaser(pkg.season, ch, finale);
  files.set(teaserKey(season, chapterNo), toDocument(teaser));
  return { files, problems };
}

export async function buildCards(info: SeasonInfo): Promise<Map<string, Uint8Array>> {
  const files = new Map<string, Uint8Array>();
  const before = await renderCard(info, "before");
  const after = await renderCard(info, "after");
  files.set(cardKey(info.number, "before"), toDocument(before));
  files.set(cardKey(info.number, "after"), toDocument(after));
  return files;
}
