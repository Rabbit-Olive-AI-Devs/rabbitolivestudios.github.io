/**
 * The pages that have aired, for outside readers such as FrameBot (DECISIONS #94). A page
 * has aired when its chapter is approved and its first airing time has passed on the
 * Chicago clock. A pure function of the time and the schedule: there is no preview time,
 * no key, and pending chapters never count, so nothing here can reveal unaired work.
 * Weekend replays repeat scenes already listed, so only the scenes and teasers are listed.
 */

import { shiftDateStr } from "../date-utils";
import { chicagoParts, sceneTimeLabel } from "./clock";
import { sceneKey, teaserKey } from "./keys";
import type { Schedule } from "./types";
import { SCENES_PER_CHAPTER } from "./types";

export interface AiredPage {
  season: number;
  chapter: number;
  /** 1-25, or null for the chapter's teaser. */
  scene: number | null;
  label: string;
  /** Passed back as /series/aired/page?p=…; checked again on that request. */
  path: string;
  /** The page's R2 key; not shown to readers. */
  key: string;
}

const pad = (n: number) => String(n).padStart(2, "0");
const SCENE_HOURS = [8, 10, 12, 14, 16];
const TEASER_DAY = 6;
const TEASER_HOUR = 19;

export function airedPages(schedule: Schedule, now: Date): AiredPage[] {
  const { date, hour } = chicagoParts(now);
  const due = (airDate: string, airHour: number) => date > airDate || (date === airDate && hour >= airHour);
  const pages: AiredPage[] = [];
  const seasons = [...schedule.seasons].sort((a, b) => a.premiere.localeCompare(b.premiere));
  for (const s of seasons) {
    const chapters = [...new Set(s.approved)].filter((c) => c >= 1 && c <= s.chapters).sort((a, b) => a - b);
    for (const c of chapters) {
      const week = shiftDateStr(s.premiere, (c - 1) * 7);
      const base = `s${pad(s.number)}/c${pad(c)}`;
      for (let n = 1; n <= SCENES_PER_CHAPTER; n++) {
        if (!due(shiftDateStr(week, Math.floor((n - 1) / 5)), SCENE_HOURS[(n - 1) % 5])) break;
        pages.push({
          season: s.number, chapter: c, scene: n,
          label: `${s.title}, chapter ${c}, scene ${n} (${sceneTimeLabel(n)})`,
          path: `${base}/s${pad(n)}`, key: sceneKey(s.number, c, n, false),
        });
      }
      if (due(shiftDateStr(week, TEASER_DAY), TEASER_HOUR)) {
        pages.push({
          season: s.number, chapter: c, scene: null,
          label: `${s.title}, chapter ${c}, teaser`,
          path: `${base}/teaser`, key: teaserKey(s.number, c),
        });
      }
    }
  }
  return pages;
}
