/**
 * The series on the three panels (DECISIONS #71, #73). The page for "now" is worked out
 * from the clock and the schedule and read from R2, where it is stored as a finished
 * one-request document: the Worker passes it through, with no image work, no KV and no
 * AI. Every panel gets the same full-colour page and converts it itself. Previews (?at=),
 * the proof page and style tests reveal unreleased work, so they need SERIES_KEY when set.
 * The aired archive (/series/aired) is public and shows only what has already aired (#94).
 */

import { hudPageHTML } from "../hud/page";
import { pngToBase64 } from "../png";
import { htmlResponse } from "../response";
import { airedPages } from "../series/aired";
import { chicagoParts, daysUntil, resolveSlot, seriesSlot } from "../series/clock";
import { renderCard } from "../series/render";
import { SCHEDULE_KEY, sceneKey, slotKey, teaserKey, testKey } from "../series/keys";
import { isSchedule } from "../series/schedule";
import type { Schedule } from "../series/types";
import { SCENES_PER_CHAPTER } from "../series/types";
import type { Env } from "../types";
import { unavailableHTML } from "./unavailable";

const TITLE = "The Series";
const notFound = () => new Response("Not found", { status: 404 });

function keyOk(env: Env, url: URL): boolean {
  return !env.SERIES_KEY || url.searchParams.get("key") === env.SERIES_KEY;
}

export async function loadSchedule(bucket: R2Bucket): Promise<Schedule | null> {
  try {
    const obj = await bucket.get(SCHEDULE_KEY);
    if (!obj) return null;
    const s: unknown = JSON.parse(await obj.text());
    if (!isSchedule(s)) {
      console.error("series: schedule has the wrong shape");
      return null;
    }
    return s;
  } catch (err) {
    console.error("series: schedule unreadable:", err);
    return null;
  }
}

/** "2026-11-09T08:30" read as Chicago wall-clock time. */
function parseChicago(at: string): Date | null {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(at)) return null;
  const guess = new Date(at + ":00Z");
  for (const offset of [5, 6]) {
    const d = new Date(guess.getTime() + offset * 3600000);
    const local = new Intl.DateTimeFormat("en-CA", { timeZone: "America/Chicago", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" })
      .format(d).replace(", ", "T");
    if (local === at) return d;
  }
  return null;
}

const textPage = (detail: string) => htmlResponse(unavailableHTML(TITLE, detail), "no-store");

/** A stored page, or the text page when it is missing or unreadable. */
async function storedPage(env: Env, key: string, missing: string): Promise<Response> {
  try {
    const obj = await env.SERIES.get(key);
    if (!obj) {
      console.error(`series: missing page ${key}`);
      return textPage(missing);
    }
    return htmlResponse(await obj.text(), "no-store");
  } catch (err) {
    console.error("series: read failed:", err);
    return textPage("This page could not be loaded. It will try again at the next refresh.");
  }
}

export async function handleSeries(env: Env, url: URL, now: Date = new Date()): Promise<Response> {
  const at = url.searchParams.get("at");
  if (at !== null && !keyOk(env, url)) return notFound();
  const when = at !== null ? parseChicago(at) : now;
  if (!when) return textPage("The preview time must look like 2026-11-09T08:30.");
  const schedule = await loadSchedule(env.SERIES);
  if (!schedule) return textPage("The series has not started yet.");
  let key: string | null;
  try {
    const slot = resolveSlot(seriesSlot(when, schedule), schedule, at !== null);
    // Before a premiere the card counts the days down, so it is drawn here, not stored (#80).
    const next = slot.kind === "between" ? schedule.seasons.find((s) => s.number === slot.next) : undefined;
    const days = next ? daysUntil(next.premiere, when) : -1;
    if (next && (days > 0 || (days === 0 && chicagoParts(when).hour < 8))) {
      try {
        const png = await renderCard(next, "before", days);
        return htmlResponse(hudPageHTML(TITLE, pngToBase64(png)), "no-store");
      } catch (err) {
        console.error("series: countdown card failed, serving the stored card:", err);
      }
    }
    key = slotKey(slot);
  } catch (err) {
    console.error("series: slot failed:", err);
    return textPage("This scene could not be loaded. It will try again at the next refresh.");
  }
  if (!key) return textPage("The series has not started yet.");
  return storedPage(env, key, "This scene is not ready yet. It will appear at the next refresh.");
}

const json = (data: unknown) => new Response(JSON.stringify(data), {
  headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" },
});

/**
 * Every page that has aired, oldest first, for outside readers (DECISIONS #94). Public and
 * keyless: it takes no preview time and never lists pending chapters, so it ignores ?at=
 * and ?key=. An unreadable schedule gives an empty list marked unavailable, never a guess.
 */
export async function handleSeriesAired(env: Env, now: Date = new Date()): Promise<Response> {
  const schedule = await loadSchedule(env.SERIES);
  if (!schedule) return json({ state: "unavailable", pages: [] });
  const pages = airedPages(schedule, now).map(({ key: _key, ...page }) => page);
  return json({ state: "ok", pages });
}

/** One aired page, the same document the panels got; 404 unless it has aired at this moment. */
export async function handleSeriesAiredPage(env: Env, url: URL, now: Date = new Date()): Promise<Response> {
  const p = url.searchParams.get("p") ?? "";
  const schedule = await loadSchedule(env.SERIES);
  const page = schedule ? airedPages(schedule, now).find((x) => x.path === p) : undefined;
  if (!page) return notFound();
  return storedPage(env, page.key, "This page is missing.");
}

/** Each scene is framed at panel size, one request per frame; inlining 26 pages would be too heavy. */
const proofFrame = (k: string, key: string | null) =>
  `<iframe src="/series/proof/page?k=${encodeURIComponent(k)}${key ? `&key=${encodeURIComponent(key)}` : ""}" width="800" height="480" loading="lazy"></iframe>`;

export async function handleSeriesProof(env: Env, url: URL): Promise<Response> {
  if (!keyOk(env, url)) return notFound();
  const season = Number(url.searchParams.get("season"));
  const chapter = Number(url.searchParams.get("chapter"));
  if (!Number.isInteger(season) || !Number.isInteger(chapter) || chapter < 1) return textPage("Ask for ?season=N&chapter=N.");
  const schedule = await loadSchedule(env.SERIES);
  const s = schedule?.seasons.find((x) => x.number === season);
  const status = s?.approved.includes(chapter) ? "APPROVED" : s?.pending.includes(chapter) ? "PENDING" : "NOT IMPORTED";
  const key = url.searchParams.get("key");
  const rows: string[] = [];
  for (let n = 1; n <= SCENES_PER_CHAPTER; n++) {
    rows.push(`<figure><figcaption>Scene ${n}</figcaption>${proofFrame(sceneKey(season, chapter, n, false), key)}</figure>`);
  }
  rows.push(`<figure><figcaption>Teaser</figcaption>${proofFrame(teaserKey(season, chapter), key)}</figure>`);
  return htmlResponse(`<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Proof S${season} C${chapter}</title>
<style>body{font-family:Helvetica,Arial,sans-serif;margin:24px;background:#fff;color:#000}figure{margin:0 0 28px}iframe{border:1px solid #000;display:block}</style></head>
<body><h1>Season ${season}, chapter ${chapter}: ${status}</h1><p>Every scene exactly as the panels receive it; each panel converts it itself. Approve with an APPROVE message in the mailbox.</p>${rows.join("")}</body></html>`, "no-store");
}

/** One finished series page, for a proof frame. Keyed, chapter pages only. */
export async function handleSeriesProofPage(env: Env, url: URL): Promise<Response> {
  if (!keyOk(env, url)) return notFound();
  const k = url.searchParams.get("k") ?? "";
  if (!/^series\/s\d{2}\/c\d{2}\/[a-z0-9-]+\.html$/.test(k)) return notFound();
  return storedPage(env, k, "This page is missing.");
}

/**
 * A writer's style test as the panels will show it, for judging art on a real device: the
 * picture untouched, only cropped to the screen. Keyed: the watcher uploads every STYLE TEST
 * to series/tests/ (DECISIONS #72).
 */
export async function handleSeriesTest(env: Env, url: URL): Promise<Response> {
  if (!keyOk(env, url)) return notFound();
  const key = testKey(url.searchParams.get("folder") ?? "", url.searchParams.get("image") ?? "");
  if (!key) return notFound();
  return storedPage(env, key, "No style test picture with that folder and image name.");
}
