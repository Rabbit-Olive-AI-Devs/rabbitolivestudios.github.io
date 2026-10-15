/**
 * The series on the three panels (DECISIONS #71). The page for "now" is worked out
 * from the clock and the schedule, read from R2 and inlined: one request, no KV,
 * no AI, no image work. Previews (?at=) and the proof page reveal unreleased
 * chapters, so they need SERIES_KEY when it is set.
 */

import { hudPageHTML } from "../hud/page";
import { pngToBase64 } from "../png";
import { htmlResponse } from "../response";
import { resolveSlot, seriesSlot } from "../series/clock";
import { SCHEDULE_KEY, sceneKey, slotKey, teaserKey, testGreyKey, testKey, testOriginalKey } from "../series/keys";
import { isSchedule } from "../series/schedule";
import type { PanelId, Schedule } from "../series/types";
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

export async function handleSeries(env: Env, url: URL, panel: PanelId, now: Date = new Date()): Promise<Response> {
  const at = url.searchParams.get("at");
  if (at !== null && !keyOk(env, url)) return notFound();
  const when = at !== null ? parseChicago(at) : now;
  if (!when) return textPage("The preview time must look like 2026-11-09T08:30.");
  const schedule = await loadSchedule(env.SERIES);
  if (!schedule) return textPage("The series has not started yet.");
  try {
    const key = slotKey(resolveSlot(seriesSlot(when, schedule), schedule, at !== null), panel);
    if (!key) return textPage("The series has not started yet.");
    const obj = await env.SERIES.get(key);
    if (!obj) {
      console.error(`series: missing page ${key}`);
      return textPage("This scene is not ready yet. It will appear at the next refresh.");
    }
    return htmlResponse(hudPageHTML(TITLE, pngToBase64(new Uint8Array(await obj.arrayBuffer()))), "no-store");
  } catch (err) {
    console.error("series: read failed:", err);
    return textPage("This scene could not be loaded. It will try again at the next refresh.");
  }
}

/** Proof images are served one per request: inlining 26 pages would blow the CPU budget. */
const proofImg = (k: string, key: string | null) =>
  `<img src="/series/proof/img?k=${encodeURIComponent(k)}${key ? `&key=${encodeURIComponent(key)}` : ""}" width="800" height="480" alt="">`;

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
    rows.push(`<figure><figcaption>Scene ${n}</figcaption>${proofImg(sceneKey(season, chapter, n, false, "office"), key)}</figure>`);
  }
  rows.push(`<figure><figcaption>Teaser</figcaption>${proofImg(teaserKey(season, chapter, "office"), key)}</figure>`);
  return htmlResponse(`<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Proof S${season} C${chapter}</title>
<style>body{font-family:Helvetica,Arial,sans-serif;margin:24px;background:#fff;color:#000}figure{margin:0 0 28px}img{border:1px solid #000;max-width:100%;height:auto}</style></head>
<body><h1>Season ${season}, chapter ${chapter}: ${status}</h1><p>Office panel version. A missing scene shows as a broken image. Approve in Slack; the engineer runs the approve step.</p>${rows.join("")}</body></html>`, "no-store");
}

/** One finished page from the series bucket, for the proof page. Keyed, PNG pages only. */
export async function handleSeriesProofImage(env: Env, url: URL): Promise<Response> {
  if (!keyOk(env, url)) return notFound();
  const k = url.searchParams.get("k") ?? "";
  if (!/^series\/s\d{2}\/c\d{2}\/[a-z0-9-]+\.png$/.test(k)) return notFound();
  const obj = await env.SERIES.get(k);
  if (!obj) return notFound();
  return new Response(await obj.arrayBuffer(), { headers: { "Content-Type": "image/png", "Cache-Control": "no-store" } });
}

/**
 * A writer's style test as one panel shows it, for judging art on the real device. Keyed:
 * the watcher uploads every STYLE TEST to series/tests/ (DECISIONS #72).
 */
export async function handleSeriesTest(env: Env, url: URL, panel: PanelId): Promise<Response> {
  if (!keyOk(env, url)) return notFound();
  const folder = url.searchParams.get("folder") ?? "", image = url.searchParams.get("image") ?? "";
  // original=1 sends the writer's picture unconverted, leaving the panel's own processing to it.
  // grey=1 sends the four-grey version, for comparing with the 1-bit one on a real E1001.
  const original = url.searchParams.get("original") === "1";
  const grey = url.searchParams.get("grey") === "1";
  const key = original ? testOriginalKey(folder, image) : grey ? testGreyKey(folder, image) : testKey(folder, image, panel);
  if (!key) return notFound();
  try {
    const obj = await env.SERIES.get(key);
    if (!obj) return textPage("No style test picture with that folder and image name.");
    return htmlResponse(hudPageHTML("Style Test", pngToBase64(new Uint8Array(await obj.arrayBuffer())), original ? "image/jpeg" : "image/png"), "no-store");
  } catch (err) {
    console.error("series: style test read failed:", err);
    return textPage("This style test could not be loaded.");
  }
}
