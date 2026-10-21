/**
 * The story book on the web (DECISIONS #97): the desk story page's frames, as the Mac mini's
 * story book job saved them (one per moment or serial beat, only once aired, repeats left out),
 * read in a phone browser and current all day. The job uploads to the private series bucket:
 *   storybook/days.json                 [{date, day, frames}]
 *   storybook/days/<date>.json          {date, day, frames: [{file, caption}]}
 *   storybook/frames/<date>/<file>      the PNG
 * Behind STORYBOOK_KEY (404 without it, and while it is unset). Reads only R2; no KV, no drawing.
 */

import type { Env } from "./types";
import { escapeHTML } from "./escape";
import { htmlResponse } from "./response";

export interface BookDay { date: string; day: number; frames: number }
export interface BookFrame { file: string; caption: string }

export const BOOK_PREFIX = "storybook/";
const DATE = /^\d{4}-\d{2}-\d{2}$/;
const FILE = /^[A-Za-z0-9._-]+\.png$/;

const notFound = () => new Response("Not found", { status: 404 });

export function bookKeyOk(env: Env, url: URL): boolean {
  return Boolean(env.STORYBOOK_KEY) && url.searchParams.get("key") === env.STORYBOOK_KEY;
}

async function json<T>(env: Env, key: string): Promise<T | null> {
  try {
    const obj = await env.SERIES.get(key);
    return obj ? ((await obj.json()) as T) : null;
  } catch (err) {
    console.error(`storybook: ${key} unreadable:`, err);
    return null;
  }
}

function page(title: string, body: string): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>${escapeHTML(title)}</title>
<style>
  :root { color-scheme: light dark; --bg: #faf8f4; --fg: #1c1b19; --muted: #6b675f; --line: #e2ddd3; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161514; --fg: #ece8e1; --muted: #9a958b; --line: #2e2c29; } }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 17px/1.5 Georgia, "Times New Roman", serif; }
  main { max-width: 820px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 1.6em; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 24px; }
  ol.days { list-style: none; padding: 0; margin: 0; }
  ol.days a { display: flex; justify-content: space-between; padding: 14px 4px; border-bottom: 1px solid var(--line); color: inherit; text-decoration: none; }
  ol.days span { color: var(--muted); }
  figure { margin: 0 0 28px; }
  img { display: block; width: 100%; height: auto; border-radius: 6px; background: #fff; }
  figcaption { color: var(--muted); font-size: 0.85em; text-align: center; margin-top: 6px; }
  nav { display: flex; justify-content: space-between; gap: 12px; margin: 8px 0 24px; }
  nav a { color: inherit; }
</style>
</head>
<body><main>${body}</main></body>
</html>`;
}

/** /storybook: the contents, or one day's frames with ?day=. */
export async function handleStorybook(env: Env, url: URL): Promise<Response> {
  if (!bookKeyOk(env, url)) return notFound();
  const key = encodeURIComponent(url.searchParams.get("key") ?? "");
  const days = ((await json<BookDay[]>(env, `${BOOK_PREFIX}days.json`)) ?? []).filter((d) => DATE.test(d.date)).sort((a, b) => a.date.localeCompare(b.date));
  const date = url.searchParams.get("day");
  if (date === null) {
    const items = days.map((d) => `<li><a href="?key=${key}&amp;day=${d.date}">Day ${d.day}<span>${d.date} &middot; ${d.frames} pages</span></a></li>`).join("");
    return htmlResponse(page("The Desk Story", `<h1>The Desk Story</h1><p class="sub">${days.length} days so far</p><ol class="days">${items || "<li>Nothing yet.</li>"}</ol>`), "no-store");
  }
  if (!DATE.test(date)) return notFound();
  const day = await json<{ date: string; day: number; frames: BookFrame[] }>(env, `${BOOK_PREFIX}days/${date}.json`);
  if (!day) return notFound();
  const i = days.findIndex((d) => d.date === date);
  const link = (d: BookDay | undefined, label: string) => (d ? `<a href="?key=${key}&amp;day=${d.date}">${label}</a>` : "<span></span>");
  const nav = `<nav>${link(days[i - 1], "&larr; Day before")}<a href="?key=${key}">Contents</a>${link(days[i + 1], "Next day &rarr;")}</nav>`;
  const frames = day.frames.filter((f) => FILE.test(f.file)).map((f, n) =>
    `<figure><img src="/storybook/frame?key=${key}&amp;day=${date}&amp;file=${encodeURIComponent(f.file)}" width="800" height="480" alt="Page ${n + 1}"${n > 1 ? ' loading="lazy"' : ""}><figcaption>${escapeHTML(f.caption)}</figcaption></figure>`).join("");
  return htmlResponse(page(`Day ${day.day}`, `${nav}<h1>Day ${day.day}</h1><p class="sub">${date}</p>${frames}${nav}`), "no-store");
}

/** /storybook/frame: one saved frame, streamed from R2. */
export async function handleStorybookFrame(env: Env, url: URL): Promise<Response> {
  if (!bookKeyOk(env, url)) return notFound();
  const date = url.searchParams.get("day") ?? "", file = url.searchParams.get("file") ?? "";
  if (!DATE.test(date) || !FILE.test(file)) return notFound();
  try {
    const obj = await env.SERIES.get(`${BOOK_PREFIX}frames/${date}/${file}`);
    if (!obj) return notFound();
    // A saved frame never changes; private, because the key is in the address.
    return new Response(obj.body, { headers: { "content-type": "image/png", "cache-control": "private, max-age=31536000, immutable" } });
  } catch (err) {
    console.error("storybook: frame read failed:", err);
    return new Response("Unavailable", { status: 503 });
  }
}
