/** Phone-friendly Long Drift reader. Uses only the existing approved-and-aired gate (#94). */
import { escapeHTML } from "../escape";
import { htmlResponse } from "../response";
import { loadSchedule } from "../pages/series";
import { airedPages } from "./aired";
import { pagePNG } from "./proof";
import type { AiredPage } from "./aired";
import type { Env } from "../types";

const ROOT = "/series/book";
const href = (p: AiredPage) => `${ROOT}?p=${encodeURIComponent(p.path)}`;
const chapterPath = (p: AiredPage) => p.path.split("/").slice(0, 2).join("/");
const chapterHref = (p: AiredPage) => `${ROOT}?c=${encodeURIComponent(chapterPath(p))}`;
const notFound = () => new Response("Not found", { status: 404 });
const seasonName = (n: number) => n === 0 ? "Prologue" : `Season ${n}`;
const sceneName = (p: AiredPage) => p.scene === null ? "Teaser" : `Scene ${p.scene}`;

function document(body: string): string {
  return `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow,noarchive">
<meta name="color-scheme" content="light dark"><title>The Long Drift — aired chapters</title>
<style>
:root{color-scheme:light dark;--bg:#faf8f4;--fg:#1c1b19;--muted:#6b675f;--line:#e2ddd3}
@media(prefers-color-scheme:dark){:root{--bg:#161514;--fg:#ece8e1;--muted:#aaa598;--line:#39352e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.5 Georgia,"Times New Roman",serif}
main{max-width:820px;margin:auto;padding:24px 16px 64px}h1{font-size:1.6em;line-height:1.25;margin:0 0 4px}
h2{font-size:1.25rem;margin:28px 0 10px}.sub,figcaption,small{color:var(--muted)}.sub{margin:0 0 24px}
a{color:inherit;text-underline-offset:4px}a:focus-visible{outline:3px solid currentColor;outline-offset:4px}
ol{list-style:none;padding:0;margin:0}ol a{display:flex;justify-content:space-between;gap:16px;padding:14px 4px;border-bottom:1px solid var(--line);text-decoration:none}
small{white-space:nowrap}nav{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:8px 0 24px;min-height:44px}nav a{padding:8px 0}
figure{margin:0 0 28px;scroll-margin-top:16px}img{display:block;width:100%;height:auto;background:#fff;border-radius:6px}figcaption{text-align:center;font-size:.85em;margin-top:6px}
@media(max-width:420px){ol a{flex-wrap:wrap;gap:4px 12px}nav{font-size:.85em;gap:8px}}
.open{display:block;text-align:center;font-size:.9rem;margin:12px 0}.eyebrow{font:12px/1.5 system-ui,sans-serif;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
</style></head><body><main>${body}</main></body></html>`;
}

function response(body: string, status = 200): Response {
  const r = htmlResponse(document(body), "no-store");
  r.headers.set("X-Robots-Tag", "noindex, nofollow, noarchive");
  return new Response(r.body, { status, headers: r.headers });
}

/** Storybook contents and scrolling chapters; retain old single-scene links. Preview times and keys are ignored. */
export async function handleSeriesBook(env: Env, url: URL, now = new Date()): Promise<Response> {
  const schedule = await loadSchedule(env.SERIES);
  if (!schedule) return response('<h1>The Long Drift</h1><p>The archive is temporarily unavailable. Please try again later.</p>', 503);
  const pages = airedPages(schedule, now);
  const groups = new Map<string, AiredPage[]>();
  for (const p of pages) {
    const key = chapterPath(p);
    groups.set(key, [...(groups.get(key) ?? []), p]);
  }
  const chapters = [...groups.values()];
  const chapter = url.searchParams.get("c");
  if (chapter !== null) {
    const ps = groups.get(chapter);
    if (!ps) return notFound();
    const p = ps[0];
    const i = chapters.indexOf(ps);
    const nav = `<nav aria-label="Chapter navigation">${i > 0 ? `<a rel="prev" href="${chapterHref(chapters[i - 1][0])}">← Chapter before</a>` : "<span></span>"}<a href="${ROOT}">Contents</a>${i + 1 < chapters.length ? `<a rel="next" href="${chapterHref(chapters[i + 1][0])}">Next chapter →</a>` : "<span></span>"}</nav>`;
    const frames = ps.map((page, n) => {
      const image = `${ROOT}/image?p=${encodeURIComponent(page.path)}`;
      return `<figure id="${page.scene === null ? "teaser" : `scene-${page.scene}`}"><img src="${image}" width="800" height="480" alt="${escapeHTML(page.label)}"${n > 1 ? ' loading="lazy"' : ""}><figcaption>${escapeHTML(page.label)} · <a href="${image}" target="_blank" rel="noopener">Open full-size page</a></figcaption></figure>`;
    }).join("");
    return response(`${nav}<h1>${seasonName(p.season)} · Chapter ${p.chapter}</h1><p class="sub">The Long Drift · ${ps.length} aired ${ps.length === 1 ? "page" : "pages"}</p>${frames}${nav}`);
  }
  const selected = url.searchParams.get("p");
  if (selected === null) {
    const items = chapters.map((ps) => {
      const p = ps[0];
      const scenes = ps.filter((x) => x.scene !== null).length;
      return `<li><a href="${chapterHref(p)}"><span>${seasonName(p.season)} · Chapter ${p.chapter}</span><small>${scenes}/25 scenes${ps.some((x) => x.scene === null) ? " + teaser" : ""}</small></a></li>`;
    }).join("");
    return response(`<h1>The Long Drift</h1><p class="sub">${chapters.length} ${chapters.length === 1 ? "chapter" : "chapters"} so far · ${pages.length} aired ${pages.length === 1 ? "page" : "pages"}</p>${items ? `<ol>${items}</ol>` : '<p>No scenes have aired yet.</p>'}<p class="sub" style="margin-top:24px">Only scenes that have already aired appear here. Open a chapter and scroll to read; refresh for newly aired pages.</p>`);
  }
  const i = pages.findIndex((p) => p.path === selected);
  if (i < 0) return notFound();
  const p = pages[i];
  const image = `${ROOT}/image?p=${encodeURIComponent(p.path)}`;
  const nav = `<nav aria-label="Scene navigation">${i > 0 ? `<a rel="prev" href="${href(pages[i - 1])}">← Previous</a>` : "<span></span>"}<a href="${ROOT}">Contents</a>${i + 1 < pages.length ? `<a rel="next" href="${href(pages[i + 1])}">Next →</a>` : "<span></span>"}</nav>`;
  return response(`${nav}<h1>${seasonName(p.season)} · Chapter ${p.chapter}</h1><p class="sub">${sceneName(p)} · The Long Drift</p><figure><img src="${image}" width="800" height="480" alt="${escapeHTML(p.label)}"><figcaption>${escapeHTML(p.label)}</figcaption></figure><a class="open" href="${image}" target="_blank" rel="noopener">Open full-size page</a>${nav}<p class="sub">Exactly the finished panel artwork, in full colour. No unaired scenes.</p>`);
}

/** The exact PNG already embedded in an aired panel page. No drawing, AI or storage writes. */
export async function handleSeriesBookImage(env: Env, url: URL, now = new Date()): Promise<Response> {
  const schedule = await loadSchedule(env.SERIES);
  const p = schedule ? airedPages(schedule, now).find((x) => x.path === url.searchParams.get("p")) : undefined;
  if (!p) return notFound();
  try {
    const obj = await env.SERIES.get(p.key);
    const png = obj ? pagePNG(await obj.text()) : null;
    if (png) return new Response(png, { headers: { "Content-Type": "image/png", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "X-Robots-Tag": "noindex, nofollow, noarchive" } });
  } catch (err) {
    console.error("series book: aired page could not be loaded", err);
  }
  return new Response("This aired page is temporarily unavailable. Please try again later.", { status: 503, headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" } });
}
