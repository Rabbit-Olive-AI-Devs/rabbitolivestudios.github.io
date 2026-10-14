/**
 * Desk panel pages (third display: a mono E1001 on the home-office desk).
 *
 * /desk/test     — calibration card for judging type, rules and fills on the panel
 * /desk/aurebesh — Aurebesh word of the day, a family member's name on their
 *                  birthday, or a special day's word; ?date=YYYY-MM-DD,
 *                  ?word=HELLO, ?sheet for the check sheet
 * /desk/planet   — a real exoplanet each day from NASA's archive; ?date=YYYY-MM-DD
 * /desk/tars     — the owner's Macs and AI allowances; GET ?key= to view, POST to report
 * /desk/flamengo — the league table, the next match, the last result and recent form
 *
 * Every page here is a Worker-drawn 1-bit PNG inlined in one HTML document, with
 * no AI. The Aurebesh pages are pure functions of the date; the planet page reads
 * one small KV entry a day; TARS shows what the Macs last reported; the Flamengo board
 * keeps ESPN's public feeds in one KV entry (DECISIONS #66–#70).
 */

import type { Env } from "../types";
import { getPlanetOfDay } from "../planet";
import { ingestReport, readState, tarsView } from "../tars";
import { flamengoView, getFlamengo } from "../flamengo";
import type { FlaSources } from "../flamengo";
import { drawFlamengoPage } from "../hud/flamengo-page";
import { drawTarsPage } from "../hud/tars-page";
import type { ArchiveQuery } from "../planet";
import { drawPlanetPage } from "../hud/planet-page";
import { hudPageResponse } from "../hud/page";
import { drawTestCard } from "../hud/test-card";
import { drawAurebeshSheet } from "../hud/aurebesh";
import { getChicagoDateParts } from "../date-utils";
import { drawAurebeshPage } from "../hud/aurebesh-page";
import { getBirthdayToday } from "../birthday";
import { cleanWord, parseDateParam, pickWord, specialDay } from "../hud/aurebesh-words";
import type { Occasion } from "../hud/aurebesh-words";

export function handleDeskTest(): Promise<Response> {
  return hudPageResponse("Desk Panel Calibration", drawTestCard);
}

/**
 * What the day calls for instead of the daily word: a family member's name on
 * their birthday (with their age), or a fixed special day. Birthdays win.
 */
export function occasionFor(dateStr: string): Occasion | null {
  const [year, month, day] = dateStr.split("-").map(Number);
  const person = getBirthdayToday(month, day);
  const name = person ? cleanWord(person.key) : null;
  if (person && name) {
    return {
      word: name, tag: "HAPPY BIRTHDAY", group: "FAMILY",
      count: { label: "AGE", value: year - person.birthYear, unit: "YEARS" },
    };
  }
  return specialDay(dateStr);
}

/**
 * /desk/aurebesh — word of the day, or the day's occasion. Free test
 * parameters, no auth, no cost: `?date=YYYY-MM-DD`, `?word=HELLO` (which
 * overrides any occasion), and `?sheet` for the check sheet. Unusable input
 * falls back to today's date and the daily word.
 */
export function handleDeskAurebesh(url: URL): Promise<Response> {
  if (url.searchParams.has("sheet")) {
    return hudPageResponse("Aurebesh Check Sheet", drawAurebeshSheet);
  }
  const dateStr = parseDateParam(url.searchParams.get("date")) ?? getChicagoDateParts().dateStr;
  const override = cleanWord(url.searchParams.get("word"));
  const occasion = override ? null : occasionFor(dateStr);
  const word = override ?? occasion?.word ?? pickWord(dateStr);
  return hudPageResponse("Aurebesh Word of the Day", () => drawAurebeshPage(word, dateStr, occasion ?? undefined));
}

/**
 * /desk/planet — a real exoplanet each day. `?date=YYYY-MM-DD` previews another
 * day without touching the cache. `query` replaces NASA's archive in tests.
 */
export async function handleDeskPlanet(
  env: Env, url: URL, ctx?: ExecutionContext, query?: ArchiveQuery,
): Promise<Response> {
  const today = getChicagoDateParts().dateStr;
  const dateStr = parseDateParam(url.searchParams.get("date")) ?? today;
  const day = await getPlanetOfDay(env, dateStr, { ctx, query, store: dateStr === today });
  return hudPageResponse("Planet of the Day", () => drawPlanetPage(day, dateStr));
}

/** Largest report body accepted. A real one is about 3 KB. */
const MAX_REPORT_BYTES = 65_536;
const NOT_FOUND = () => new Response("Not found", { status: 404 });

/**
 * GET /desk/tars?key=… — the TARS page. It shows plans and spend, so with a key
 * configured anything but the right key gets a 404, as if the route were not
 * there. With no key configured (local development) the page is open.
 */
export async function handleDeskTars(env: Env, url: URL): Promise<Response> {
  if (env.TARS_KEY && url.searchParams.get("key") !== env.TARS_KEY) return NOT_FOUND();
  const now = new Date();
  const view = tarsView(await readState(env), now.getTime());
  return hudPageResponse("TARS", () => drawTarsPage(view, now));
}

/**
 * POST /desk/tars — a report from one of the owner's Macs, with the key as a
 * bearer token. Without a configured key nobody can post at all.
 */
export async function handleDeskTarsReport(request: Request, env: Env): Promise<Response> {
  if (!env.TARS_KEY || request.headers.get("Authorization") !== `Bearer ${env.TARS_KEY}`) return NOT_FOUND();
  const text = await request.text();
  if (text.length > MAX_REPORT_BYTES) return new Response("Report too large", { status: 413 });
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return new Response("Report is not JSON", { status: 400 });
  }
  const result = await ingestReport(env, body, Date.now());
  return new Response(result, { status: result === "invalid" ? 400 : 200 });
}

/**
 * /desk/flamengo — the Flamengo board. Draws from the cached entry and lets a
 * refresh finish in the background; with nothing cached and ESPN silent it
 * still draws, with its empty states. `sources` replaces ESPN in tests.
 */
export async function handleDeskFlamengo(env: Env, ctx?: ExecutionContext, sources?: FlaSources): Promise<Response> {
  const now = new Date();
  const view = flamengoView(await getFlamengo(env, { ctx, sources }), now.getTime());
  return hudPageResponse("Flamengo", () => drawFlamengoPage(view, now));
}
