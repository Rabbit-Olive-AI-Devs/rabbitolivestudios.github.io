/**
 * Desk panel pages (third display: a mono E1001 on the home-office desk).
 *
 * /desk/test     — calibration card for judging type, rules and fills on the panel
 * /desk/aurebesh — Aurebesh word of the day, a family member's name on their
 *                  birthday, or a special day's word; ?date=YYYY-MM-DD,
 *                  ?word=HELLO, ?sheet for the check sheet
 * /desk/planet   — a real exoplanet each day from NASA's archive; ?date=YYYY-MM-DD
 *
 * Every page here is a Worker-drawn 1-bit PNG inlined in one HTML document, with
 * no AI. The Aurebesh pages are pure functions of the date; the planet page reads
 * one small KV entry a day (DECISIONS #66, #67, #68).
 */

import type { Env } from "../types";
import { getPlanetOfDay } from "../planet";
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
