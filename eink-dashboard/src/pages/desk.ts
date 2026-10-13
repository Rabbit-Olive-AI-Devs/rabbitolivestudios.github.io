/**
 * Desk panel pages (third display: a mono E1001 on the home-office desk).
 *
 * /desk/test     — calibration card for judging type, rules and fills on the panel
 * /desk/aurebesh — Aurebesh word of the day; ?date=YYYY-MM-DD, ?word=HELLO,
 *                  ?sheet for the 26-letter check sheet
 *
 * Every page here is a Worker-drawn 1-bit PNG inlined in one HTML document, a
 * pure function of the date: no KV, no AI, no cron (DECISIONS #66).
 */

import { hudPageResponse } from "../hud/page";
import { drawTestCard } from "../hud/test-card";
import { drawAurebeshSheet } from "../hud/aurebesh";
import { getChicagoDateParts } from "../date-utils";
import { drawAurebeshPage } from "../hud/aurebesh-page";
import { cleanWord, parseDateParam, pickWord } from "../hud/aurebesh-words";

export function handleDeskTest(): Promise<Response> {
  return hudPageResponse("Desk Panel Calibration", drawTestCard);
}

/**
 * /desk/aurebesh — word of the day. Free test parameters, no auth, no cost:
 * `?date=YYYY-MM-DD`, `?word=HELLO`, and `?sheet` for the 26-letter check sheet.
 * Unusable input falls back to today's date and the daily word.
 */
export function handleDeskAurebesh(url: URL): Promise<Response> {
  if (url.searchParams.has("sheet")) {
    return hudPageResponse("Aurebesh Check Sheet", drawAurebeshSheet);
  }
  const dateStr = parseDateParam(url.searchParams.get("date")) ?? getChicagoDateParts().dateStr;
  const word = cleanWord(url.searchParams.get("word")) ?? pickWord(dateStr);
  return hudPageResponse("Aurebesh Word of the Day", () => drawAurebeshPage(word, dateStr));
}
