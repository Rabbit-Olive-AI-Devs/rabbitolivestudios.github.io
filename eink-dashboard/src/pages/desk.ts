/**
 * Desk panel pages (third display: a mono E1001 on the home-office desk).
 *
 * /desk/test — calibration card for judging type, rules and fills on the panel
 *
 * Every page here is a Worker-drawn 1-bit PNG inlined in one HTML document, a
 * pure function of the date: no KV, no AI, no cron (DECISIONS #66).
 */

import { hudPageResponse } from "../hud/page";
import { drawTestCard } from "../hud/test-card";

export function handleDeskTest(): Promise<Response> {
  return hudPageResponse("Desk Panel Calibration", drawTestCard);
}
