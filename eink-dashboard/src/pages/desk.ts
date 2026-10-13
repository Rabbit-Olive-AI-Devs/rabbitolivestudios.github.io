/**
 * Desk panel pages (third display: a mono E1001 on the home-office desk).
 *
 * /desk/test     — calibration card for judging type, rules and fills on the panel
 * /desk/aurebesh — Aurebesh check sheet (all 26 letters)
 *
 * Every page here is a Worker-drawn 1-bit PNG inlined in one HTML document, a
 * pure function of the date: no KV, no AI, no cron (DECISIONS #66).
 */

import { hudPageResponse } from "../hud/page";
import { drawTestCard } from "../hud/test-card";
import { drawAurebeshSheet } from "../hud/aurebesh";

export function handleDeskTest(): Promise<Response> {
  return hudPageResponse("Desk Panel Calibration", drawTestCard);
}

/** /desk/aurebesh — all 26 letters, for checking against a chart. */
export function handleDeskAurebesh(_url: URL): Promise<Response> {
  return hudPageResponse("Aurebesh Check Sheet", drawAurebeshSheet);
}
