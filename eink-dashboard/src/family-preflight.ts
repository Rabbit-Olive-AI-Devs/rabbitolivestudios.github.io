/**
 * Nightly family-picture preflight (DECISIONS #106).
 *
 * Workers Free allows five cron triggers for the whole account and they are all in use,
 * so this rides the existing periodic cron. That expression also fires at 02:05 and 03:05 UTC;
 * only the fire whose Chicago clock reads 21:05 runs. The other one returns without doing
 * any work. Tomorrow is ready when its ordinary upload is in R2, or when its own staged
 * manifest is unused, unclaimed, and its photo hash matches. A spare in the pool for some
 * other day does not count.
 */

import type { Env } from "./types";
import { shiftDateStr } from "./date-utils";
import { fetchWithTimeout } from "./fetch-timeout";
import { sendAlertEmail } from "./alert";
import { listStagedManifests, sha256Hex, stagedPhotoKey } from "./family-staged";

/** Approved Slack mention for the preflight request. Not an email address. */
export const FAMILY_PREFLIGHT_MENTION = "<@U0C7QQZUTJQ>";

const PREFLIGHT_KEY = (date: string) => `family:preflight:v1:${date}`;
const PERIODIC_HOURS = new Set([0, 6, 12, 18]);

export function chicagoHourMinute(epochMs: number): { hour: number; minute: number } {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(new Date(epochMs));
  const hour = parseInt(parts.find((p) => p.type === "hour")!.value, 10) % 24;
  const minute = parseInt(parts.find((p) => p.type === "minute")!.value, 10);
  return { hour, minute };
}

export function chicagoDateStr(epochMs: number): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date(epochMs));
  const year = parts.find((p) => p.type === "year")!.value;
  const month = parts.find((p) => p.type === "month")!.value;
  const day = parts.find((p) => p.type === "day")!.value;
  return `${year}-${month}-${day}`;
}

/** True only at 21:05 America/Chicago, CDT or CST. */
export function isFamilyPreflightInstant(epochMs: number): boolean {
  const { hour, minute } = chicagoHourMinute(epochMs);
  return hour === 21 && minute === 5;
}

/**
 * What the extended periodic cron should do for one fire.
 * `preflight` is 21:05 Chicago. `noop` is the other new UTC hour. `refresh` is an original fire.
 */
export function periodicFireAction(epochMs: number): "preflight" | "refresh" | "noop" {
  if (isFamilyPreflightInstant(epochMs)) return "preflight";
  const when = new Date(epochMs);
  if (when.getUTCMinutes() === 5 && PERIODIC_HOURS.has(when.getUTCHours())) return "refresh";
  if (when.getUTCMinutes() === 5 && (when.getUTCHours() === 2 || when.getUTCHours() === 3)) return "noop";
  return "refresh";
}

export function familyPreflightMessage(date: string, missing: string, unused: number, healthUrl: string): string {
  return `${FAMILY_PREFLIGHT_MENTION} REQUEST [eink-family-preflight] Family picture for ${date} is not ready at 21:05 CT: ${missing}; staged unused: ${unused}. Health: ${healthUrl}`;
}

export interface FamilyReadiness {
  ready: boolean;
  missing: string;
  unused: number;
}

/** Tomorrow's ordinary pair, or tomorrow's own staged manifest. Other unused entries are only counted. */
export async function familySourceReady(env: Env, date: string): Promise<FamilyReadiness> {
  const [datedMeta, datedPhoto, records] = await Promise.all([
    env.PHOTOS.head(`family/${date}/photo.json`),
    env.PHOTOS.head(`family/${date}/photo.jpg`),
    listStagedManifests(env),
  ]);
  const unused = records.filter((r) => !r.manifest.used && r.manifest.claimFor === null).length;
  if (datedMeta && datedPhoto) return { ready: true, missing: "", unused };

  const own = records.find((r) => r.manifest.date === date) ?? null;
  let stagedNote = "";
  if (!own) stagedNote = `family/staged/${date}/photo.json`;
  else if (own.manifest.used) stagedNote = `family/staged/${date}/photo.json (used)`;
  else if (own.manifest.claimFor) stagedNote = `family/staged/${date}/photo.json (claimed)`;
  else {
    const photo = await env.PHOTOS.get(stagedPhotoKey(date));
    const bytes = photo ? new Uint8Array(await photo.arrayBuffer()) : null;
    const hashOk = !!bytes && (await sha256Hex(bytes)) === own.manifest.photoSha256;
    if (hashOk) return { ready: true, missing: "", unused };
    stagedNote = `family/staged/${date}/photo.jpg (hash mismatch)`;
  }

  const parts: string[] = [];
  if (!datedMeta) parts.push(`family/${date}/photo.json`);
  if (!datedPhoto) parts.push(`family/${date}/photo.jpg`);
  if (stagedNote) parts.push(stagedNote);
  return { ready: false, missing: parts.join(", "), unused };
}

export interface PreflightResult {
  posted: boolean;
  skipped: string;
}

/**
 * Post at most one Slack message for a date. No webhook: log and skip, and do not email.
 * Email is only the fallback when the Slack post itself fails.
 */
export async function runFamilyPreflight(env: Env, epochMs: number, baseUrl: string): Promise<PreflightResult> {
  if (!isFamilyPreflightInstant(epochMs)) return { posted: false, skipped: "not-2105" };
  const date = shiftDateStr(chicagoDateStr(epochMs), 1);
  const status = await familySourceReady(env, date);
  if (status.ready) return { posted: false, skipped: "ready" };

  const marker = PREFLIGHT_KEY(date);
  if (await env.CACHE.get(marker)) return { posted: false, skipped: "deduped" };

  const text = familyPreflightMessage(date, status.missing, status.unused, `${baseUrl}/health-detailed`);
  const webhook = env.JARVIS_SLACK_WEBHOOK;
  if (!webhook) {
    console.log(`Family preflight: JARVIS_SLACK_WEBHOOK unset; skipping post for ${date}`);
    return { posted: false, skipped: "no-webhook" };
  }

  let slackOk = false;
  try {
    const res = await fetchWithTimeout(webhook, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    }, 10_000);
    slackOk = res.ok;
    if (!slackOk) console.error(`Family preflight: Slack post failed (HTTP ${res.status})`);
  } catch {
    console.error("Family preflight: Slack post failed");
  }

  if (!slackOk) {
    try {
      await sendAlertEmail(env, `[e-ink] Family picture for ${date} is not ready`, text);
    } catch (err) {
      console.error("Family preflight: email fallback failed:", err instanceof Error ? err.message : "send failed");
      return { posted: false, skipped: "delivery-failed" };
    }
  }

  await env.CACHE.put(marker, "sent", { expirationTtl: 604800 });
  return { posted: true, skipped: slackOk ? "slack" : "email" };
}
