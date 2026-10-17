/**
 * Failure alerting.
 *
 * Both August 2026 outages were found by looking at a physical panel — the
 * Worker degraded exactly as designed and said nothing for two days
 * (DECISIONS #59). The signal that would have caught either one on day one is
 * simply: *the day's images are not in the cache*.
 *
 * Delivery is Cloudflare Email Routing's `send_email` binding, so there is no
 * third-party service, API key or account involved. The address lives in the
 * ALERT_TO secret rather than in wrangler.toml, because this repo is public.
 */
import type { Env } from "./types";
import { getAiBudgetBlock } from "./cache-guard";
import { seriesProblems } from "./series/watch";
import { readState } from "./tars";
import { jarvisDrew } from "./art";

const ALERT_STATE_KEY = "alert:v1:last";

/** A quiet state still records a fingerprint, so recovery can be detected. */
export const OK_FINGERPRINT = "ok";

/** How long an unchanged problem stays quiet before it is repeated. */
export const ALERT_REPEAT_MS = 24 * 60 * 60 * 1000;

/**
 * Chicago hour before which the check is skipped.
 *
 * The 6h cron fires at 00:05/06:05/12:05/18:05 UTC. The 06:05 UTC firing lands
 * at 01:05 Chicago in CDT (00:05 in CST) — the same moment the daily image warm
 * is running — so checking then would alert on a cache that is still filling.
 * The other three firings land well clear of it.
 */
const EARLIEST_CHECK_HOUR = 3;

export interface AlertState {
  dateStr: string;
  /** Labels of the daily images that should exist for dateStr but do not. */
  missingImages: string[];
  aiBlocked: boolean;
  aiBlockSource?: string;
  /** Series problems from src/series/watch.ts (DECISIONS #71). */
  series?: string[];
  /** Machines whose TARS reports stopped: the jobs on them are not running (DECISIONS #90). */
  quietMachines?: string[];
  /** The day's pictures Workers AI drew because Jarvis delivered none (DECISIONS #89, #90). */
  jarvisMissed?: string[];
}

/** Machines that run the e-ink jobs and must keep reporting (the Mac mini runs the mailbox). */
export const WATCHED_MACHINES: readonly string[] = ["mini"];
/** Reports come every 30 minutes; this long without one means the machine is not running its jobs. */
export const QUIET_AFTER_MS = 2 * 60 * 60 * 1000;

/** Watched machines whose last TARS report is older than QUIET_AFTER_MS (or that never reported). */
export function quietMachines(lastReport: Record<string, number | undefined>, now: number): string[] {
  return WATCHED_MACHINES.filter((id) => now - (lastReport[id] ?? 0) > QUIET_AFTER_MS);
}

export interface AlertRecord {
  fingerprint: string;
  sentAt: number;
}

export type AlertDecision = { send: false } | { send: true; kind: AlertKind };
export type AlertKind = "problem" | "recovery";

export function shouldRunAlertCheck(chicagoHour: number): boolean {
  return chicagoHour >= EARLIEST_CHECK_HOUR;
}

/** Human-readable problem lines. Empty means healthy. */
export function describeProblems(state: AlertState): string[] {
  const problems: string[] = [];
  if (state.missingImages.length > 0) {
    problems.push(
      `Daily images missing for ${state.dateStr}: ${state.missingImages.join(", ")}`,
    );
  }
  if (state.aiBlocked) {
    const src = state.aiBlockSource ? ` (set by ${state.aiBlockSource})` : "";
    problems.push(`AI generation is blocked by the neuron budget guard${src}`);
  }
  problems.push(...(state.series ?? []));
  for (const id of state.quietMachines ?? []) {
    problems.push(
      `The ${id === "mini" ? "Mac mini" : id} has not reported for over 2 hours, so its jobs are not running ` +
      `(mailbox watcher, Jarvis's briefs, TARS). After a restart it waits for its password (FileVault): log in on it.`,
    );
  }
  if (state.jarvisMissed && state.jarvisMissed.length > 0) {
    problems.push(`Jarvis did not deliver ${state.dateStr}'s ${state.jarvisMissed.join(" and ")}; Workers AI drew ${state.jarvisMissed.length === 1 ? "it" : "them"}`);
  }
  return problems;
}

/**
 * Stable identity of the current problem state.
 *
 * Deliberately excludes the date: a fault that persists past Chicago midnight
 * is the same fault, and should not re-alert just because the date rolled. The
 * ALERT_REPEAT_MS reminder is what covers a long-running problem.
 */
export function alertFingerprint(state: AlertState): string {
  const parts: string[] = [];
  if (state.missingImages.length > 0) {
    parts.push(`missing:${[...state.missingImages].sort().join(",")}`);
  }
  if (state.aiBlocked) parts.push("ai:blocked");
  if (state.series && state.series.length > 0) parts.push(`series:${[...state.series].sort().join(";")}`);
  if (state.quietMachines && state.quietMachines.length > 0) parts.push(`quiet:${[...state.quietMachines].sort().join(",")}`);
  if (state.jarvisMissed && state.jarvisMissed.length > 0) parts.push(`jarvis:${[...state.jarvisMissed].sort().join(",")}`);
  return parts.length === 0 ? OK_FINGERPRINT : parts.join("|");
}

export function decideAlert(
  fingerprint: string,
  prev: AlertRecord | null,
  now: number,
  repeatMs: number = ALERT_REPEAT_MS,
): AlertDecision {
  if (fingerprint === OK_FINGERPRINT) {
    // Only worth an email if something was previously wrong.
    if (prev && prev.fingerprint !== OK_FINGERPRINT) return { send: true, kind: "recovery" };
    return { send: false };
  }
  if (!prev || prev.fingerprint !== fingerprint) return { send: true, kind: "problem" };
  if (now - prev.sentAt >= repeatMs) return { send: true, kind: "problem" };
  return { send: false };
}

export interface AlertEmailContext {
  dateStr: string;
  baseUrl: string;
}

/** Subject + plain-text body. Kept pure so the wording is testable. */
export function buildAlertEmail(
  kind: AlertKind,
  problems: string[],
  ctx: AlertEmailContext,
): { subject: string; text: string } {
  // A subject is a single header line; newlines from any interpolated value
  // would break it (and, on a raw-MIME path, allow header injection).
  const oneLine = (s: string) => s.replace(/[\r\n]+/g, " ").trim();

  if (kind === "recovery") {
    return {
      subject: oneLine(`[e-ink] Recovered — all checks passing (${ctx.dateStr})`),
      text: [
        `The e-ink dashboard has recovered. All daily images are cached for ${ctx.dateStr} and AI generation is not blocked.`,
        "",
        `Details: ${ctx.baseUrl}/health-detailed`,
      ].join("\n"),
    };
  }

  const count = problems.length;
  return {
    subject: oneLine(
      `[e-ink] ${count} problem${count === 1 ? "" : "s"}: ${problems.map(oneLine)[0]}`,
    ),
    text: [
      `The e-ink dashboard check found ${count} problem${count === 1 ? "" : "s"} on ${ctx.dateStr}:`,
      "",
      ...problems.map((p) => `  - ${oneLine(p)}`),
      "",
      "The panels keep showing the most recent cached image, so this is usually",
      "degraded rather than blank — but nothing new is being generated.",
      "",
      `Health:    ${ctx.baseUrl}/health-detailed`,
      `Runbook:   DECISIONS.md #57-#59 and INCIDENT-2026-08-20-neuron-budget-blowout.md`,
      "",
      "This repeats at most once every 24h while unchanged, and you will get a",
      "single follow-up when it clears.",
    ].join("\n"),
  };
}

// --- delivery ----------------------------------------------------------------

/** True when the binding and both addresses are configured. */
export function alertingConfigured(env: Env): boolean {
  return Boolean(env.SEND_EMAIL && env.ALERT_TO && env.ALERT_FROM);
}

export async function sendAlertEmail(
  env: Env,
  subject: string,
  text: string,
): Promise<void> {
  if (!alertingConfigured(env)) {
    throw new Error("alerting not configured (needs SEND_EMAIL binding + ALERT_TO/ALERT_FROM secrets)");
  }
  await env.SEND_EMAIL!.send({
    from: env.ALERT_FROM!,
    to: env.ALERT_TO!,
    subject,
    text,
  });
}

// --- orchestration -----------------------------------------------------------

export interface AlertCheckResult {
  checked: boolean;
  sent: boolean;
  fingerprint?: string;
  problems?: string[];
  reason?: string;
}

/**
 * Read the day's cache state, decide, and email if warranted.
 *
 * `imageKeys` maps a display label to the KV key that should hold that image
 * for `dateStr`; the caller owns key construction so this stays in step with
 * /health-detailed rather than duplicating birthday/skyline-mode logic.
 *
 * Never throws: alerting must not be able to break the cron that generates the
 * images it is watching.
 */
export async function runAlertCheck(
  env: Env,
  opts: {
    dateStr: string;
    chicagoHour: number;
    imageKeys: Record<string, string>;
    /** The day's pictures Jarvis is meant to draw ("moment" or "birthday", and "skyline"). */
    jarvisKinds?: ("moment" | "birthday" | "skyline")[];
    baseUrl: string;
    now?: number;
    force?: boolean;
  },
): Promise<AlertCheckResult> {
  const now = opts.now ?? Date.now();
  try {
    if (!opts.force && !shouldRunAlertCheck(opts.chicagoHour)) {
      return { checked: false, sent: false, reason: "too close to the daily warm" };
    }

    const labels = Object.keys(opts.imageKeys);
    const kinds = opts.jarvisKinds ?? [];
    const [present, block, series, tars, drew] = await Promise.all([
      Promise.all(labels.map((l) => env.CACHE.get(opts.imageKeys[l], "stream"))),
      getAiBudgetBlock(env),
      seriesProblems(env, new Date(now)),
      readState(env),
      Promise.all(kinds.map((k) => jarvisDrew(env, opts.dateStr, k))),
    ]);
    const lastReport = Object.fromEntries(Object.entries(tars.machines).map(([id, m]) => [id, m.receivedAt]));

    const state: AlertState = {
      dateStr: opts.dateStr,
      missingImages: labels.filter((_, i) => present[i] === null),
      aiBlocked: block !== null,
      aiBlockSource: block?.source,
      series,
      quietMachines: quietMachines(lastReport, now),
      // Only a picture that exists counts as "drawn by Workers AI"; a missing one is reported above.
      jarvisMissed: kinds.filter((k, i) => !drew[i] && present[labels.indexOf(k === "skyline" ? "skyline" : "color_moment")] !== null),
    };

    const problems = describeProblems(state);
    const fingerprint = alertFingerprint(state);

    let prev: AlertRecord | null = null;
    const rawPrev = await env.CACHE.get(ALERT_STATE_KEY);
    if (rawPrev) {
      try { prev = JSON.parse(rawPrev) as AlertRecord; } catch { /* treat as unseen */ }
    }

    const decision = decideAlert(fingerprint, prev, now);
    if (!decision.send) {
      return { checked: true, sent: false, fingerprint, problems, reason: "no change" };
    }

    const { subject, text } = buildAlertEmail(decision.kind, problems, {
      dateStr: opts.dateStr,
      baseUrl: opts.baseUrl,
    });
    await sendAlertEmail(env, subject, text);

    // Only recorded after a successful send, so a delivery failure retries on
    // the next run instead of being silently swallowed.
    await env.CACHE.put(
      ALERT_STATE_KEY,
      JSON.stringify({ fingerprint, sentAt: now } satisfies AlertRecord),
      { expirationTtl: 604800 },
    );
    console.log(`Alert: sent ${decision.kind} — ${subject}`);
    return { checked: true, sent: true, fingerprint, problems };
  } catch (err) {
    console.error("Alert check failed:", err);
    return { checked: true, sent: false, reason: String((err as Error)?.message ?? err) };
  }
}
