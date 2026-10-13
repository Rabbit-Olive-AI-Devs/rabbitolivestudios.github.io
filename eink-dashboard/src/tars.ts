/**
 * TARS: the desk panel's dashboard of the owner's computers and AI allowances.
 *
 * The Worker cannot see the computers, so each one runs a small reporter
 * (scripts/tars-report.sh) that posts its own numbers and, where CodexBar is
 * installed, a snapshot of AI usage. Everything is merged into one KV document,
 * `tars:v1:state`: one write per report (DECISIONS #62, #69).
 *
 * Reports come from outside, so nothing in them is trusted: every number is
 * bounded, every label is reduced to a short run of capitals, and nothing else
 * is kept. Account emails are dropped even when CodexBar has redacted them.
 */

import type { Env } from "./types";

// --- What is stored --------------------------------------------------------------------

export interface Machine {
  id: string;
  name: string;
  diskTotalGb: number;
  diskFreeGb: number;
  memUsedPct: number | null;
  uptimeSec: number | null;
}

export interface UsageWindow {
  kind: string;
  label: string;
  usedPercent: number;
  /** When the allowance refills, in ms since the epoch; null when unknown. */
  resetAt: number | null;
}

export interface Reading {
  id: string;
  name: string;
  plan: string;
  windows: UsageWindow[];
  /** Estimated 30-day spend at list prices, in US dollars. */
  costUsd: number | null;
}

export interface StoredReading extends Reading {
  /** When this reading was received, and which machine sent it. */
  at: number;
  from: string;
}

export interface TarsState {
  machines: Record<string, Machine & { receivedAt: number }>;
  /** The newest good reading of each provider, whichever machine sent it. */
  providers: Record<string, StoredReading>;
  /** 30-day spend as each machine reports it: costs[machineId][providerId]. */
  costs: Record<string, Record<string, number>>;
  /** Providers CodexBar is set up for but has never managed to read: id -> name. */
  unread: Record<string, string>;
}

export interface ParsedReport {
  machine: Machine;
  providers: Reading[];
  /** Providers CodexBar tried and could not read this time. */
  failed: { id: string; name: string }[];
}

export const TARS_STATE_KEY = "tars:v1:state";
const KV_TTL_SECONDS = 604800;
/** A reporter runs every 30 minutes; anything faster from one machine is a fault. */
const MIN_REPORT_GAP_MS = 5 * 60_000;
const MAX_MACHINES = 4;
const MAX_PROVIDERS = 12;
const MAX_WINDOWS = 6;

export function emptyState(): TarsState {
  return { machines: {}, providers: {}, costs: {}, unread: {} };
}

// --- Reading a report -------------------------------------------------------------------

const obj = (v: unknown): Record<string, unknown> | null =>
  (typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null);

const finite = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));

/** Capitals, digits and a little punctuation, at most `max` characters. */
function labelOf(v: unknown, max: number): string {
  if (typeof v !== "string") return "";
  return v.toUpperCase().replace(/\([^)]*\)/g, " ").replace(/[^A-Z0-9 .+\-]/g, "").replace(/\s+/g, " ").trim().slice(0, max).trim();
}

/** A provider or machine id: lower-case letters, digits and hyphens only. */
const idOf = (v: unknown): string | null => (typeof v === "string" && /^[a-z0-9-]{1,16}$/.test(v) ? v : null);

/** Names shown for providers whose own name does not fit or reads badly. */
const PROVIDER_NAMES: Readonly<Record<string, string>> = { museai: "MUSE", minimax: "MINIMAX" };

function readingOf(raw: unknown): { reading: Reading | null; id: string | null; name: string; failed: boolean } {
  const p = obj(raw);
  const id = p ? idOf(p.id) : null;
  if (!p || !id) return { reading: null, id: null, name: "", failed: false };
  const name = PROVIDER_NAMES[id] ?? (labelOf(p.name, 14) || id.toUpperCase());
  if (p.error) return { reading: null, id, name, failed: true };
  const windows = (Array.isArray(p.windows) ? p.windows : []).slice(0, MAX_WINDOWS).flatMap((w): UsageWindow[] => {
    const win = obj(w);
    const used = win ? finite(win.usedPercent) : null;
    if (!win || used === null) return [];
    const reset = typeof win.resetAt === "string" ? Date.parse(win.resetAt) : NaN;
    return [{
      kind: labelOf(win.kind, 30).toLowerCase(),
      label: labelOf(win.label, 12),
      usedPercent: clamp(Math.round(used), 0, 100),
      resetAt: Number.isFinite(reset) ? reset : null,
    }];
  });
  const cost = finite(obj(p.cost)?.last30DaysUSD);
  return {
    id,
    name,
    failed: false,
    reading: {
      id,
      name,
      plan: labelOf(obj(p.identity)?.plan, 12).replace(/^MUSE /, ""),
      windows,
      costUsd: cost === null ? null : clamp(cost, 0, 1_000_000),
    },
  };
}

/** A posted report, cleaned; null when it is not a report at all. */
export function parseReport(body: unknown): ParsedReport | null {
  const m = obj(obj(body)?.machine);
  const id = m ? idOf(m.id) : null;
  const name = m ? labelOf(m.name, 14) : "";
  const total = m ? finite(m.diskTotalGb) : null;
  const free = m ? finite(m.diskFreeGb) : null;
  if (!m || !id || !name || total === null || total <= 0 || free === null) return null;
  const diskTotalGb = clamp(Math.round(total), 1, 100_000);
  const mem = finite(m.memUsedPct);
  const up = finite(m.uptimeSec);

  const providers: Reading[] = [];
  const failed: { id: string; name: string }[] = [];
  const list = obj(obj(body)?.codexbar)?.providers;
  for (const raw of (Array.isArray(list) ? list : []).slice(0, MAX_PROVIDERS)) {
    const r = readingOf(raw);
    if (r.reading) providers.push(r.reading);
    else if (r.failed && r.id) failed.push({ id: r.id, name: r.name });
  }
  return {
    machine: {
      id,
      name,
      diskTotalGb,
      diskFreeGb: clamp(Math.round(free), 0, diskTotalGb),
      memUsedPct: mem === null ? null : clamp(Math.round(mem), 0, 100),
      uptimeSec: up === null ? null : clamp(Math.round(up), 0, 10 * 365 * 86400),
    },
    providers,
    failed,
  };
}

// --- Merging and storing ------------------------------------------------------------------

/** The state after a report received at `now`. Does not modify `state`. */
export function mergeReport(state: TarsState, report: ParsedReport, now: number): TarsState {
  const machines = { ...state.machines, [report.machine.id]: { ...report.machine, receivedAt: now } };
  // keep the most recently heard-from machines if there are ever too many
  const kept = Object.values(machines).sort((a, b) => b.receivedAt - a.receivedAt).slice(0, MAX_MACHINES);
  const providers = { ...state.providers };
  const mine: Record<string, number> = {};
  for (const r of report.providers) {
    providers[r.id] = { ...r, at: now, from: report.machine.id };
    if (r.costUsd !== null) mine[r.id] = r.costUsd;
  }
  const costs = { ...state.costs, [report.machine.id]: mine };
  const unread = { ...state.unread };
  for (const f of report.failed) if (!providers[f.id]) unread[f.id] = f.name;
  for (const id of Object.keys(providers)) delete unread[id];
  return {
    machines: Object.fromEntries(kept.map((m) => [m.id, m])),
    providers,
    costs: Object.fromEntries(Object.entries(costs).filter(([id]) => kept.some((m) => m.id === id))),
    unread: Object.fromEntries(Object.entries(unread).slice(0, MAX_PROVIDERS)),
  };
}

export async function readState(env: Env): Promise<TarsState> {
  try {
    const raw = await env.CACHE.get(TARS_STATE_KEY);
    const s = raw ? (JSON.parse(raw) as TarsState) : null;
    return s && obj(s.machines) && obj(s.providers) && obj(s.costs) ? { ...s, unread: obj(s.unread) ? s.unread : {} } : emptyState();
  } catch {
    return emptyState();
  }
}

export type IngestResult = "stored" | "too-soon" | "invalid";

/** Store a posted report. One KV write, or none when it is refused. */
export async function ingestReport(env: Env, body: unknown, now: number): Promise<IngestResult> {
  const report = parseReport(body);
  if (!report) return "invalid";
  const state = await readState(env);
  const last = state.machines[report.machine.id]?.receivedAt ?? 0;
  if (now - last < MIN_REPORT_GAP_MS) return "too-soon";
  await env.CACHE.put(TARS_STATE_KEY, JSON.stringify(mergeReport(state, report, now)), { expirationTtl: KV_TTL_SECONDS });
  return "stored";
}

// --- What the page shows ---------------------------------------------------------------------

export interface MachineView {
  name: string;
  awake: boolean;
  diskFreeGb: number;
  diskUsedPct: number;
  /** Null for a machine that is asleep: its last memory figure is not current. */
  memUsedPct: number | null;
  /** Uptime when awake; the time it was last seen when not. */
  since: string;
}

export interface ProviderView {
  id: string;
  name: string;
  plan: string;
  /** Percent used. `undefined` = this provider has no such limit; `null` = no recent reading. */
  session: number | null | undefined;
  week: number | null | undefined;
  reset: string;
  note: string;
  low: boolean;
}

export interface TarsView {
  machines: MachineView[];
  providers: ProviderView[];
  alerts: string[];
  quip: string[];
  joke: boolean;
  next: { numeral: string; text: string; what: string } | null;
  spend: string;
  /** When the newest report arrived, as HH:MM in Chicago; "" before the first one. */
  reported: string;
}

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
/** Reports come every 30 minutes; two missed in a row means the machine is not running. */
const AWAKE_WITHIN_MS = 75 * MINUTE;
/** A provider reading older than this is no longer shown as current. */
const FRESH_WITHIN_MS = 3 * HOUR;
const LOW_AT_PERCENT = 90;
const DISK_ALERT_PERCENT = 85;
const MAX_ROWS = 5;
const MAX_MACHINE_ROWS = 2;
const MAX_ALERTS = 4;
const PROVIDER_ORDER = ["codex", "claude", "cursor", "minimax", "museai"];
/** Providers whose spend CodexBar estimates from each machine's own logs, so machines add up. */
const SPEND_PER_MACHINE = new Set(["codex", "claude"]);

const two = (n: number) => String(n).padStart(2, "0");

/** "6D 08H", "2H 42M", "42M", or "NOW" for a span that has already passed. */
export function formatSpan(ms: number): string {
  if (ms <= 0) return "NOW";
  const minutes = Math.floor(ms / MINUTE);
  if (ms >= DAY) return `${Math.floor(ms / DAY)}D ${two(Math.floor((ms % DAY) / HOUR))}H`;
  if (ms >= HOUR) return `${Math.floor(ms / HOUR)}H ${two(minutes % 60)}M`;
  return `${Math.max(1, minutes)}M`;
}

const chicago = (ms: number, opts: Intl.DateTimeFormatOptions) =>
  new Intl.DateTimeFormat("en-US", { timeZone: "America/Chicago", ...opts }).format(new Date(ms));

/** Clock time in Chicago, with the weekday in front when it was not within the last day. */
function stamp(ms: number, now: number): string {
  const time = chicago(ms, { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
  return now - ms >= DAY ? `${chicago(ms, { weekday: "short" }).toUpperCase()} ${time}` : time;
}

const titleCase = (s: string) => s.charAt(0) + s.slice(1).toLowerCase();

function hash(s: string): number {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return h;
}

const CALM: readonly string[] = [
  "All systems nominal.",
  "Nothing needs you right now.",
  "Plenty of room to run.",
  "Every limit has slack in it.",
];
const CALM_JOKES: readonly string[] = [
  "I have no complaints. Yet.",
  "Humor setting holding at 75.",
  "Quiet. I do not trust it.",
];

/** Everything the TARS page draws, worked out from the stored state and the time. */
export function tarsView(state: TarsState, now: number): TarsView {
  const alerts: string[] = [];
  const quip: string[] = [];
  let joke = false;

  const sorted = Object.values(state.machines).sort((a, b) => a.id.localeCompare(b.id));
  const machines = sorted.slice(0, MAX_MACHINE_ROWS).map((m): MachineView => {
    const awake = now - m.receivedAt < AWAKE_WITHIN_MS;
    const diskUsedPct = Math.round(((m.diskTotalGb - m.diskFreeGb) / m.diskTotalGb) * 100);
    return {
      name: m.name,
      awake,
      diskFreeGb: m.diskFreeGb,
      diskUsedPct,
      memUsedPct: awake ? m.memUsedPct : null,
      since: awake ? (m.uptimeSec === null ? "UNKNOWN" : formatSpan(m.uptimeSec * 1000)) : stamp(m.receivedAt, now),
    };
  });

  const rank = (id: string) => { const i = PROVIDER_ORDER.indexOf(id); return i < 0 ? PROVIDER_ORDER.length : i; };
  // A provider that has never answered is carried as a reading with no windows and no date.
  const never: StoredReading[] = Object.entries(state.unread)
    .map(([id, name]) => ({ id, name, plan: "", windows: [], costUsd: null, at: 0, from: "" }));
  const readings = [...Object.values(state.providers), ...never]
    .sort((a, b) => rank(a.id) - rank(b.id) || a.id.localeCompare(b.id))
    .slice(0, MAX_ROWS);

  let next: { at: number; what: string } | null = null;
  const stale: string[] = [];
  const spent: { name: string; until: number | null }[] = [];
  const nearly: string[] = [];

  const providers = readings.map((r): ProviderView => {
    const weekly = r.windows.find((w) => w.kind === "weekly" || /WEEK/.test(w.label));
    const session = r.windows.find((w) => w.kind === "session" && w !== weekly);
    if (now - r.at >= FRESH_WITHIN_MS) {
      stale.push(r.name);
      alerts.push(`${r.name} NO DATA`);
      return {
        id: r.id, name: r.name, plan: r.plan,
        // with no reading at all we cannot know which limits it has, so both show as missing
        session: session || r.at === 0 ? null : undefined, week: weekly || r.at === 0 ? null : undefined,
        reset: "UNKNOWN", note: r.at === 0 ? "NEVER READ" : `LAST GOOD ${stamp(r.at, now)}`, low: false,
      };
    }
    const shown = [session, weekly].filter((w): w is UsageWindow => !!w);
    for (const w of shown) {
      if (w.usedPercent >= LOW_AT_PERCENT) alerts.push(`${r.name} ${w === weekly ? "WEEK" : "SESSION"} ${w.usedPercent}%`);
    }
    const worst = Math.max(0, ...shown.map((w) => w.usedPercent));
    const resets = shown.map((w) => w.resetAt).filter((t): t is number => t !== null && t > now).sort((a, b) => a - b);
    if (worst >= 100) spent.push({ name: r.name, until: resets[0] ?? null });
    else if (worst >= LOW_AT_PERCENT) nearly.push(r.name);
    if (resets.length && (!next || resets[0] < next.at)) next = { at: resets[0], what: r.name };
    const weekReset = weekly?.resetAt ?? null;
    return {
      id: r.id, name: r.name, plan: r.plan,
      session: session?.usedPercent, week: weekly?.usedPercent,
      reset: resets.length ? formatSpan(resets[0] - now) : "UNKNOWN",
      note: session && weekReset !== null && weekReset > now && weekReset !== resets[0] ? `WEEK ${formatSpan(weekReset - now)}` : "",
      low: worst >= LOW_AT_PERCENT,
    };
  });

  for (const m of machines) {
    if (m.diskUsedPct >= DISK_ALERT_PERCENT) alerts.push(`${m.name} DISK ${m.diskUsedPct}%`);
  }

  // What TARS says: the most pressing things first, in plain short sentences.
  for (const s of spent) {
    const day = s.until === null ? "" : chicago(s.until, { weekday: "long" });
    quip.push(day ? `${titleCase(s.name)} is out until ${day}.` : `${titleCase(s.name)} is used up.`);
  }
  for (const name of nearly) quip.push(`${titleCase(name)} is nearly spent.`);
  for (const name of stale) quip.push(`I cannot reach ${titleCase(name)}.`);
  for (const m of machines) {
    if (m.diskUsedPct >= DISK_ALERT_PERCENT) quip.push(`${titleCase(m.name)} is short of room.`);
  }
  const asleep = machines.filter((m) => !m.awake);
  if (asleep.length && quip.length < 3) {
    quip.push(`The ${titleCase(asleep[0].name)} is asleep.`);
    if (quip.length < 3) { quip.push("Lucky machine."); joke = true; }
  }
  const seed = hash(chicago(now, { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", hourCycle: "h23" }));
  if (sorted.length === 0) {
    quip.push("Waiting for the first report.");
  } else if (quip.length === 0) {
    quip.push(CALM[seed % CALM.length]);
    if (seed % 3 === 0) { quip.push(CALM_JOKES[seed % CALM_JOKES.length]); joke = true; }
  } else if (quip.length < 3 && spent.length + nearly.length + stale.length < readings.length && !joke) {
    quip.push("Everything else has room.");
  }

  // Spend: per-machine estimates add up; an account-wide figure is counted once.
  let dollars = 0;
  const providerIds = new Set(Object.values(state.costs).flatMap((c) => Object.keys(c)));
  for (const id of providerIds) {
    const values = Object.values(state.costs).map((c) => c[id]).filter((v): v is number => typeof v === "number");
    dollars += SPEND_PER_MACHINE.has(id) ? values.reduce((a, b) => a + b, 0) : Math.max(0, ...values);
  }

  const newest = Math.max(0, ...sorted.map((m) => m.receivedAt));
  const upcoming = next as { at: number; what: string } | null;
  const span = upcoming ? formatSpan(upcoming.at - now) : "";
  return {
    machines,
    providers,
    alerts: alerts.slice(0, MAX_ALERTS),
    quip: quip.slice(0, 3).map((l) => l.slice(0, 31)),
    joke,
    next: upcoming ? { numeral: span.match(/\d+/)?.[0] ?? "0", text: span, what: upcoming.what } : null,
    spend: `$${Math.round(dollars).toLocaleString("en-US")}`,
    reported: newest ? chicago(newest, { hour: "2-digit", minute: "2-digit", hourCycle: "h23" }) : "",
  };
}
