/**
 * Finding the most recent usable entry in a date-keyed cache family.
 *
 * The stale-image fallbacks (DECISIONS #57) originally rebuilt past cache keys
 * with `fact4CacheKey(date)` and friends. That embeds the *current* cache-key
 * version, so the moment anyone bumped `FACT4_CACHE_VERSION` the fallback began
 * searching for keys that had never existed — silently disabling itself exactly
 * when it was most needed, since CLAUDE.md requires a version bump after any
 * pipeline change and a fresh pipeline is what most often fails.
 *
 * Listing by prefix sidesteps that: it discovers whatever versions are actually
 * present, needs no hand-maintained list of old version strings, and costs one
 * KV list instead of one get per candidate day.
 */

import type { Env } from "./types";
import { shiftDateStr } from "./date-utils";
import { getAiBudgetBlock } from "./cache-guard";

const DATE_RE = /(\d{4}-\d{2}-\d{2})/;
const VERSION_RE = /:v(\d+)(?::|$)/;

export interface DatedKeyOpts {
  /** How many days back from `latestDate`, inclusive. Default 7. */
  maxDaysBack?: number;
  /** Extra filter, e.g. to separate `:bw` skylines from colour ones. */
  accept?: (keyName: string) => boolean;
}

/**
 * Pick the newest cache key at or before `latestDate` from a list of key names
 * that embed a `YYYY-MM-DD`.
 *
 * Newest date wins. When the same date exists under two cache-key versions
 * (the window around a bump), the higher version wins.
 */
export function pickMostRecentDated(
  keyNames: string[],
  latestDate: string,
  opts: DatedKeyOpts = {},
): string | null {
  const maxDaysBack = opts.maxDaysBack ?? 7;
  const earliest = shiftDateStr(latestDate, -(maxDaysBack - 1));

  let best: string | null = null;
  let bestDate = "";
  let bestVersion = -1;

  for (const name of keyNames) {
    if (opts.accept && !opts.accept(name)) continue;
    const m = DATE_RE.exec(name);
    if (!m) continue;
    const date = m[1];
    // String compare is valid ordering for zero-padded ISO dates.
    if (date > latestDate || date < earliest) continue;

    const vm = VERSION_RE.exec(name);
    const version = vm ? parseInt(vm[1], 10) : -1;

    if (date > bestDate || (date === bestDate && version > bestVersion)) {
      best = name;
      bestDate = date;
      bestVersion = version;
    }
  }
  return best;
}

/** List every key under `prefix`, following pagination. */
async function listKeys(env: Env, prefix: string): Promise<string[]> {
  const names: string[] = [];
  let cursor: string | undefined;
  // A cache family holds a handful of keys; the cap is a runaway guard only.
  for (let page = 0; page < 5; page++) {
    const res: any = await env.CACHE.list({ prefix, cursor });
    for (const k of res.keys) names.push(k.name);
    if (res.list_complete || !res.cursor) break;
    cursor = res.cursor;
  }
  return names;
}

/**
 * Find and read the most recent cached value at or before `latestDate`.
 * Returns the key alongside the value so callers can log which day they served.
 */
export async function findMostRecentCached(
  env: Env,
  prefix: string,
  latestDate: string,
  opts: DatedKeyOpts = {},
): Promise<{ key: string; value: string } | null> {
  const names = await listKeys(env, prefix);
  // Walk candidates newest-first so a corrupt or already-expired entry falls
  // through to the next one rather than failing the whole lookup.
  const remaining = new Set(names);
  for (let attempt = 0; attempt < 8; attempt++) {
    const key = pickMostRecentDated([...remaining], latestDate, opts);
    if (!key) return null;
    const value = await env.CACHE.get(key);
    if (value) return { key, value };
    remaining.delete(key); // listed but gone (raced its own expiry)
  }
  return null;
}

/**
 * Whether any usable entry exists for `dateStr` or the days before it.
 *
 * Used by the HTML wrapper pages so they never emit an <img> pointing at a
 * route that is already known to have nothing to serve.
 */
export async function hasRecentCached(
  env: Env,
  prefix: string,
  dateStr: string,
  opts: DatedKeyOpts = {},
): Promise<boolean> {
  const names = await listKeys(env, prefix);
  return pickMostRecentDated(names, dateStr, { maxDaysBack: 8, ...opts }) !== null;
}

/**
 * Whether an image route is certain to fail, so a wrapper page can render text
 * instead of an <img> that will break.
 *
 * Deliberately conservative: it reports "will fail" ONLY when AI generation is
 * blocked *and* there is nothing cached to fall back on. A cold cache on a
 * healthy day is not a failure — the route will simply generate the image.
 */
export async function imageUnavailable(
  env: Env,
  prefix: string,
  dateStr: string,
  opts: DatedKeyOpts = {},
): Promise<boolean> {
  if (!(await getAiBudgetBlock(env))) return false;
  return !(await hasRecentCached(env, prefix, dateStr, opts));
}
