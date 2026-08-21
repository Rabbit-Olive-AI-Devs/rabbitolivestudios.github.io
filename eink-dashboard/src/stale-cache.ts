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
 *
 * The list result is then memoised in KV (DECISIONS #62). Listing per request
 * was measured at ~40 operations per hour of outage, against a free-tier cap of
 * 1,000 lists/day — so a multi-day outage would have exhausted the quota the
 * fallback itself depends on, and the panels would have returned to broken
 * images on day two: the safety net failing in exactly the situation it exists
 * for. The index makes an outage cost a handful of lists per day instead.
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

/** Namespace for the memoised key listings. Never collides with a cache family's own prefix. */
export const STALE_INDEX_PREFIX = "stale-idx:v1:";

/**
 * Six hours: long enough to make an outage cost a handful of lists a day, short
 * enough that the index can never drift far from entries that live for 7 days.
 * Correctness does not depend on it — every consumer verifies before trusting.
 */
const INDEX_TTL_SECONDS = 6 * 60 * 60;

/**
 * Key names under `prefix`, from the KV-memoised index when one is warm.
 *
 * `fresh` tells the caller whether these came from a real list, so it can decide
 * whether a miss is worth re-listing for.
 */
async function listKeysCached(
  env: Env,
  prefix: string,
  forceFresh = false,
): Promise<{ names: string[]; fresh: boolean }> {
  const indexKey = STALE_INDEX_PREFIX + prefix;

  if (!forceFresh) {
    try {
      const raw = await env.CACHE.get(indexKey);
      if (raw) {
        const names = JSON.parse(raw);
        if (Array.isArray(names)) return { names, fresh: false };
      }
    } catch { /* unreadable or corrupt index — fall through to a real list */ }
  }

  const names = await listKeys(env, prefix);
  try {
    await env.CACHE.put(indexKey, JSON.stringify(names), { expirationTtl: INDEX_TTL_SECONDS });
  } catch {
    // The index is an optimisation. A failed write (quota, transient) must never
    // be the reason a stale-image lookup fails.
  }
  return { names, fresh: true };
}

/**
 * Walk candidates newest-first until `read` yields something, re-listing once if
 * a memoised index turned out to be entirely stale.
 *
 * `read` is what distinguishes existence checks from value reads, so both share
 * one implementation of the candidate walk and the staleness retry.
 */
async function resolveMostRecent<T>(
  env: Env,
  prefix: string,
  latestDate: string,
  opts: DatedKeyOpts,
  read: (key: string) => Promise<T | null>,
): Promise<{ key: string; value: T } | null> {
  for (let pass = 0; pass < 2; pass++) {
    const { names, fresh } = await listKeysCached(env, prefix, pass === 1);
    const remaining = new Set(names);
    let sawCandidate = false;
    for (let attempt = 0; attempt < 8; attempt++) {
      const key = pickMostRecentDated([...remaining], latestDate, opts);
      if (!key) break;
      sawCandidate = true;
      const value = await read(key);
      if (value !== null && value !== undefined) return { key, value };
      remaining.delete(key); // listed but gone (raced its own expiry)
    }
    // Re-list only when a memoised index was *demonstrably* wrong: it named
    // entries in range and every one of them turned out to be gone. An index
    // with nothing in range is a valid memoised negative, and re-listing on it
    // would defeat the memo entirely for families that are legitimately empty —
    // `birthday:` on any non-birthday day is checked on every /fact request
    // (DECISIONS #62).
    if (fresh || !sawCandidate) return null;
  }
  return null;
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
  return resolveMostRecent(env, prefix, latestDate, opts, (key) => env.CACHE.get(key));
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
  // Verified with a real read rather than trusted from the listing: a memoised
  // index that still names an expired entry would otherwise let a wrapper page
  // emit an <img> that 404s — the broken-image glyph #58 exists to prevent.
  // Reads are ~1% of their free-tier cap, so this is the cheap half of the trade.
  const hit = await resolveMostRecent(
    env,
    prefix,
    dateStr,
    { maxDaysBack: 8, ...opts },
    async (key) => (await env.CACHE.get(key, "stream")) !== null || null,
  );
  return hit !== null;
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
