/**
 * World Cup 2026 data layer.
 *
 * Stale-while-revalidate over KV (mirrors weather.ts):
 *  - fresh cache (<SOFT_TTL): return it
 *  - stale cache: return immediately, refresh in background (ctx.waitUntil)
 *  - cold cache: bounded refresh (withBudget) on the request path
 *
 * Refresh chain: football-data.org (primary) -> openfootball (fallback).
 * Phase + todayMatches + generatedAt are computed here from the full match list.
 */

import { withBudget } from "./with-budget";
import { worldCupCacheKey } from "./cache-keys";
import { fetchFootballData } from "./worldcup-football-data";
import { fetchOpenFootball } from "./worldcup-openfootball";
import { computePhase, qualifiedFlags, teamCode } from "./worldcup-ui";
import { getChicagoDateISO } from "./date-utils";
import type { Env, WorldCupData, WcMatch, CachedValue } from "./types";

const SOFT_TTL_MS = 12 * 60 * 1000;   // 12 min freshness (drives SWR)
const HARD_TTL_S = 86400;             // 24h KV survival for stale fallback (DECISIONS.md #24)
const REFRESH_BUDGET_MS = 5000;       // cold-path wall-clock bound (SenseCraft renderer)

/** Finalize a normalized blob: compute phase, todayMatches, generatedAt; strip carry-along. */
function finalize(data: WorldCupData): WorldCupData {
  const all: WcMatch[] = (data as any)._allMatches ?? [];
  const today = getChicagoDateISO();
  const phase = computePhase(all);
  const todayMatches = all
    .filter((m) => m.dateChicago === today)
    .sort((a, b) => a.kickoffISO.localeCompare(b.kickoffISO));
  // Recompute each group's "guaranteed top-2" flags fixture-aware (the full match list is
  // only available here, before _allMatches is stripped). See DECISIONS #47.
  const groups = data.groups.map((g) => {
    const remaining = all
      .filter((m) => m.stage === "GROUP" && m.group === g.name && m.status !== "FINISHED")
      .map((m) => ({ home: teamCode(m.home), away: teamCode(m.away) }));
    const flags = qualifiedFlags(g.rows, remaining);
    return { ...g, rows: g.rows.map((r, i) => ({ ...r, qualifying: flags[i] })) };
  });
  const out: WorldCupData = {
    ...data,
    phase,
    todayMatches,
    groups,
    generatedAt: Date.now(),
  };
  delete (out as any)._allMatches;
  return out;
}

async function refresh(env: Env, cacheKey: string): Promise<WorldCupData | null> {
  let data = await fetchFootballData(env);
  if (!data) {
    console.warn("WorldCup: football-data unavailable, trying openfootball");
    data = await fetchOpenFootball(env);
  }
  if (!data) {
    console.error("WorldCup: all sources failed");
    return null;
  }
  const finalized = finalize(data);
  await env.CACHE.put(
    cacheKey,
    JSON.stringify({ data: finalized, timestamp: Date.now() }),
    { expirationTtl: HARD_TTL_S },
  );
  console.log(`WorldCup: refreshed from ${finalized.source} (phase=${finalized.phase})`);
  return finalized;
}

/** Get World Cup data with stale-while-revalidate + degradation. */
export async function getWorldCupData(env: Env, ctx?: ExecutionContext): Promise<WorldCupData> {
  const cacheKey = worldCupCacheKey();
  const cached = await env.CACHE.get<CachedValue<WorldCupData>>(cacheKey, "json");
  const fresh = cached && Date.now() - cached.timestamp < SOFT_TTL_MS;

  if (cached && fresh) {
    console.log("WorldCup: cache hit");
    return cached.data;
  }

  const doRefresh = () => refresh(env, cacheKey);

  if (cached) {
    console.log("WorldCup: serving stale, revalidating");
    if (ctx) {
      ctx.waitUntil(doRefresh().catch((e) => console.error("WorldCup bg refresh:", e)));
    } else {
      const r = await doRefresh();
      if (r) return r;
    }
    return cached.data;
  }

  // Cold start: bound the wait on the request path.
  const refreshPromise = doRefresh();
  let result: WorldCupData | null;
  if (ctx) {
    result = await withBudget(refreshPromise, REFRESH_BUDGET_MS);
    if (result === null) {
      console.warn(`WorldCup: cold refresh exceeded ${REFRESH_BUDGET_MS}ms budget`);
      ctx.waitUntil(refreshPromise.catch(() => {}));
    }
  } else {
    result = await refreshPromise;
  }
  if (result) return result;
  // The stale branch above always returns when `cached` exists, so here it is null.
  throw new Error("WorldCup: no data from any source and no cache");
}
