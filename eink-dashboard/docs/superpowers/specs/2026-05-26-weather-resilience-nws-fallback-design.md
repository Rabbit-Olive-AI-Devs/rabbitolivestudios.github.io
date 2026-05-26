# Weather Resilience + NWS Fallback — Design

**Date:** 2026-05-26
**Status:** Approved (design)
**Branch:** `weather-resilience-nws-fallback`

## Problem

The E1002 `/color/weather` page went blank (gray panel, "Failed to load remote image" in
SenseCraft HMI). Investigation found:

- The worker serves valid HTML 200 and renders perfectly in Chromium at 800×480.
- `/color/weather` was taking **8–10 seconds** to respond, while the other E1002 pages
  (`/color/headlines`, `/skyline`, `/color/moment`) respond in <0.25s.
- Root cause: **Open-Meteo is returning 502/504 after ~10s** (an upstream outage, confirmed
  even for a minimal `current=temperature_2m` query). `getWeatherForLocation` *awaits* that
  slow failing fetch (10s `fetchWithTimeout`) before falling back to stale cache, so every
  request pays ~10s.
- SenseCraft's renderer times out waiting for the page → "Failed to load remote image" →
  blank/gray panel.

Two faults combined: **(A)** Open-Meteo's outage, and **(B)** our page *blocks* on the slow
upstream instead of instantly serving the stale cache it already holds. We own (B), and (B)
is what makes any provider outage blank the display.

## Goals

1. **Resilience:** `/color/weather` and `/weather` must respond in **<1s when the cache is
   warm**, regardless of upstream health. Never block the request path on the network when a
   usable cache exists.
2. **Fallback:** When Open-Meteo fails, fetch fresh forecast data from **NWS
   (api.weather.gov, no API key)** instead of only serving stale data.

## Non-Goals

- Replacing Open-Meteo as the primary source (it is excellent and free normally).
- Adding any keyed/paid provider (keeps the project's no-key preference, DECISIONS #16).
- Changing the weather page UI/layout.

## Approach (chosen: B)

Stale-while-revalidate + NWS fallback.

- **A — Short timeout only:** rejected. Still blocks up to N seconds every request during an
  outage; SenseCraft's render timeout is unknown (working pages are <0.25s, so even 3s is a
  gamble); no freshness benefit.
- **B — Stale-while-revalidate + NWS fallback:** chosen. Serve cache instantly; refresh in
  the background via `ctx.waitUntil`; refresh tries Open-Meteo then NWS.
- **C — Switch primary to NWS:** rejected. Bigger mapping change, US-only lock-in, still
  needs the resilience fix anyway.

## Design

### 1. Resilience (stale-while-revalidate)

Thread `ctx: ExecutionContext` from the worker `fetch` handler into the weather path:

```
fetch(request, env, ctx)
  → handleColorWeatherPage(env, url, ctx)   // and handleWeatherPageV2(env, url, ctx)
    → getWeatherForLocation(env, lat, lon, zip, name, ctx)
```

New cache logic in `getWeatherForLocation`:

- **Fresh** (`age < CACHE_TTL_MS`, 15 min): return cached. *(unchanged)*
- **Soft-stale** (cache exists, older than 15 min): return cached **immediately**, and call
  `ctx.waitUntil(refresh(...))` to repopulate KV for the next request. Sub-second response.
- **Cold** (no cache at all): `await refresh(...)` (blocks) so a first-ever request still
  produces data. Open-Meteo timeout shortened to **~4s** here so the NWS fallback is reached
  quickly rather than ~10s later.

`ctx` is optional in the signature (`ctx?: ExecutionContext`); when absent (e.g. cron path),
the function falls back to awaiting the refresh instead of backgrounding it.

### 2. NWS fallback provider — `src/weather-nws.ts`

Mirrors the `alerts.ts` pattern: no API key, requires `User-Agent` header, uses
`fetchWithTimeout`. One exported function:

```ts
fetchWeatherFromNWS(lat, lon, zip, name): Promise<WeatherResponse | null>
```

NWS is two-step:

1. `GET https://api.weather.gov/points/{lat},{lon}` → `properties.forecast` (12 day/night
   periods) and `properties.forecastHourly` URLs. These are static per location → cache as
   `nws-points:{zip}` with a long TTL (604800).
2. `GET forecastHourly?units=si` and `GET forecast?units=si` (Celsius directly).

Map into the existing `WeatherResponse` shape (page code untouched):

- **current** ← first `forecastHourly` period (temperature, relativeHumidity, windSpeed,
  windDirection, isDaytime). `feels_like_c` ← temperature (NWS hourly has no apparent temp).
- **hourly_12h** ← next hourly periods (temperature, `probabilityOfPrecipitation.value`).
- **daily_5d** ← `forecast` periods collapsed into day/night pairs → high/low (a "daytime"
  period high + matching "overnight" period low), up to 5 days.
- **icon** ← NWS returns text (`shortForecast`, e.g. "Partly Cloudy", "Rain Showers"), not
  WMO codes. A pure `nwsTextToIcon(shortForecast, isDaytime)` mapper returns an existing icon
  key (clear/clear_night/partly_cloudy/cloudy/fog/drizzle/rain/snow/thunder). Unmatched →
  `cloudy` (or clear/clear_night if "sunny"/"clear").
- **wind** ← `parseNwsWind("10 mph"|"5 to 10 mph")` → km/h number (pure function).
- **precip_next_2h** ← `[]` (NWS has no 15-min data; `getRainWarning` already degrades to the
  hourly-probability path).
- Returns `null` on any failure (caller treats as "fallback also failed").

### 3. Integration & caching

Refresh chain (shared by background refresh and cold start):

```
refresh():
  try Open-Meteo (4s timeout) → normalize → cache {data, timestamp, source:"open-meteo"}
  catch → try NWS              → cache {data, timestamp, source:"nws"}
  catch → leave existing cache untouched (never overwrite good data with nothing)
```

- `CachedValue<WeatherResponse>` gains an optional `source?: "open-meteo" | "nws"` field
  (additive). Surfaced in `/health-detailed`.
- **Cache key stays `weather:{zip}:v2`** (NOT bumped). Stored shape is unchanged (additive
  optional field). Keeping the key means the existing ~24h stale cache serves instantly the
  moment this deploys — essential while Open-Meteo is actively down. Bumping would force a
  cold start into a dead Open-Meteo. Deliberate exception to the "bump on pipeline change"
  rule, documented in DECISIONS.
- **Dedup:** `getWeather(env, ctx?)` becomes a thin wrapper calling
  `getWeatherForLocation(env, NAPERVILLE_LAT, NAPERVILLE_LON, "60540", "Naperville, IL", ctx)`.
  Required so both weather pages get the fix from one implementation; not unrelated
  refactoring.

### 4. Error handling

- Cold start, both providers fail, no cache → existing try/catch in the page handlers returns
  a plain-text 503 with `Retry-After: 300` (unchanged).
- Background refresh failure → logged, cache left intact, page already served stale.
- NWS module never throws to the caller (returns `null`).

### 5. Testing

- **Unit (`test:utils`, no network):** `nwsTextToIcon`, `parseNwsWind`, and day/night→daily
  high/low collapsing, fed canned NWS JSON fixtures.
- **Local (`wrangler dev`):** dev-only `?test-provider=nws` and `?test-provider=stale` query
  params on the weather pages to force the fallback / stale path without waiting for a real
  Open-Meteo outage. Verify:
  - Warm cache → response <1s.
  - Forced Open-Meteo failure → NWS renders correctly at 800×480 (normal, alert banner, rain
    warning states).
  - Cold + both down → 503, no hang.
- **Production after deploy:** `/color/weather` timing drops ~9s → <1s; `/health-detailed`
  shows live `source`; device renders the panel.

### 6. Docs (same commit)

DECISIONS.md (resilience + NWS fallback + cache-key-not-bumped exception), README (provider
table + fallback note + `?test-provider` params), MEMORY.md, `package.json` + `src/index.ts`
VERSION bump.

## Files Touched

- `src/index.ts` — `fetch(request, env, ctx)`; pass `ctx` to weather handlers.
- `src/pages/color-weather.ts` — accept `ctx`, pass to `getWeatherForLocation`; `?test-provider`.
- `src/pages/weather2.ts` — accept `ctx`, pass through; `?test-provider`.
- `src/weather.ts` — stale-while-revalidate, refresh chain, `getWeather` dedup, 4s OM timeout.
- `src/weather-nws.ts` — **new** NWS provider + pure helpers.
- `src/types.ts` — `CachedValue.source?`.
- tests, DECISIONS.md, README.md, MEMORY.md, package.json.

## Rollback

Isolated on `weather-resilience-nws-fallback`. If it misbehaves after merge, revert the
commit; the prior code returns (which blocks ~10s during an Open-Meteo outage but works
normally otherwise).
