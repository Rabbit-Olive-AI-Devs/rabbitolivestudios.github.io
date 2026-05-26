# Weather Resilience + NWS Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/color/weather` and `/weather` respond in <1s regardless of upstream health by serving cache instantly and refreshing in the background, with NWS (no-key) as a fallback forecast provider when Open-Meteo fails.

**Architecture:** Stale-while-revalidate in the weather data layer using `ctx.waitUntil` for background refresh; a new `src/weather-nws.ts` module maps NWS's two-step API into the existing `WeatherResponse` shape; `getWeather` is folded into `getWeatherForLocation` so both pages get the fix once.

**Tech Stack:** TypeScript, Cloudflare Workers, KV cache, `node --test` (pure-function unit tests compiled via `tsconfig.test.json`).

**Design spec:** `docs/superpowers/specs/2026-05-26-weather-resilience-nws-fallback-design.md`

**Branch:** `weather-resilience-nws-fallback` (already created).

---

## File Structure

- `src/weather-nws.ts` — **new.** NWS provider: pure helpers (`nwsTextToIcon`, `parseNwsWind`, `collapseNwsDaily`, `normalizeNws`) + `fetchWeatherFromNWS` orchestration. Self-contained; depends only on types + `fetch-timeout`.
- `src/weather.ts` — **modify.** `getWeatherForLocation` gains `ctx?` + stale-while-revalidate; new `refreshWeather` chain (Open-Meteo → NWS); `getWeather` becomes a thin wrapper.
- `src/types.ts` — **modify.** `CachedValue.source?`.
- `src/index.ts` — **modify.** `fetch(request, env, ctx)`; thread `ctx` to `handleWeather`, `handleWeatherPageV2`, `handleColorWeatherPage`.
- `src/pages/color-weather.ts` — **modify.** Accept `ctx`, pass to `getWeatherForLocation`; honor `?test-provider`.
- `src/pages/weather2.ts` — **modify.** Accept `ctx`, pass through; honor `?test-provider`.
- `tests/weather-nws.test.js` — **new.** Unit tests for the NWS pure helpers.
- `DECISIONS.md`, `README.md`, `MEMORY.md`, `package.json`, `src/index.ts` VERSION — **modify.** Docs + version.

---

## Task 1: NWS pure helpers (icon + wind + daily collapse)

**Files:**
- Create: `src/weather-nws.ts`
- Create: `tests/weather-nws.test.js`

- [ ] **Step 1: Write the failing tests**

Create `tests/weather-nws.test.js`:

```js
const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { nwsTextToIcon, parseNwsWind, collapseNwsDaily } = fromBuild("src/weather-nws.js");

test("nwsTextToIcon maps NWS shortForecast text to icon keys", () => {
  assert.equal(nwsTextToIcon("Sunny", true), "clear");
  assert.equal(nwsTextToIcon("Clear", false), "clear_night");
  assert.equal(nwsTextToIcon("Mostly Sunny", true), "partly_cloudy");
  assert.equal(nwsTextToIcon("Partly Cloudy", false), "partly_cloudy_night");
  assert.equal(nwsTextToIcon("Mostly Cloudy", true), "cloudy");
  assert.equal(nwsTextToIcon("Overcast", true), "cloudy");
  assert.equal(nwsTextToIcon("Rain Showers", true), "rain");
  assert.equal(nwsTextToIcon("Light Drizzle", true), "drizzle");
  assert.equal(nwsTextToIcon("Snow", true), "snow");
  assert.equal(nwsTextToIcon("Chance Thunderstorms", true), "thunder");
  assert.equal(nwsTextToIcon("Patchy Fog", true), "fog");
  assert.equal(nwsTextToIcon("Something Unknown", true), "cloudy");
});

test("parseNwsWind extracts the max number from an si wind string (km/h)", () => {
  assert.equal(parseNwsWind("16 km/h"), 16);
  assert.equal(parseNwsWind("8 to 21 km/h"), 21);
  assert.equal(parseNwsWind(""), 0);
  assert.equal(parseNwsWind("Calm"), 0);
});

test("collapseNwsDaily pairs day/night periods into high/low by date", () => {
  const periods = [
    { startTime: "2026-05-26T06:00:00-05:00", isDaytime: true,  temperature: 24, shortForecast: "Sunny",        probabilityOfPrecipitation: { value: 10 } },
    { startTime: "2026-05-26T18:00:00-05:00", isDaytime: false, temperature: 14, shortForecast: "Clear",        probabilityOfPrecipitation: { value: 0 } },
    { startTime: "2026-05-27T06:00:00-05:00", isDaytime: true,  temperature: 22, shortForecast: "Rain Showers", probabilityOfPrecipitation: { value: 80 } },
    { startTime: "2026-05-27T18:00:00-05:00", isDaytime: false, temperature: 12, shortForecast: "Cloudy",       probabilityOfPrecipitation: { value: 30 } },
  ];
  const days = collapseNwsDaily(periods);
  assert.equal(days.length, 2);
  assert.equal(days[0].date, "2026-05-26");
  assert.equal(days[0].high_c, 24);
  assert.equal(days[0].low_c, 14);
  assert.equal(days[0].icon, "clear"); // from the daytime period
  assert.equal(days[0].precip_prob_pct, 10);
  assert.equal(days[1].date, "2026-05-27");
  assert.equal(days[1].high_c, 22);
  assert.equal(days[1].low_c, 12);
  assert.equal(days[1].icon, "rain");
});

test("collapseNwsDaily handles a leading night period (missing daytime high)", () => {
  const periods = [
    { startTime: "2026-05-26T18:00:00-05:00", isDaytime: false, temperature: 14, shortForecast: "Clear", probabilityOfPrecipitation: { value: 0 } },
  ];
  const days = collapseNwsDaily(periods);
  assert.equal(days.length, 1);
  assert.equal(days[0].low_c, 14);
  assert.equal(days[0].high_c, 14); // falls back to low when no daytime period
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm run test:utils`
Expected: FAIL — `Cannot find module '.../src/weather-nws.js'` (module not created yet).

- [ ] **Step 3: Create `src/weather-nws.ts` with the pure helpers**

```ts
/**
 * NWS (api.weather.gov) fallback forecast provider for when Open-Meteo fails.
 *
 * No API key (requires User-Agent header, like alerts.ts). Two-step API:
 *   1. GET /points/{lat},{lon} -> forecast + forecastHourly URLs
 *   2. GET those URLs with ?units=si (Celsius, km/h)
 * Maps into the existing WeatherResponse shape so page code is unchanged.
 */

import { fetchWithTimeout } from "./fetch-timeout";
import { getWeatherInfo } from "./weather-codes";
import type { Env, WeatherResponse, HourlyEntry, DailyEntry, NWSAlert } from "./types";

const UA = "(eink-dashboard, rabbitolivestudios@gmail.com)";

interface NwsPeriod {
  startTime: string;
  isDaytime: boolean;
  temperature: number;
  shortForecast: string;
  probabilityOfPrecipitation?: { value: number | null };
  relativeHumidity?: { value: number | null };
  windSpeed?: string;
  windDirection?: string;
}

/** Map an NWS shortForecast string to one of our existing icon keys. */
export function nwsTextToIcon(shortForecast: string, isDaytime: boolean): string {
  const s = shortForecast.toLowerCase();
  if (s.includes("thunder")) return "thunder";
  if (s.includes("snow") || s.includes("flurr") || s.includes("sleet") || s.includes("ice")) return "snow";
  if (s.includes("drizzle")) return "drizzle";
  if (s.includes("rain") || s.includes("shower")) return "rain";
  if (s.includes("fog") || s.includes("haze") || s.includes("mist") || s.includes("smoke")) return "fog";
  if (s.includes("partly") || s.includes("mostly sunny") || s.includes("mostly clear")) {
    return isDaytime ? "partly_cloudy" : "partly_cloudy_night";
  }
  if (s.includes("cloud") || s.includes("overcast")) return "cloudy";
  if (s.includes("sunny") || s.includes("clear") || s.includes("fair")) {
    return isDaytime ? "clear" : "clear_night";
  }
  return "cloudy";
}

/** Extract the max integer from an NWS si wind string ("16 km/h", "8 to 21 km/h"). */
export function parseNwsWind(windSpeed: string): number {
  const nums = (windSpeed.match(/\d+/g) || []).map(Number);
  return nums.length ? Math.max(...nums) : 0;
}

/** Collapse NWS day/night forecast periods into one entry per calendar date. */
export function collapseNwsDaily(periods: NwsPeriod[]): DailyEntry[] {
  const byDate = new Map<string, DailyEntry>();
  const order: string[] = [];
  for (const p of periods) {
    const date = p.startTime.slice(0, 10);
    let entry = byDate.get(date);
    if (!entry) {
      entry = {
        date,
        high_c: NaN,
        low_c: NaN,
        precip_prob_pct: 0,
        precipitation_sum_mm: 0,
        snowfall_sum_cm: 0,
        code: 0,
        icon: "cloudy",
        sunrise: "",
        sunset: "",
      };
      byDate.set(date, entry);
      order.push(date);
    }
    const prob = p.probabilityOfPrecipitation?.value ?? 0;
    entry.precip_prob_pct = Math.max(entry.precip_prob_pct, prob);
    if (p.isDaytime) {
      entry.high_c = Math.round(p.temperature);
      entry.icon = nwsTextToIcon(p.shortForecast, true); // daytime icon represents the day
    } else {
      entry.low_c = Math.round(p.temperature);
      if (entry.icon === "cloudy" && Number.isNaN(entry.high_c)) {
        entry.icon = nwsTextToIcon(p.shortForecast, false);
      }
    }
  }
  // Fill missing high/low from whichever is present.
  for (const date of order) {
    const e = byDate.get(date)!;
    if (Number.isNaN(e.high_c)) e.high_c = Number.isNaN(e.low_c) ? 0 : e.low_c;
    if (Number.isNaN(e.low_c)) e.low_c = e.high_c;
  }
  return order.map((d) => byDate.get(d)!);
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm run test:utils`
Expected: PASS (all `weather-nws` tests green, plus the existing `utils` tests still pass).

- [ ] **Step 5: Commit**

```bash
git add src/weather-nws.ts tests/weather-nws.test.js
git commit -m "Add NWS forecast pure helpers (icon/wind/daily mappers)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: NWS normalize + fetch orchestration

**Files:**
- Modify: `src/weather-nws.ts`
- Modify: `tests/weather-nws.test.js`

- [ ] **Step 1: Write the failing test for `normalizeNws`**

Append to `tests/weather-nws.test.js`:

```js
const { normalizeNws } = fromBuild("src/weather-nws.js");

test("normalizeNws builds a WeatherResponse from forecast + hourly JSON", () => {
  const forecast = { properties: { periods: [
    { startTime: "2026-05-26T06:00:00-05:00", isDaytime: true,  temperature: 24, shortForecast: "Sunny", probabilityOfPrecipitation: { value: 10 } },
    { startTime: "2026-05-26T18:00:00-05:00", isDaytime: false, temperature: 14, shortForecast: "Clear", probabilityOfPrecipitation: { value: 0 } },
  ] } };
  const hourly = { properties: { periods: [
    { startTime: "2026-05-26T09:00:00-05:00", isDaytime: true, temperature: 20, shortForecast: "Sunny",
      probabilityOfPrecipitation: { value: 5 }, relativeHumidity: { value: 50 }, windSpeed: "13 km/h", windDirection: "SW" },
    { startTime: "2026-05-26T10:00:00-05:00", isDaytime: true, temperature: 21, shortForecast: "Sunny",
      probabilityOfPrecipitation: { value: 0 }, relativeHumidity: { value: 48 }, windSpeed: "16 km/h", windDirection: "S" },
  ] } };
  const alerts = [];
  const w = normalizeNws(forecast, hourly, alerts, 41.88, -87.63, "60606", "Chicago, IL");

  assert.equal(w.location.zip, "60606");
  assert.equal(w.current.temp_c, 20);
  assert.equal(w.current.humidity_pct, 50);
  assert.equal(w.current.wind_kmh, 13);
  assert.equal(w.current.wind_dir_label, "SW");
  assert.equal(w.current.is_day, true);
  assert.equal(w.hourly_12h.length, 2);
  assert.equal(w.hourly_12h[0].time, "2026-05-26T09:00");
  assert.equal(w.hourly_12h[0].temp_c, 20);
  assert.equal(w.daily_5d.length, 1);
  assert.equal(w.daily_5d[0].high_c, 24);
  assert.equal(w.daily_5d[0].low_c, 14);
  assert.deepEqual(w.precip_next_2h, []);
  assert.equal(w.sunrise, ""); // NWS gives no sunrise/sunset -> sun line omitted on the page
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test:utils`
Expected: FAIL — `normalizeNws is not a function`.

- [ ] **Step 3: Add `normalizeNws` + `fetchWeatherFromNWS` to `src/weather-nws.ts`**

Append to `src/weather-nws.ts`:

```ts
/** Build a WeatherResponse from NWS forecast + forecastHourly JSON (pure). */
export function normalizeNws(
  forecast: any,
  hourly: any,
  alerts: NWSAlert[],
  lat: number,
  lon: number,
  zip: string,
  name: string,
): WeatherResponse {
  const hourlyPeriods: NwsPeriod[] = hourly?.properties?.periods ?? [];
  const dailyPeriods: NwsPeriod[] = forecast?.properties?.periods ?? [];
  if (hourlyPeriods.length === 0) throw new Error("NWS hourly has no periods");

  const cur = hourlyPeriods[0];
  const curInfo = getWeatherInfo(0, cur.isDaytime); // code unused; we override icon/label below
  const curIcon = nwsTextToIcon(cur.shortForecast, cur.isDaytime);

  const hourly_12h: HourlyEntry[] = hourlyPeriods.slice(0, 24).map((h) => ({
    time: h.startTime.slice(0, 16),
    temp_c: Math.round(h.temperature),
    precip_prob_pct: h.probabilityOfPrecipitation?.value ?? 0,
    precip_mm: 0,
    code: 0,
    icon: nwsTextToIcon(h.shortForecast, h.isDaytime),
    is_day: h.isDaytime,
  }));

  const daily_5d: DailyEntry[] = collapseNwsDaily(dailyPeriods).slice(0, 5);

  return {
    location: { zip, name, lat, lon, tz: "America/Chicago" },
    updated_at: cur.startTime,
    current: {
      temp_c: Math.round(cur.temperature),
      feels_like_c: Math.round(cur.temperature), // NWS hourly has no apparent temp
      humidity_pct: Math.round(cur.relativeHumidity?.value ?? 0),
      wind_kmh: parseNwsWind(cur.windSpeed ?? ""),
      wind_dir_deg: 0,
      wind_dir_label: cur.windDirection ?? "",
      wind_gusts_kmh: 0,
      is_day: cur.isDaytime,
      precip_mm_hr: 0,
      condition: { code: 0, label: cur.shortForecast || curInfo.label, icon: curIcon },
    },
    hourly_12h,
    daily_5d,
    precip_next_2h: [],
    alerts,
    sunrise: "",
    sunset: "",
  };
}

/**
 * Fetch a full WeatherResponse from NWS. Returns null on any failure so the
 * caller treats it as "fallback also failed". `alerts` is passed in (already
 * fetched/cached by the caller). pointsCacheKey caches the static points lookup.
 */
export async function fetchWeatherFromNWS(
  env: Env,
  lat: number,
  lon: number,
  zip: string,
  name: string,
  alerts: NWSAlert[],
  timeoutMs = 6000,
): Promise<WeatherResponse | null> {
  try {
    const pointsKey = `nws-points:${zip}`;
    let urls = await env.CACHE.get<{ forecast: string; forecastHourly: string }>(pointsKey, "json");
    if (!urls) {
      const pres = await fetchWithTimeout(
        `https://api.weather.gov/points/${lat},${lon}`,
        { headers: { "User-Agent": UA, Accept: "application/geo+json" } },
        timeoutMs,
      );
      if (!pres.ok) throw new Error(`NWS points ${pres.status}`);
      const pjson: any = await pres.json();
      urls = { forecast: pjson.properties.forecast, forecastHourly: pjson.properties.forecastHourly };
      await env.CACHE.put(pointsKey, JSON.stringify(urls), { expirationTtl: 604800 });
    }

    const withUnits = (u: string) => u + (u.includes("?") ? "&" : "?") + "units=si";
    const [fres, hres] = await Promise.all([
      fetchWithTimeout(withUnits(urls.forecast), { headers: { "User-Agent": UA, Accept: "application/geo+json" } }, timeoutMs),
      fetchWithTimeout(withUnits(urls.forecastHourly), { headers: { "User-Agent": UA, Accept: "application/geo+json" } }, timeoutMs),
    ]);
    if (!fres.ok || !hres.ok) throw new Error(`NWS forecast ${fres.status}/${hres.status}`);
    const forecast: any = await fres.json();
    const hourly: any = await hres.json();
    return normalizeNws(forecast, hourly, alerts, lat, lon, zip, name);
  } catch (err) {
    console.error(`NWS fallback failed (${zip}):`, err);
    return null;
  }
}
```

- [ ] **Step 4: Run tests + typecheck to verify they pass**

Run: `npm run test:utils && npm run typecheck`
Expected: PASS — `normalizeNws` test green, no type errors.

- [ ] **Step 5: Commit**

```bash
git add src/weather-nws.ts tests/weather-nws.test.js
git commit -m "Add NWS normalize + fetch orchestration (returns WeatherResponse or null)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Weather resilience (stale-while-revalidate + refresh chain + dedup)

**Files:**
- Modify: `src/types.ts` (CachedValue.source)
- Modify: `src/weather.ts`

- [ ] **Step 1: Add optional `source` to `CachedValue` in `src/types.ts`**

Replace the `CachedValue` interface (around line 133):

```ts
export interface CachedValue<T> {
  data: T;
  timestamp: number;
  /** Which provider produced this weather data (diagnostics; weather only). */
  source?: "open-meteo" | "nws";
}
```

- [ ] **Step 2: Rewrite `src/weather.ts` with the refresh chain + stale-while-revalidate**

Replace the top constants + `getWeather` + `getWeatherForLocation` (lines 1–112) with the following. Keep `windDirLabel`, `round2`, and `normalizeWeather` (the Open-Meteo normalizer) unchanged below.

```ts
import { getWeatherInfo } from "./weather-codes";
import { fetchAlerts, fetchAlertsForLocation } from "./alerts";
import { fetchWithTimeout } from "./fetch-timeout";
import { fetchWeatherFromNWS } from "./weather-nws";
import type { Env, WeatherResponse, HourlyEntry, DailyEntry, CachedValue, NWSAlert } from "./types";

const NAPERVILLE_LAT = 41.7508;
const NAPERVILLE_LON = -88.1535;
const CACHE_TTL_MS = 15 * 60 * 1000; // 15 minutes (soft TTL: when to revalidate)
const OPEN_METEO_TIMEOUT_MS = 4000;   // short so the NWS fallback is reached quickly

const WIND_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] as const;

function openMeteoURL(lat: number, lon: number): string {
  return `https://api.open-meteo.com/v1/forecast` +
    `?latitude=${lat}&longitude=${lon}` +
    `&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m,is_day,wind_gusts_10m` +
    `&hourly=temperature_2m,precipitation_probability,precipitation,weather_code,wind_speed_10m,is_day` +
    `&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code,sunrise,sunset,precipitation_sum,snowfall_sum` +
    `&minutely_15=precipitation&forecast_minutely_15=8&past_minutely_15=0` +
    `&timezone=America%2FChicago&forecast_days=5&forecast_hours=24`;
}

/** E1001 home weather — thin wrapper so both pages share resilience + fallback. */
export async function getWeather(env: Env, ctx?: ExecutionContext): Promise<WeatherResponse> {
  return getWeatherForLocation(env, NAPERVILLE_LAT, NAPERVILLE_LON, "60540", "Naperville, IL", ctx);
}

/**
 * Fetch weather for a location with stale-while-revalidate:
 *  - fresh cache (<15min): return it
 *  - stale cache: return it immediately, refresh in background (ctx.waitUntil)
 *  - no cache (cold): block on the refresh chain (Open-Meteo -> NWS)
 *
 * `opts.forceProvider` is for local testing only (?test-provider).
 */
export async function getWeatherForLocation(
  env: Env,
  lat: number,
  lon: number,
  zip: string,
  name: string,
  ctx?: ExecutionContext,
  opts?: { forceProvider?: "nws" | "fail" },
): Promise<WeatherResponse> {
  const cacheKey = `weather:${zip}:v2`;
  const alertsCacheKey = `alerts:${zip}:v1`;

  const cached = await env.CACHE.get<CachedValue<WeatherResponse>>(cacheKey, "json");
  const fresh = cached && Date.now() - cached.timestamp < CACHE_TTL_MS;

  // Test override forces a refresh regardless of cache.
  if (cached && fresh && !opts?.forceProvider) {
    console.log(`Weather ${zip}: cache hit`);
    return cached.data;
  }

  const doRefresh = () => refreshWeather(env, lat, lon, zip, name, cacheKey, alertsCacheKey, opts);

  if (cached && !opts?.forceProvider) {
    // Stale: serve immediately, revalidate in background (or await if no ctx, e.g. cron).
    console.log(`Weather ${zip}: serving stale, revalidating`);
    if (ctx) {
      ctx.waitUntil(doRefresh().catch((e) => console.error(`Weather ${zip} bg refresh:`, e)));
    } else {
      const refreshed = await doRefresh();
      if (refreshed) return refreshed;
    }
    return cached.data;
  }

  // Cold start (or forced test): must block on the chain.
  const refreshed = await doRefresh();
  if (refreshed) return refreshed;
  if (cached) return cached.data;
  throw new Error(`Weather ${zip}: no data from any provider and no cache`);
}

/**
 * Try Open-Meteo (short timeout), then NWS. On success, write the cache and
 * return the data. On total failure, return null and leave the cache untouched.
 */
async function refreshWeather(
  env: Env,
  lat: number,
  lon: number,
  zip: string,
  name: string,
  cacheKey: string,
  alertsCacheKey: string,
  opts?: { forceProvider?: "nws" | "fail" },
): Promise<WeatherResponse | null> {
  const alerts = zip === "60540"
    ? await fetchAlerts(env)
    : await fetchAlertsForLocation(env, lat, lon, alertsCacheKey);

  // Open-Meteo primary (skipped when a test override forces NWS/fail).
  if (!opts?.forceProvider) {
    try {
      const res = await fetchWithTimeout(openMeteoURL(lat, lon), undefined, OPEN_METEO_TIMEOUT_MS);
      if (!res.ok) throw new Error(`Open-Meteo returned ${res.status}`);
      const raw: any = await res.json();
      const weather = normalizeWeather(raw, alerts, lat, lon, zip, name);
      await env.CACHE.put(cacheKey, JSON.stringify({ data: weather, timestamp: Date.now(), source: "open-meteo" }), { expirationTtl: 86400 });
      console.log(`Weather ${zip}: refreshed from Open-Meteo`);
      return weather;
    } catch (err) {
      console.error(`Weather ${zip}: Open-Meteo failed, trying NWS:`, err);
    }
  }

  // NWS fallback.
  if (opts?.forceProvider !== "fail") {
    const nws = await fetchWeatherFromNWS(env, lat, lon, zip, name, alerts);
    if (nws) {
      await env.CACHE.put(cacheKey, JSON.stringify({ data: nws, timestamp: Date.now(), source: "nws" }), { expirationTtl: 86400 });
      console.log(`Weather ${zip}: refreshed from NWS fallback`);
      return nws;
    }
  }

  console.error(`Weather ${zip}: all providers failed`);
  return null;
}
```

Note: `normalizeWeather`, `windDirLabel`, `round2` remain below this block unchanged. Remove the now-unused module-level `LAT`/`LON`/`CACHE_KEY`/`OPEN_METEO_URL` constants (replaced by `openMeteoURL()` and the Naperville constants).

- [ ] **Step 3: Typecheck**

Run: `npm run typecheck`
Expected: PASS. (If `ExecutionContext` is unknown, it is provided by `@cloudflare/workers-types` — already a devDependency.)

- [ ] **Step 4: Run the existing unit tests (regression)**

Run: `npm run test:utils`
Expected: PASS — no existing tests broken.

- [ ] **Step 5: Commit**

```bash
git add src/types.ts src/weather.ts
git commit -m "Add stale-while-revalidate weather fetch with Open-Meteo->NWS refresh chain

Serve cache instantly and revalidate in the background (ctx.waitUntil) so the
weather pages never block on a slow/failing upstream. getWeather now delegates
to getWeatherForLocation so both pages share the logic. Cache key stays v2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Thread `ctx` through the router + `?test-provider`

**Files:**
- Modify: `src/index.ts`
- Modify: `src/pages/color-weather.ts`
- Modify: `src/pages/weather2.ts`

- [ ] **Step 1: Update the `fetch` handler signature in `src/index.ts`**

Change (around line 848):

```ts
export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
```

- [ ] **Step 2: Pass `ctx` to the three request-path weather callers in `src/index.ts`**

```ts
// /weather.json route handler call:
async function handleWeather(env: Env, ctx: ExecutionContext): Promise<Response> {
  try {
    const weather = await getWeather(env, ctx);
```

Find where `handleWeather(env)` is routed and change it to `handleWeather(env, ctx)`. Then update the two page routes:

```ts
      case "/weather":
        return handleWeatherPageV2(env, url, ctx);
...
      case "/color/weather":
        return handleColorWeatherPage(env, url, ctx);
```

Leave the cron calls (`getWeather(env)`, `getWeatherForLocation(env, ...)` inside `handleScheduled`) **without** `ctx` — awaiting the refresh there is correct.

- [ ] **Step 3: Update `handleColorWeatherPage` in `src/pages/color-weather.ts`**

Change the signature and the `getWeatherForLocation` call:

```ts
export async function handleColorWeatherPage(env: Env, url: URL, ctx?: ExecutionContext): Promise<Response> {
  try {
    const forceProvider = url.searchParams.get("test-provider") as ("nws" | "fail" | null);
    const [weather, device] = await Promise.all([
      getWeatherForLocation(env, OFFICE_LAT, OFFICE_LON, OFFICE_ZIP, OFFICE_NAME, ctx,
        forceProvider ? { forceProvider } : undefined),
      fetchDeviceData(env, E1002_DEVICE_ID),
    ]);
```

(The rest of the function body is unchanged.)

- [ ] **Step 4: Update `handleWeatherPageV2` in `src/pages/weather2.ts`**

Add `ctx?: ExecutionContext` to the signature and pass it (plus `?test-provider`) into its `getWeather`/`getWeatherForLocation` call. Match whichever the file uses:

```ts
export async function handleWeatherPageV2(env: Env, url: URL, ctx?: ExecutionContext): Promise<Response> {
  // ...
  const forceProvider = url.searchParams.get("test-provider") as ("nws" | "fail" | null);
  const weather = await getWeather(env, ctx); // pass forceProvider only if this handler calls getWeatherForLocation directly
```

If `weather2.ts` calls `getWeather(env)`, change to `getWeather(env, ctx)`. (Mono home page; `?test-provider` is optional there — wiring it into the color page is what matters for verification. If `getWeather` does not accept opts, leave the mono test override out.)

- [ ] **Step 5: Typecheck + dry-run**

Run: `npm run typecheck && npm run dry-run`
Expected: PASS — compiles and the Worker builds.

- [ ] **Step 6: Commit**

```bash
git add src/index.ts src/pages/color-weather.ts src/pages/weather2.ts
git commit -m "Thread ExecutionContext into weather pages + add ?test-provider override

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Local verification, docs, version, deploy

**Files:**
- Modify: `DECISIONS.md`, `README.md`, `MEMORY.md`, `package.json`, `src/index.ts` (VERSION)

- [ ] **Step 1: Start local dev server**

```bash
lsof -ti:8790    # confirm free (kill if a stale process holds it)
npx wrangler dev --port 8790
```

- [ ] **Step 2: Verify the resilience + fallback behaviors locally**

Run each and confirm:

```bash
# Normal path (Open-Meteo primary). Should return HTML 200.
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" "http://localhost:8790/color/weather"

# Force NWS fallback: should render NWS data (HTML 200, valid markup).
curl -s "http://localhost:8790/color/weather?test-provider=nws" | grep -c "CHICAGO"

# Warm-cache timing: second normal request should be well under 1s.
curl -s -o /dev/null -w "warm: %{time_total}s\n" "http://localhost:8790/color/weather"
```

Expected: `200`; the `?test-provider=nws` response contains the rendered page (grep count ≥ 1); warm request `time_total` < 1s.

- [ ] **Step 3: Visual check at 800×480**

Open `http://localhost:8790/color/weather?test-provider=nws` in a browser at an 800×480 viewport (or via Playwright). Confirm: temps colored correctly, 5-day + hourly cards populated from NWS, no broken layout, no sun/moon line (expected — NWS has no sunrise/sunset). Also check `?test-provider=nws&test-alert=tornado` and `?test-provider=nws&test-rain` render the banners.

- [ ] **Step 4: Stop dev server, update docs**

- `DECISIONS.md`: add a decision documenting (a) the root cause (Open-Meteo outage + blocking fetch → SenseCraft render timeout), (b) stale-while-revalidate, (c) NWS fallback, (d) the deliberate choice NOT to bump the `weather:*:v2` cache key.
- `README.md`: update the weather provider description + endpoints table note that weather falls back to NWS; document `?test-provider=nws|fail`.
- `MEMORY.md`: update the NEXT SESSION block with the new version + behavior; keep under the line limit.
- `package.json` + `src/index.ts` VERSION: bump (e.g. `3.11.4` → `3.12.0` — new feature).

- [ ] **Step 5: Build + commit docs**

```bash
npm run typecheck && npm run test:utils && npm run dry-run
git add DECISIONS.md README.md package.json src/index.ts
# MEMORY.md lives outside the repo checkout; update it separately if applicable.
git commit -m "Document weather resilience + NWS fallback; bump to v3.12.0

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Deploy + production verification (after user approval)**

```bash
npx wrangler deploy
# Timing should drop from ~9s to <1s:
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" "https://eink-dashboard.thiago-oliveira77.workers.dev/color/weather"
# Confirm which provider is live:
curl -s "https://eink-dashboard.thiago-oliveira77.workers.dev/health-detailed" | grep -i source
```

Expected: HTTP 200, `time_total` < 1s. Then the user confirms the panel renders on the E1002 device.

---

## Self-Review

**Spec coverage:**
- Resilience (serve stale instantly + bg refresh) → Task 3 (stale-while-revalidate) + Task 4 (ctx threading). ✓
- NWS fallback module → Tasks 1–2. ✓
- Refresh chain Open-Meteo→NWS, source field, cache key kept → Task 3. ✓
- getWeather dedup → Task 3 Step 2. ✓
- 4s Open-Meteo timeout → Task 3 (`OPEN_METEO_TIMEOUT_MS`). ✓
- `?test-provider` dev override → Task 3 (opts) + Task 4 (wiring). ✓
- Unit tests for pure helpers → Tasks 1–2. ✓
- Local 800×480 + forced-failure verification → Task 5 Steps 2–3. ✓
- Docs sweep + version bump → Task 5 Step 4. ✓
- Cold + both-down → 503 → Task 3 (`throw` when no data and no cache; page handler's existing try/catch returns 503). ✓

**Placeholder scan:** No TBD/TODO. Task 4 Step 4 leaves the mono `?test-provider` wiring conditional on the handler's actual call (verified during implementation) — the color page (the reported bug) is fully specified.

**Type consistency:** `getWeatherForLocation(env, lat, lon, zip, name, ctx?, opts?)`, `refreshWeather(...)` returns `WeatherResponse | null`, `fetchWeatherFromNWS(...)` returns `WeatherResponse | null`, `normalizeNws/collapseNwsDaily/nwsTextToIcon/parseNwsWind` signatures match between Tasks 1–2 and their consumers in Task 3. `CachedValue.source` is the same union everywhere. `DailyEntry`/`HourlyEntry` fields match `normalizeWeather`'s existing output. ✓
