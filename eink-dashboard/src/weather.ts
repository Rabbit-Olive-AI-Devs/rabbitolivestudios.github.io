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

function windDirLabel(deg: number): string {
  return WIND_DIRS[Math.round(deg / 45) % 8];
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function normalizeWeather(
  raw: any,
  alerts: import("./types").NWSAlert[],
  lat: number,
  lon: number,
  zip: string,
  name: string,
): WeatherResponse {
  const current = raw.current;
  const hourly = raw.hourly;
  const daily = raw.daily;
  const currentIsDay = current.is_day === 1;
  const condInfo = getWeatherInfo(current.weather_code, currentIsDay);

  const hourly12h: HourlyEntry[] = [];
  const hourlyLen = Math.min(hourly.time.length, 24);
  for (let i = 0; i < hourlyLen; i++) {
    const isDay = hourly.is_day?.[i] === 1;
    const info = getWeatherInfo(hourly.weather_code[i], isDay);
    hourly12h.push({
      time: hourly.time[i],
      temp_c: Math.round(hourly.temperature_2m[i]),
      precip_prob_pct: hourly.precipitation_probability[i] ?? 0,
      precip_mm: round2(hourly.precipitation[i] ?? 0),
      code: hourly.weather_code[i],
      icon: info.icon,
      is_day: isDay,
    });
  }

  const daily5d: DailyEntry[] = [];
  const dailyLen = Math.min(daily.time.length, 5);
  for (let i = 0; i < dailyLen; i++) {
    const info = getWeatherInfo(daily.weather_code[i]);
    daily5d.push({
      date: daily.time[i],
      high_c: Math.round(daily.temperature_2m_max[i]),
      low_c: Math.round(daily.temperature_2m_min[i]),
      precip_prob_pct: daily.precipitation_probability_max[i] ?? 0,
      precipitation_sum_mm: round2(daily.precipitation_sum?.[i] ?? 0),
      snowfall_sum_cm: round2(daily.snowfall_sum?.[i] ?? 0),
      code: daily.weather_code[i],
      icon: info.icon,
      sunrise: daily.sunrise[i],
      sunset: daily.sunset[i],
    });
  }

  const precip_next_2h: number[] = [];
  const minutely15 = raw.minutely_15;
  if (minutely15?.precipitation) {
    for (let i = 0; i < Math.min(minutely15.precipitation.length, 8); i++) {
      precip_next_2h.push(round2(minutely15.precipitation[i] ?? 0));
    }
  }

  return {
    location: { zip, name, lat, lon, tz: "America/Chicago" },
    updated_at: current.time,
    current: {
      temp_c: Math.round(current.temperature_2m),
      feels_like_c: Math.round(current.apparent_temperature),
      humidity_pct: Math.round(current.relative_humidity_2m),
      wind_kmh: Math.round(current.wind_speed_10m),
      wind_dir_deg: Math.round(current.wind_direction_10m),
      wind_dir_label: windDirLabel(current.wind_direction_10m),
      wind_gusts_kmh: Math.round(current.wind_gusts_10m ?? 0),
      is_day: currentIsDay,
      precip_mm_hr: round2(current.precipitation),
      condition: { code: current.weather_code, label: condInfo.label, icon: condInfo.icon },
    },
    hourly_12h: hourly12h,
    daily_5d: daily5d,
    precip_next_2h,
    alerts,
    sunrise: daily5d[0]?.sunrise ?? "",
    sunset: daily5d[0]?.sunset ?? "",
  };
}
