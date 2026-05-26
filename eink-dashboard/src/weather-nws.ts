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
  // "mostly" alone (e.g. "Mostly Cloudy") falls through to cloudy below;
  // only "mostly sunny"/"mostly clear" count as partly cloudy.
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
  const nums = (windSpeed.match(/\d+/g) ?? []).map(Number);
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
      entry.icon = nwsTextToIcon(p.shortForecast, true);
    } else {
      entry.low_c = Math.round(p.temperature);
      // Only override icon from night period if we never got a daytime period for this date
      if (entry.icon === "cloudy" && Number.isNaN(entry.high_c)) {
        entry.icon = nwsTextToIcon(p.shortForecast, false);
      }
    }
  }
  // Fill in missing high/low for edge cases (e.g. leading night period)
  for (const date of order) {
    const e = byDate.get(date)!;
    if (Number.isNaN(e.high_c)) e.high_c = Number.isNaN(e.low_c) ? 0 : e.low_c;
    if (Number.isNaN(e.low_c)) e.low_c = e.high_c;
  }
  return order.map((d) => byDate.get(d)!);
}

const UA = "(eink-dashboard, rabbitolivestudios@gmail.com)";

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
  const curInfo = getWeatherInfo(0, cur.isDaytime); // code unused; icon/label overridden below
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
      feels_like_c: Math.round(cur.temperature),
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
 * fetched/cached by the caller). The static points lookup is cached per zip.
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
