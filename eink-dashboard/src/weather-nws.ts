/**
 * NWS (api.weather.gov) fallback forecast provider for when Open-Meteo fails.
 *
 * No API key (requires User-Agent header, like alerts.ts). Two-step API:
 *   1. GET /points/{lat},{lon} -> forecast + forecastHourly URLs
 *   2. GET those URLs with ?units=si (Celsius, km/h)
 * Maps into the existing WeatherResponse shape so page code is unchanged.
 */

import type { DailyEntry } from "./types";

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
