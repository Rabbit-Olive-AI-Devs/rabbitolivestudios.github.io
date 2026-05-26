import { fetchWithTimeout } from "./fetch-timeout";
import type { Env, NWSAlert, CachedValue } from "./types";

const ALERTS_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

const SEVERITY_ORDER: Record<string, number> = {
  Extreme: 0,
  Severe: 1,
  Moderate: 2,
  Minor: 3,
  Unknown: 4,
};

/**
 * Fetch alerts for a specific location.
 * Used by the color weather page (different zip code than E1001).
 */
export async function fetchAlertsForLocation(
  env: Env,
  lat: number,
  lon: number,
  cacheKey: string,
): Promise<NWSAlert[]> {
  const cached = await env.CACHE.get<CachedValue<NWSAlert[]>>(cacheKey, "json");
  if (cached && Date.now() - cached.timestamp < ALERTS_CACHE_TTL_MS) {
    console.log(`Alerts ${cacheKey}: cache hit`);
    return cached.data;
  }

  let attempts = 0;
  while (attempts < 2) {
    attempts++;
    try {
      const res = await fetchWithTimeout(
        `https://api.weather.gov/alerts/active?point=${lat},${lon}`,
        {
          headers: {
            "User-Agent": "(eink-dashboard, rabbitolivestudios@gmail.com)",
            Accept: "application/geo+json",
          },
        },
      );
      if (!res.ok) throw new Error(`NWS returned ${res.status}`);
      const data: any = await res.json();
      const alerts: NWSAlert[] = (data.features ?? []).map((f: any) => ({
        event: f.properties.event ?? "Unknown",
        severity: f.properties.severity ?? "Unknown",
        headline: f.properties.headline ?? "",
        onset: f.properties.onset ?? "",
        expires: f.properties.expires ?? "",
      }));
      alerts.sort(
        (a, b) =>
          (SEVERITY_ORDER[a.severity] ?? 4) - (SEVERITY_ORDER[b.severity] ?? 4)
      );
      await env.CACHE.put(cacheKey, JSON.stringify({ data: alerts, timestamp: Date.now() }), { expirationTtl: 86400 });
      return alerts;
    } catch (err) {
      if (attempts >= 2) console.error("NWS alerts error:", err);
    }
  }

  if (cached) return cached.data;
  return [];
}
