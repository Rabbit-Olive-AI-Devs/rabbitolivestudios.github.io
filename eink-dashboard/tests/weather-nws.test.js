const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { nwsTextToIcon, parseNwsWind, collapseNwsDaily, normalizeNws } = fromBuild("src/weather-nws.js");

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
  assert.equal(days[0].icon, "clear");
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
  assert.equal(days[0].high_c, 14);
  assert.equal(days[0].icon, "clear_night"); // night period sets the icon when no daytime period exists
});

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
  assert.equal(w.sunrise, "");
});

test("normalizeNws throws when there are no hourly periods", () => {
  const forecast = { properties: { periods: [] } };
  const hourly = { properties: { periods: [] } };
  assert.throws(() => normalizeNws(forecast, hourly, [], 41.88, -87.63, "60606", "Chicago, IL"));
});
