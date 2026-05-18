const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { parseMonth, parseDay, parseStyleIdx } = fromBuild("src/validate.js");
const { thresholdFromHistogram } = fromBuild("src/convert-1bit.js");
const { moonPhaseHTML } = fromBuild("src/weather-ui.js");
const {
  parseDateParts,
  pickSkylineCity,
  pickSkylineStyle,
  DEFAULT_ROTATE_MIN,
} = fromBuild("src/skyline.js");
const {
  fact4CacheKey,
  fact1CacheKey,
  colorMomentCacheKey,
  skylineCacheKey,
  generationLockKey,
} = fromBuild("src/cache-keys.js");
const { tempColor, batteryIcon } = fromBuild("src/pages/color-weather.js");

test("query param validators clamp to safe defaults", () => {
  assert.equal(parseMonth("12"), 12);
  assert.equal(parseMonth("0"), 1);
  assert.equal(parseMonth("abc"), 1);
  assert.equal(parseDay("31"), 31);
  assert.equal(parseDay("99"), 1);
  assert.equal(parseStyleIdx("9"), 9);
  assert.equal(parseStyleIdx("99"), 0);
  assert.equal(parseStyleIdx(null), undefined);
});

test("skyline date and picker behavior is deterministic", () => {
  const parts = parseDateParts("2026-06-01");
  const opts = { mode: "daily", rotateMin: DEFAULT_ROTATE_MIN, bucket: 0 };
  assert.equal(parts.dayOfYear, 152);
  assert.equal(pickSkylineCity(parts, opts).key, "chicago");
  assert.equal(pickSkylineStyle(parts, opts).key, pickSkylineStyle(parts, opts).key);
});

test("moon phase override is bounded in rendered HTML", () => {
  assert.match(
    moonPhaseHTML("#fff", "#000", 22, new Date("2026-01-01T12:00:00Z"), 99),
    /Waning Crescent/,
  );
});

test("histogram threshold and cache keys stay stable", () => {
  const ramp = Uint8Array.from({ length: 256 }, (_, i) => i);
  assert.equal(thresholdFromHistogram(ramp, 0.5), 127);
  assert.equal(fact4CacheKey("2026-04-27"), "fact4:v4:2026-04-27");
  assert.equal(fact1CacheKey("2026-04-27"), "fact1:v7:2026-04-27");
  assert.equal(colorMomentCacheKey("2026-04-27", "gouache"), "color-moment:v2:2026-04-27:gouache");
  assert.equal(skylineCacheKey("2026-04-27", "daily", 15, 0, true), "skyline:v3:2026-04-27:daily:bw");
  assert.equal(generationLockKey("fact4:v4:2026-04-27"), "gen-lock:v1:fact4:v4:2026-04-27");
});

test("tempColor: blue below 10C, green 10-27C, red above 27C", () => {
  // < 10C -> blue (cold)
  assert.equal(tempColor(-10), "var(--s6-blue)");
  assert.equal(tempColor(0), "var(--s6-blue)");
  assert.equal(tempColor(9), "var(--s6-blue)");
  // 10C <= t <= 27C -> green (comfortable)
  assert.equal(tempColor(10), "var(--s6-green)"); // lower boundary -> green
  assert.equal(tempColor(20), "var(--s6-green)");
  assert.equal(tempColor(27), "var(--s6-green)"); // upper boundary -> green
  // > 27C -> red (warm / hot)
  assert.equal(tempColor(28), "var(--s6-red)");
  assert.equal(tempColor(40), "var(--s6-red)");
  // yellow and black are never returned as a temperature color
  for (let t = -20; t <= 50; t++) {
    const c = tempColor(t);
    assert.notEqual(c, "var(--s6-yellow)");
    assert.notEqual(c, "#000");
  }
});

test("batteryIcon: red at or below 20%, green above, never yellow", () => {
  const red = "rgb(178,19,24)";
  const green = "rgb(18,95,32)";
  const yellow = "rgb(239,222,68)";
  assert.ok(batteryIcon(10, false, 20).includes(red));
  assert.ok(batteryIcon(20, false, 20).includes(red)); // <= 20 -> red
  assert.ok(batteryIcon(21, false, 20).includes(green));
  assert.ok(batteryIcon(50, false, 20).includes(green)); // was yellow
  assert.ok(batteryIcon(90, false, 20).includes(green));
  // a mid-range level no longer renders yellow
  assert.ok(!batteryIcon(35, false, 20).includes(yellow));
});
