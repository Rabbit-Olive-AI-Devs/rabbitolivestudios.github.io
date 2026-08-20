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
const { withBudget } = fromBuild("src/with-budget.js");
const { pickCleanColorIndex, parseCleanColor, CLEAN_SEQUENCE } = fromBuild("src/clean.js");
const { nextUtcMidnight } = fromBuild("src/cache-guard.js");
const { shiftDateStr } = fromBuild("src/date-utils.js");
const { pickMostRecentDated } = fromBuild("src/stale-cache.js");

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

test("withBudget resolves with value when promise settles inside the budget", async () => {
  const fast = new Promise((resolve) => setTimeout(() => resolve("ok"), 5));
  const result = await withBudget(fast, 100);
  assert.equal(result, "ok");
});

test("withBudget resolves null when the budget elapses first", async () => {
  const slow = new Promise((resolve) => setTimeout(() => resolve("ok"), 100));
  const result = await withBudget(slow, 10);
  assert.equal(result, null);
});

test("withBudget rethrows when the promise rejects inside the budget", async () => {
  const failing = Promise.reject(new Error("boom"));
  await assert.rejects(() => withBudget(failing, 100), /boom/);
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

test("clean sequence is prime-length and covers all 6 pigments", () => {
  // Prime length avoids aliasing with the device's refresh interval.
  assert.equal(CLEAN_SEQUENCE.length, 7);
  const covered = new Set(CLEAN_SEQUENCE);
  for (let i = 0; i < 6; i++) assert.ok(covered.has(i), `pigment ${i} missing`);
});

test("pickCleanColorIndex rotates through the sequence by time", () => {
  // secondsPerFrame=1: each second advances one frame, wrapping the sequence.
  for (let t = 0; t < 14; t++) {
    assert.equal(pickCleanColorIndex(t, 1), CLEAN_SEQUENCE[t % 7]);
  }
  // secondsPerFrame holds each frame longer.
  assert.equal(pickCleanColorIndex(0, 5), CLEAN_SEQUENCE[0]);
  assert.equal(pickCleanColorIndex(4, 5), CLEAN_SEQUENCE[0]);
  assert.equal(pickCleanColorIndex(5, 5), CLEAN_SEQUENCE[1]);
  // Bad/zero secondsPerFrame falls back to 1 (never divides by zero).
  assert.equal(pickCleanColorIndex(3, 0), CLEAN_SEQUENCE[3]);
});

test("parseCleanColor accepts names and indices, rejects garbage", () => {
  assert.equal(parseCleanColor("black"), 0);
  assert.equal(parseCleanColor("WHITE"), 1);
  assert.equal(parseCleanColor(" blue "), 5);
  assert.equal(parseCleanColor("2"), 2);
  assert.equal(parseCleanColor(null), null);
  assert.equal(parseCleanColor(""), null);
  assert.equal(parseCleanColor("purple"), null);
  assert.equal(parseCleanColor("9"), null);
});

// --- AI budget block expiry (DECISIONS #57) ---
// Workers AI resets the free neuron allocation at 00:00 UTC. A fixed-length
// block that lifts before the reset just burns neurons on doomed retries.

test("nextUtcMidnight returns the next 00:00 UTC boundary", () => {
  assert.equal(
    nextUtcMidnight(Date.UTC(2026, 7, 20, 13, 54, 36)),
    Date.UTC(2026, 7, 21, 0, 0, 0, 0),
  );
  // Exactly midnight advances a full day (never returns "now").
  assert.equal(
    nextUtcMidnight(Date.UTC(2026, 7, 20, 0, 0, 0)),
    Date.UTC(2026, 7, 21, 0, 0, 0, 0),
  );
  // One second before the reset.
  assert.equal(
    nextUtcMidnight(Date.UTC(2026, 7, 20, 23, 59, 59)),
    Date.UTC(2026, 7, 21, 0, 0, 0, 0),
  );
  // Month and year rollover.
  assert.equal(
    nextUtcMidnight(Date.UTC(2026, 11, 31, 23, 0, 0)),
    Date.UTC(2027, 0, 1, 0, 0, 0, 0),
  );
});

test("nextUtcMidnight is always in the future and at most 24h out", () => {
  for (const h of [0, 1, 6, 12, 18, 23]) {
    const now = Date.UTC(2026, 7, 20, h, 30, 0);
    const next = nextUtcMidnight(now);
    assert.ok(next > now, `not in the future at ${h}:30`);
    assert.ok(next - now <= 24 * 3600 * 1000, `more than 24h out at ${h}:30`);
  }
});

// --- Date shifting for stale-cache lookback (DECISIONS #57) ---

test("shiftDateStr walks calendar days without drifting", () => {
  assert.equal(shiftDateStr("2026-08-20", -1), "2026-08-19");
  assert.equal(shiftDateStr("2026-08-20", 0), "2026-08-20");
  assert.equal(shiftDateStr("2026-08-20", -7), "2026-08-13");
  // Month boundary.
  assert.equal(shiftDateStr("2026-03-01", -1), "2026-02-28");
  // Year boundary.
  assert.equal(shiftDateStr("2026-01-01", -1), "2025-12-31");
  // Leap day (2028 is a leap year).
  assert.equal(shiftDateStr("2028-03-01", -1), "2028-02-29");
});

test("shiftDateStr is DST-proof (uses UTC, not local time)", () => {
  // US DST transitions — a local-time implementation drifts here.
  assert.equal(shiftDateStr("2026-03-09", -1), "2026-03-08"); // spring forward
  assert.equal(shiftDateStr("2026-11-02", -1), "2026-11-01"); // fall back
});

// --- Version-agnostic stale-cache lookup (DECISIONS #57) ---
// The fallback must survive a cache-key version bump: rebuilding past keys with
// the CURRENT version makes it search for keys that never existed.

test("pickMostRecentDated returns the newest prior day", () => {
  const keys = [
    "fact4:v4:2026-08-18",
    "fact4:v4:2026-08-19",
    "fact4:v4:2026-08-14",
  ];
  assert.equal(pickMostRecentDated(keys, "2026-08-19"), "fact4:v4:2026-08-19");
});

test("pickMostRecentDated survives a cache-key version bump", () => {
  // Only OLD-version keys exist (v4) while the code has moved to v5.
  // A key-rebuilding fallback would look for fact4:v5:* and find nothing.
  const keys = ["fact4:v4:2026-08-18", "fact4:v4:2026-08-19"];
  assert.equal(pickMostRecentDated(keys, "2026-08-19"), "fact4:v4:2026-08-19");
});

test("pickMostRecentDated prefers the higher version on the same day", () => {
  const keys = ["fact4:v4:2026-08-19", "fact4:v5:2026-08-19"];
  assert.equal(pickMostRecentDated(keys, "2026-08-19"), "fact4:v5:2026-08-19");
  // Order of the listing must not matter.
  assert.equal(pickMostRecentDated(keys.reverse(), "2026-08-19"), "fact4:v5:2026-08-19");
});

test("pickMostRecentDated ignores dates after the bound", () => {
  const keys = ["fact4:v4:2026-08-20", "fact4:v4:2026-08-21"];
  assert.equal(pickMostRecentDated(keys, "2026-08-19"), null);
});

test("pickMostRecentDated honours the lookback window", () => {
  const keys = ["fact4:v4:2026-08-01"];
  // Window is maxDaysBack days ending at the bound, inclusive.
  assert.equal(pickMostRecentDated(keys, "2026-08-19", { maxDaysBack: 7 }), null);
  assert.equal(
    pickMostRecentDated(keys, "2026-08-01", { maxDaysBack: 1 }),
    "fact4:v4:2026-08-01",
  );
});

test("pickMostRecentDated separates bw from colour skylines", () => {
  const keys = [
    "skyline:v3:2026-08-19:daily",
    "skyline:v3:2026-08-19:daily:bw",
  ];
  const bw = (n) => n.endsWith(":bw");
  assert.equal(pickMostRecentDated(keys, "2026-08-19", { accept: bw }),
    "skyline:v3:2026-08-19:daily:bw");
  assert.equal(pickMostRecentDated(keys, "2026-08-19", { accept: (n) => !bw(n) }),
    "skyline:v3:2026-08-19:daily");
});

test("pickMostRecentDated tolerates undated and unrelated keys", () => {
  const keys = ["gen-lock:v1:fact4:v4:2026-08-19", "fact4:v4", "", "fact4:v4:2026-08-18"];
  assert.equal(pickMostRecentDated(keys, "2026-08-19"), "gen-lock:v1:fact4:v4:2026-08-19");
  // With locks filtered out, the real entry wins.
  assert.equal(
    pickMostRecentDated(keys, "2026-08-19", { accept: (n) => !n.startsWith("gen-lock:") }),
    "fact4:v4:2026-08-18",
  );
});
