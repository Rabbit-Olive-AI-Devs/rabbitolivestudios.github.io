const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { castawayView, castawayMoment, chicagoParts, dayTimeline, dayNumber, arcFor, islandWeather, matchOn, lightAt, rotate } = fromBuild("src/castaway.js");
const { ARCS, ARRIVAL, ROUTINE, MILESTONES, DATED_EGGS, DAY_EGGS, RARE_EGGS, MORNING_LOGS, PLANET_NIGHTS, SETTINGS } = fromBuild("src/castaway-story.js");
const { drawCastawayPage, wrap } = fromBuild("src/hud/castaway-page.js");
const { measureType } = fromBuild("src/hud/type.js");
const { FACES } = fromBuild("src/hud/fonts-data.js");

test("day numbers start on the epoch and storylines never repeat back to back", () => {
  assert.equal(dayNumber("2026-10-17"), 1);
  assert.equal(dayNumber("2027-10-17"), 366);
  assert.equal(arcFor(1).arc.id, "arrival");
  assert.equal(arcFor(3).arc.id, "arrival");
  assert.equal(arcFor(30).arc.id, "mast");
  assert.deepEqual([arcFor(98).arc.id, arcFor(100).arc.id, arcFor(100).index], ["parcel", "parcel", 2]);
  let prev = null;
  for (let d = 4; d < 800; d++) {
    const { arc, index } = arcFor(d);
    if (MILESTONES.some((m) => m.arc === arc)) continue;
    if (index === 0) { assert.notEqual(arc.id, prev, `day ${d}`); prev = arc.id; }
    assert.ok(index < arc.days.length);
  }
});

test("every beat fits the page and uses only glyphs the fonts have", () => {
  // The templated lines at their widest: a long planet name, four-digit days, the longest setting.
  const longest = SETTINGS.reduce((a, b) => (b.length > a.length ? b : a));
  const templated = [...MORNING_LOGS.map((f) => f({ day: 1000, battery: 64, setting: longest, value: 95 })),
    ...PLANET_NIGHTS.map((f) => f("OGLE-2005-BLG-390L B"))];
  const beats = [...templated, ...ARRIVAL.days.flat(), ...ARCS.flatMap((a) => a.days.flat()), ...MILESTONES.flatMap((m) => m.arc.days.flat()), ...Object.values(ROUTINE).flat(),
    ...DATED_EGGS.map((e) => e.beat), ...DAY_EGGS.map((e) => e.beat), ...RARE_EGGS];
  for (const b of beats) {
    assert.ok(wrap("plex13r", b.text, 548).length <= 3, `too long: ${b.text}`);
    assert.ok(measureType("plex12m", b.log) <= 192, `log too long: ${b.log}`);
    for (const ch of b.text + b.log) assert.ok(FACES.plex13r.glyphs[ch] || ch === " ", `no glyph for "${ch}" in: ${b.text}`);
  }
  for (const a of ARCS) assert.ok(measureType("plex13s", a.title) <= 548, a.title);
});

test("the page draws at every quarter hour for two months, through holidays and birthdays", () => {
  const start = Date.UTC(2026, 9, 17, 5);
  for (let i = 0; i < 61 * 96; i += 7) {
    const v = castawayView(new Date(start + i * 15 * 60_000), { planet: "Kepler-22 b", weather: ["clear", "cloudy", "rain", "storm"][i % 4] });
    assert.ok(v.text.length > 0 && v.day >= 1);
    drawCastawayPage(v);
  }
});

test("day 1 begins with the arrival and has nothing earlier in the log", () => {
  const v = castawayView(new Date("2026-10-17T12:00:00Z"));
  assert.match(v.text, /crawls out of the surf/);
  assert.equal(v.log.length, 0);
});

test("nights are for sleeping, the calendar adds its moments", () => {
  assert.equal(castawayView(new Date("2026-11-10T09:00:00Z")).scene.pose, "sleep");   // 03:00 Chicago
  const bday = dayTimeline("2026-10-20", {}).map((m) => m.beat.log);
  assert.ok(bday.some((l) => /Cake for THIAGO/.test(l)));
  assert.ok(dayTimeline("2026-10-31", {}).some((m) => /Halloween/.test(m.beat.text)));
  assert.ok(dayTimeline("2026-10-27", {}).some((m) => (m.beat.props || []).includes("pumpkin")));
});

test("weather, light and a Flamengo match", () => {
  assert.equal(islandWeather(0), "clear");
  assert.equal(islandWeather(3), "cloudy");
  assert.equal(islandWeather(63), "rain");
  assert.equal(islandWeather(73), "rain");
  assert.equal(islandWeather(95), "storm");
  assert.equal(lightAt(420, 430, 1100), "dawn");
  assert.equal(lightAt(720, 430, 1100), "day");
  assert.equal(lightAt(1100, 430, 1100), "dusk");
  assert.equal(lightAt(60, 430, 1100), "night");
  const kickoff = Date.parse("2026-10-25T21:30:00Z");   // 16:30 Chicago
  const state = { results: [{ id: "1", at: kickoff, league: "x", home: { id: "819", name: "Flamengo", abbr: "FLA" }, away: { id: "2", name: "Palmeiras", abbr: "PAL" }, homeScore: 2, awayScore: 1, done: true }], fixtures: [] };
  const match = matchOn(state, "2026-10-25");
  assert.deepEqual([match.opponent, match.us, match.them], ["PAL", 2, 1]);
  const day = dayTimeline("2026-10-25", { match }).map((m) => m.beat.log);
  assert.ok(day.includes("MENGO! 2-1"));
  assert.equal(matchOn(state, "2026-10-26"), null);
});

test("every calendar moment in a whole year fits and has glyphs (holidays, birthdays, the planet line)", () => {
  for (let i = 0; i < 400; i++) {
    const d = new Date(Date.UTC(2026, 9, 17 + i)).toISOString().slice(0, 10);
    for (const m of dayTimeline(d, { planet: "TOI-1452 b" })) {
      assert.ok(wrap("plex13r", m.beat.text, 548).length <= 3, `${d}: ${m.beat.text}`);
      assert.ok(measureType("plex12m", m.beat.log) <= 192, `${d}: ${m.beat.log}`);
      for (const ch of m.beat.text + m.beat.log) assert.ok(FACES.plex13r.glyphs[ch] || ch === " ", `${d}: no glyph "${ch}"`);
    }
  }
});

test("easter eggs turn up: dated ones on their day, rare ones now and then", () => {
  assert.ok(dayTimeline("2027-10-21", {}).some((m) => m.beat === DATED_EGGS[0].beat));
  let rare = 0;
  for (let i = 4; i < 370; i++) {
    const d = new Date(Date.UTC(2026, 9, 17 + i)).toISOString().slice(0, 10);
    if (dayTimeline(d, {}).some((m) => RARE_EGGS.includes(m.beat))) rare++;
  }
  assert.ok(rare > 10 && rare < 45, `rare eggs in a year: ${rare}`);
});

test("the moment key changes exactly when the story moves on, and gives its start", () => {
  const ctx = { sunrise: 433, sunset: 1088 };
  let prev = null;
  for (let t = Date.UTC(2026, 9, 18, 5); t < Date.UTC(2026, 9, 21, 5); t += 60_000) {
    const now = new Date(t);
    const v = castawayView(now, ctx);
    const story = JSON.stringify([v.text, v.log]);
    const m = castawayMoment(now, ctx);
    const p = chicagoParts(m.start);
    assert.ok(m.start.getTime() <= t, now.toISOString());
    assert.equal(m.key, `${p.date}T${String(Math.floor(p.min / 60)).padStart(2, "0")}:${String(p.min % 60).padStart(2, "0")}`);
    if (prev) assert.equal(m.key !== prev.key, story !== prev.story, now.toISOString());
    prev = { key: m.key, story };
  }
  // The context moves moments: a later sunrise starts the morning one later.
  const at = new Date("2026-10-19T12:20:00Z");   // 07:20 Chicago
  assert.notEqual(castawayMoment(at, { sunrise: 430 }).key, castawayMoment(at, { sunrise: 445 }).key);
});

test("routine: four slots a day besides the morning log, and no line back before its pool is used up", () => {
  const pools = [ROUTINE.night, ROUTINE.afternoon, ROUTINE.dusk, ROUTINE.evening, MORNING_LOGS, PLANET_NIGHTS];
  for (const pool of pools) {
    assert.ok(pool.length >= 20, `pool of ${pool.length}`);
    for (let start = -5; start < 400; start += 37) {
      const seen = new Set();
      for (let d = start; d < start + pool.length; d++) seen.add(rotate(pool, d, 3));
      assert.equal(seen.size, pool.length, `pool of ${pool.length} from day ${start}`);
      assert.equal(rotate(pool, start, 3), rotate(pool, start + pool.length, 3));
    }
  }
  for (const b of ROUTINE.night) assert.equal(b.pose, "sleep", b.log);
  const lastDay = new Map();
  for (let i = 3; i < 200; i++) {
    const d = new Date(Date.UTC(2026, 9, 17 + i)).toISOString().slice(0, 10);
    const day = dayNumber(d);
    const timeline = dayTimeline(d, { planet: "Kepler-22 b" });
    assert.equal(timeline.filter((m) => m.routine).length, 5, d);
    for (const m of timeline.filter((x) => x.routine)) {
      const key = m.beat.text.replace(/\d+/g, "#");
      const pool = Object.values(ROUTINE).find((p) => p.includes(m.beat));
      if (pool && lastDay.has(key)) assert.ok(day - lastDay.get(key) >= pool.length, `${d}: back after ${day - lastDay.get(key)} days`);
      lastDay.set(key, day);
    }
  }
});

test("the log lists only story moments", () => {
  const ctx = { planet: "Kepler-22 b" };
  for (const at of ["2026-10-25T04:30:00Z", "2026-11-03T22:10:00Z", "2026-11-04T03:30:00Z", "2026-11-04T19:50:00Z"]) {
    const now = new Date(at);
    const { date, min } = chicagoParts(now);
    const today = dayTimeline(date, ctx);
    const idx = today.reduce((acc, m, i) => (m.min <= min ? i : acc), 0);
    const yesterday = dayTimeline(new Date(Date.parse(date) - 86_400_000).toISOString().slice(0, 10), ctx);
    const story = [...yesterday, ...today.slice(0, idx)].filter((m) => !m.routine).map((m) => m.beat.log);
    assert.deepEqual(castawayView(now, ctx).log.map((e) => e.text), story.slice(-2).reverse(), at);
  }
});
