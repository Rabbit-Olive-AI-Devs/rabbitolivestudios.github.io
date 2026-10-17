const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { castawayView, dayTimeline, dayNumber, arcFor, islandWeather, matchOn, lightAt } = fromBuild("src/castaway.js");
const { ARCS, ARRIVAL, ROUTINE, MILESTONES } = fromBuild("src/castaway-story.js");
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
  const beats = [...ARRIVAL.days.flat(), ...ARCS.flatMap((a) => a.days.flat()), ...MILESTONES.flatMap((m) => m.arc.days.flat()), ...Object.values(ROUTINE).flat()];
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
