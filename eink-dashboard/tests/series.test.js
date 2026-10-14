const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const clock = fromBuild("src/series/clock.js");
const keys = fromBuild("src/series/keys.js");
const types = fromBuild("src/series/types.js");

/** A Chicago wall-clock time as a Date. October 2026 is CDT (UTC-5); January is CST (UTC-6). */
function chi(iso, offsetHours) {
  return new Date(Date.parse(iso + "Z") + offsetHours * 3600 * 1000);
}
const CDT = 5, CST = 6;

const SCHEDULE = {
  seasons: [
    { number: 0, show: "THE LONG DRIFT", title: "PROLOGUE", writer: "Claude", premiere: "2026-10-26", chapters: 10, approved: [1, 2], pending: [3] },
    { number: 1, show: "THE LONG DRIFT", title: "S1", writer: "OpenAI", premiere: "2027-01-04", chapters: 13, approved: [1], pending: [] },
  ],
};

test("prologue has 10 chapters, other seasons 13", () => {
  assert.equal(types.plannedChapters(0), 10);
  assert.equal(types.plannedChapters(1), 13);
});

test("weekday scenes change at 08, 10, 12, 14, 16 Chicago", () => {
  const at = (hhmm) => clock.seriesSlot(chi(`2026-10-26T${hhmm}:00`, CDT), SCHEDULE);
  assert.deepEqual(at("08:00"), { kind: "scene", season: 0, chapter: 1, scene: 1, replay: false });
  assert.deepEqual(at("09:59"), { kind: "scene", season: 0, chapter: 1, scene: 1, replay: false });
  assert.deepEqual(at("10:00"), { kind: "scene", season: 0, chapter: 1, scene: 2, replay: false });
  assert.deepEqual(at("16:00"), { kind: "scene", season: 0, chapter: 1, scene: 5, replay: false });
  assert.deepEqual(at("23:59"), { kind: "scene", season: 0, chapter: 1, scene: 5, replay: false });
});

test("Tuesday before 08:00 still shows Monday's last scene; Friday 16:00 is scene 25", () => {
  assert.deepEqual(clock.seriesSlot(chi("2026-10-27T07:59:00", CDT), SCHEDULE),
    { kind: "scene", season: 0, chapter: 1, scene: 5, replay: false });
  assert.deepEqual(clock.seriesSlot(chi("2026-10-30T16:00:00", CDT), SCHEDULE),
    { kind: "scene", season: 0, chapter: 1, scene: 25, replay: false });
});

test("weekend replays the week, then the teaser from Sunday 19:00 to Monday 08:00", () => {
  // 1 Nov 2026 is the clock change: Saturday is CDT, Sunday after 02:00 is CST.
  const at = (iso) => clock.seriesSlot(chi(iso, iso.startsWith("2026-11-01") ? CST : CDT), SCHEDULE);
  assert.deepEqual(at("2026-10-31T07:59:00"), { kind: "scene", season: 0, chapter: 1, scene: 25, replay: false });
  assert.deepEqual(at("2026-10-31T08:00:00"), { kind: "scene", season: 0, chapter: 1, scene: 1, replay: true });
  assert.deepEqual(at("2026-10-31T21:00:00"), { kind: "scene", season: 0, chapter: 1, scene: 14, replay: true });
  assert.deepEqual(at("2026-10-31T23:30:00"), { kind: "scene", season: 0, chapter: 1, scene: 14, replay: true });
  assert.deepEqual(at("2026-11-01T07:00:00"), { kind: "scene", season: 0, chapter: 1, scene: 14, replay: true });
  assert.deepEqual(at("2026-11-01T08:00:00"), { kind: "scene", season: 0, chapter: 1, scene: 15, replay: true });
  assert.deepEqual(at("2026-11-01T18:59:00"), { kind: "scene", season: 0, chapter: 1, scene: 25, replay: true });
  assert.deepEqual(at("2026-11-01T19:00:00"), { kind: "teaser", season: 0, chapter: 1 });
});

test("Monday 00:00-07:59 still shows the previous chapter's teaser", () => {
  // 1 Nov 2026 is the DST change (02:00 CDT -> 01:00 CST); Monday 2 Nov is CST.
  assert.deepEqual(clock.seriesSlot(chi("2026-11-02T07:59:00", CST), SCHEDULE), { kind: "teaser", season: 0, chapter: 1 });
  assert.deepEqual(clock.seriesSlot(chi("2026-11-02T08:00:00", CST), SCHEDULE),
    { kind: "scene", season: 0, chapter: 2, scene: 1, replay: false });
});

test("the DST change night itself holds replay scene 14 on every instant", () => {
  for (const utc of ["2026-11-01T06:30:00Z", "2026-11-01T07:30:00Z", "2026-11-01T08:30:00Z"]) {
    assert.deepEqual(clock.seriesSlot(new Date(utc), SCHEDULE), { kind: "scene", season: 0, chapter: 1, scene: 14, replay: true });
  }
});

test("before a premiere and in a gap the slot is between seasons", () => {
  assert.deepEqual(clock.seriesSlot(chi("2026-10-20T12:00:00", CDT), SCHEDULE), { kind: "between", next: 0, prev: null });
  // Premiere Monday before 08:00 belongs to Sunday night: still before the premiere.
  assert.deepEqual(clock.seriesSlot(chi("2026-10-26T07:00:00", CDT), SCHEDULE), { kind: "between", next: 0, prev: null });
  const late = { seasons: [SCHEDULE.seasons[0]] };
  assert.deepEqual(clock.seriesSlot(chi("2027-01-04T12:00:00", CST), late), { kind: "between", next: null, prev: 0 });
});

test("a season's last Sunday evening is its finale teaser, and the next Monday 08:00 starts the next season", () => {
  assert.deepEqual(clock.seriesSlot(chi("2027-01-03T20:00:00", CST), SCHEDULE), { kind: "teaser", season: 0, chapter: 10 });
  assert.deepEqual(clock.seriesSlot(chi("2027-01-04T08:00:00", CST), SCHEDULE),
    { kind: "scene", season: 1, chapter: 1, scene: 1, replay: false });
});

test("an unapproved chapter falls back to the last approved chapter's final scene", () => {
  const slot = { kind: "scene", season: 0, chapter: 3, scene: 7, replay: false };
  assert.deepEqual(clock.resolveSlot(slot, SCHEDULE, false), { kind: "scene", season: 0, chapter: 2, scene: 25, replay: false });
  assert.deepEqual(clock.resolveSlot(slot, SCHEDULE, true), slot); // previews see pending chapters
  const none = { kind: "teaser", season: 1, chapter: 4 };
  assert.deepEqual(clock.resolveSlot({ kind: "scene", season: 0, chapter: 1, scene: 3, replay: false },
    { seasons: [{ ...SCHEDULE.seasons[0], approved: [] }] }, false), { kind: "between", next: 0, prev: null });
  assert.deepEqual(clock.resolveSlot(none, SCHEDULE, false), { kind: "scene", season: 1, chapter: 1, scene: 25, replay: false });
});

test("scene time labels", () => {
  assert.equal(clock.sceneTimeLabel(1), "MON 08:00");
  assert.equal(clock.sceneTimeLabel(25), "FRI 16:00");
  assert.equal(clock.nextSceneLabel(1), "10:00");
  assert.equal(clock.nextSceneLabel(5), "TUE 08:00");
  assert.equal(clock.nextSceneLabel(25), "MON 08:00");
});

test("R2 keys", () => {
  assert.equal(keys.SCHEDULE_KEY, "series/schedule.json");
  assert.equal(keys.sceneKey(0, 3, 14, false, "desk"), "series/s00/c03/s14-desk.png");
  assert.equal(keys.sceneKey(0, 3, 14, true, "office"), "series/s00/c03/r14-office.png");
  assert.equal(keys.teaserKey(1, 13, "home"), "series/s01/c13/teaser-home.png");
  assert.equal(keys.cardKey(0, "before", "desk"), "series/s00/before-desk.png");
  assert.equal(keys.slotKey({ kind: "between", next: 1, prev: 0 }, "home"), "series/s01/before-home.png");
  assert.equal(keys.slotKey({ kind: "between", next: null, prev: 0 }, "home"), "series/s00/after-home.png");
  assert.equal(keys.slotKey({ kind: "between", next: null, prev: null }, "home"), null);
});
