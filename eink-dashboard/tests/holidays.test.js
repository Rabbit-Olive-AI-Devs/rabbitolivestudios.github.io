const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { easter, holidayWeek, holidayMoment, withHolidayMood, HOLIDAYS } = fromBuild("src/holidays.js");
const { buildSkylinePrompt, formatSkylineCaption, SKYLINE_CITIES, SKYLINE_STYLES } = fromBuild("src/skyline.js");
const { buildArtBrief } = fromBuild("src/art.js");

const week = (d) => { const w = holidayWeek(d); return w && `${w.holiday.id}${w.isDay ? "!" : ""}`; };

test("Easter dates match the published ones", () => {
  assert.deepEqual(easter(2026), [4, 5]);
  assert.deepEqual(easter(2027), [3, 28]);
  assert.deepEqual(easter(2028), [4, 16]);
  assert.deepEqual(easter(2019), [4, 21]);
});

test("moving holidays land on their real dates", () => {
  const date = (id, y) => HOLIDAYS.find((h) => h.id === id).date(y);
  assert.deepEqual(date("carnaval", 2027), [2, 9]);
  assert.deepEqual(date("carnaval", 2026), [2, 17]);
  assert.deepEqual(date("thanksgiving", 2026), [11, 26]);
  assert.deepEqual(date("thanksgiving", 2027), [11, 25]);
});

test("a holiday week is the 7 days up to and including the holiday", () => {
  assert.equal(week("2026-10-24"), null);
  assert.equal(week("2026-10-25"), "halloween");
  assert.equal(week("2026-10-30"), "halloween");
  assert.equal(week("2026-10-31"), "halloween!");
  assert.equal(week("2026-11-01"), null);
  assert.equal(week("2026-11-02"), null, "Finados is out (owner, 2026-10-17)");
  assert.equal(week("2026-11-20"), "thanksgiving");
  assert.equal(week("2026-11-26"), "thanksgiving!");
  assert.equal(week("2026-12-25"), "christmas!");
  assert.equal(week("2026-12-26"), "new_year", "New Year's week crosses the year end");
  assert.equal(week("2027-01-01"), "new_year!");
  assert.equal(week("2027-01-02"), null);
  assert.equal(week("2027-02-09"), "carnaval!");
});

test("no two holiday weeks overlap in the coming years", () => {
  for (let y = 2026; y <= 2040; y++) {
    const days = new Map();
    for (const h of HOLIDAYS) {
      const [m, d] = h.date(y);
      const end = Date.UTC(y, m - 1, d);
      for (let i = 0; i < 7; i++) {
        const k = end - i * 86400000;
        assert.ok(!days.has(k), `${h.id} overlaps ${days.get(k)} in ${y}`);
        days.set(k, h.id);
      }
    }
  }
});

test("the moment: history on the day, mood in the week, untouched otherwise", () => {
  const day = holidayMoment("2026-10-31");
  assert.equal(day.title, "First Halloween Parade");
  assert.equal(holidayMoment("2026-10-30"), null);
  assert.match(withHolidayMood("2026-10-30", "A bridge."), /^A bridge\. An autumn mood with a hint of Halloween/);
  assert.equal(withHolidayMood("2026-10-31", "A bridge."), "A bridge.");
  assert.equal(withHolidayMood("2026-10-17", "A bridge."), "A bridge.");
  for (const h of HOLIDAYS) assert.ok(h.history.title && h.history.year > 0 && h.history.imagePrompt, h.id);
});

test("the skyline: dressed and captioned in a holiday week, unchanged otherwise", () => {
  const city = SKYLINE_CITIES[0];
  const style = SKYLINE_STYLES.find((s) => s.colorMode === "color");
  assert.match(buildSkylinePrompt(city, style, "2026-10-28"), /jack-o'-lanterns.*full moon/);
  assert.doesNotMatch(buildSkylinePrompt(city, style, "2026-10-28"), /golden-hour/);
  assert.match(buildSkylinePrompt(city, style, "2026-10-17"), /golden-hour/);
  assert.equal(formatSkylineCaption(city, "Oct 28, 2026", "2026-10-28").center, "World Skyline Series - Halloween Week");
  assert.equal(formatSkylineCaption(city, "Oct 31, 2026", "2026-10-31").center, "World Skyline Series - Halloween");
  assert.equal(formatSkylineCaption(city, "Oct 17, 2026", "2026-10-17").center, "World Skyline Series");
  for (const h of HOLIDAYS) assert.match(h.name, /^[\x20-\x7e]+$/, "caption font is ASCII");
});

test("the brief on a holiday carries the holiday moment without an LLM call", async () => {
  const env = {
    CACHE: { get: async () => null, put: async () => {} },
    AI: { run: async () => { throw new Error("no LLM on a holiday"); } },
    PHOTOS: { get: async () => null, head: async () => null },
  };
  const brief = await buildArtBrief(env, "2026-11-26");
  const moment = brief.pictures.find((p) => p.kind === "moment");
  assert.equal(moment.subject.event, "The Plymouth Harvest Feast");
  assert.equal(brief.holiday, "Thanksgiving");
  assert.match(brief.pictures.find((p) => p.kind === "skyline").prompt, /Thanksgiving/);

  // 31 Oct is Álvaro's birthday: the portrait wins (never themed), the skyline still goes Halloween.
  const halloween = await buildArtBrief(env, "2026-10-31");
  assert.deepEqual(halloween.pictures.map((p) => p.kind), ["birthday", "skyline"]);
  assert.doesNotMatch(halloween.pictures[0].prompt, /Halloween|jack-o/);
  assert.match(halloween.pictures[1].prompt, /jack-o'-lanterns/);
});
