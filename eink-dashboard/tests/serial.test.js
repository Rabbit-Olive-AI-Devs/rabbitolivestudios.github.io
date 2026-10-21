const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { drawSerialPage, storyTextFits, SERIAL_PICTURE } = fromBuild("src/hud/castaway-page.js");
const { colorAt } = fromBuild("src/hud/canvas.js");
const { currentBeat, serialPage, serialDayKey, serialPageKey } = fromBuild("src/serial.js");

const solid = (r, g, b) => {
  const rgb = new Uint8Array(SERIAL_PICTURE.w * SERIAL_PICTURE.h * 3);
  for (let i = 0; i < rgb.length; i += 3) rgb.set([r, g, b], i);
  return { w: SERIAL_PICTURE.w, h: SERIAL_PICTURE.h, rgb };
};
const view = { day: 5, time: "13:30", chapter: "CHAPTER", text: "One short line.", log: [{ time: "10:00", text: "Earlier." }] };

test("serial page: painting in colour in the band, rule and top bar in black", () => {
  const c = drawSerialPage(view, solid(200, 30, 30));
  assert.deepEqual(colorAt(c, 200 * c.w + 400), [200, 30, 30]);
  for (let x = 0; x < c.w; x += 50) assert.deepEqual(colorAt(c, 352 * c.w + x), [0, 0, 0]);
  let ink = 0;
  for (let i = 0; i < 36 * c.w; i++) if (colorAt(c, i)[0] === 0) ink++;
  assert.ok(ink > 100, "top bar drawn");
});

test("serial page: refuses a picture that is not 800x315", () => {
  assert.throws(() => drawSerialPage(view, { w: 800, h: 300, rgb: new Uint8Array(800 * 300 * 3) }));
});

test("storyTextFits: fits, or says why", () => {
  assert.equal(storyTextFits("CHAPTER", "A short story line.", view.log), null);
  assert.match(storyTextFits("CHAPTER", "word ".repeat(200), []), /four lines/);
  assert.match(storyTextFits("X".repeat(80), "ok", []), /chapter/);
  assert.match(storyTextFits("CHAPTER", "ok", [{ time: "10:00", text: "a much too long log line for the narrow column" }]), /log line/);
  assert.match(storyTextFits("CHAPTER", "ok", [view.log[0], view.log[0], view.log[0]]), /two log/);
});

const day = (date, ...starts) => ({ date, day: 5, beats: starts.map((s, i) => ({ id: `${date}-${i + 1}`, start: s, key: serialPageKey(`${date}-${i + 1}`) })) });
const at = (hhmm) => Number(hhmm.slice(0, 2)) * 60 + Number(hhmm.slice(3));

test("currentBeat: latest started today, else yesterday's last, else none", () => {
  const today = day("2026-10-21", "16:00", "13:30");
  assert.equal(currentBeat(today, null, at("15:59")).start, "13:30");
  assert.equal(currentBeat(today, null, at("16:00")).start, "16:00");
  assert.equal(currentBeat(today, null, at("12:00")), null);
  assert.equal(currentBeat(day("2026-10-22", "07:00"), today, at("03:00")).start, "16:00");
  assert.equal(currentBeat(null, null, at("03:00")), null);
});

const bucket = (objects, fail = false) => ({
  get: async (key) => {
    if (fail) throw new Error("R2 down");
    if (!(key in objects)) return null;
    const body = objects[key];
    return { text: async () => body, json: async () => JSON.parse(body) };
  },
});
// 2026-10-21 14:00 CDT = 19:00 UTC
const now = new Date("2026-10-21T19:00:00Z");

test("serialPage: serves the stored page of the current beat", async () => {
  const objects = { [serialDayKey("2026-10-21")]: JSON.stringify(day("2026-10-21", "13:30")), [serialPageKey("2026-10-21-1")]: "<html>beat</html>" };
  assert.equal(await serialPage({ SERIES: bucket(objects) }, now), "<html>beat</html>");
});

test("serialPage: null (engine fallback) when anything is missing or broken", async () => {
  const d = serialDayKey("2026-10-21");
  assert.equal(await serialPage({ SERIES: bucket({}) }, now), null);
  assert.equal(await serialPage({ SERIES: bucket({ [d]: "not json" }) }, now), null);
  assert.equal(await serialPage({ SERIES: bucket({ [d]: JSON.stringify(day("2026-10-21", "13:30")) }) }, now), null);
  assert.equal(await serialPage({ SERIES: bucket({}, true) }, now), null);
  // before the first beat of the first day
  assert.equal(await serialPage({ SERIES: bucket({ [d]: JSON.stringify(day("2026-10-21", "13:30")) }) }, new Date("2026-10-21T17:00:00Z")), null);
});

const { serialProblems, serialHealth } = fromBuild("src/serial.js");
const dayFile = (date, built, planned, extra = {}) => JSON.stringify({ ...day(date, ...["07:00", "10:00", "13:00", "16:00", "19:00", "22:00"].slice(0, built)), planned, ...extra });
// 2026-10-23 07:05 CDT
const morning = new Date("2026-10-23T12:05:00Z");

test("serialProblems: quiet when today is ready, on the first day, between seasons, or before any serial", async () => {
  const k = serialDayKey;
  assert.deepEqual(await serialProblems({ SERIES: bucket({ [k("2026-10-23")]: dayFile("2026-10-23", 6, 6) }) }, morning), []);
  assert.deepEqual(await serialProblems({ SERIES: bucket({ [k("2026-10-23")]: dayFile("2026-10-23", 1, 4, { same_day: true }) }) }, morning), []);
  assert.deepEqual(await serialProblems({ SERIES: bucket({ [k("2026-10-22")]: dayFile("2026-10-22", 6, 6, { season_end: true }) }) }, morning), []);
  assert.deepEqual(await serialProblems({ SERIES: bucket({}) }, morning), []);
});

test("serialProblems: says how many pages are missing, without story", async () => {
  const k = serialDayKey;
  const [p] = await serialProblems({ SERIES: bucket({ [k("2026-10-23")]: dayFile("2026-10-23", 2, 6) }) }, morning);
  assert.match(p, /2 of 6 paintings in for 2026-10-23/);
  const [q] = await serialProblems({ SERIES: bucket({ [k("2026-10-22")]: dayFile("2026-10-22", 6, 6) }) }, morning);
  assert.match(q, /no pages for 2026-10-23/);
});

test("serialHealth: ready/planned for today and tomorrow", async () => {
  const h = await serialHealth({ SERIES: bucket({ [serialDayKey("2026-10-23")]: dayFile("2026-10-23", 2, 6) }) }, morning);
  assert.deepEqual(h, { "2026-10-23": "2/6", "2026-10-24": "none" });
});

test("serialProblems and health: a stand-in is not counted as a painting", async () => {
  const k = serialDayKey("2026-10-23");
  const d = JSON.parse(dayFile("2026-10-23", 6, 6));
  d.beats[5].stand_in = true;
  const env = { SERIES: bucket({ [k]: JSON.stringify(d) }) };
  assert.match((await serialProblems(env, morning))[0], /5 of 6 paintings/);
  assert.equal((await serialHealth(env, morning))["2026-10-23"], "5/6");
});
