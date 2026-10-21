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
