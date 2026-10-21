const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const script = () => import(path.join(__dirname, "..", "scripts", "serial-build.mjs"));
const { encodePNGRGB } = require(path.join(buildDir, "src/png.js"));

const day = { date: "2026-10-21", day: 5 };
const beat = { id: "2026-10-21-1", start: "13:30", title: "CHAPTER", text: ["A short line."], log: "Earlier.", scene: { focus: { fx: 0.5, fy: 0.5 } } };
const picture = (w, h, flat = false) => {
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) rgb.set(flat ? [120, 120, 120] : [(i * 7) % 256, (i * 13) % 256, (i >> 4) % 256], i * 3);
  return encodePNGRGB(rgb, w, h);
};

test("buildBeat: a wide painting becomes a one-request page", async () => {
  const { load, buildBeat } = await script();
  const r = await buildBeat(load(buildDir), day, beat, [], await picture(1600, 630));
  assert.equal(r.error, undefined);
  assert.equal(r.html.split("data:image/png;base64,").length, 2);
});

test("buildBeat: refuses what would not show well, never clips", async () => {
  const { load, buildBeat } = await script();
  const lib = load(buildDir);
  assert.match((await buildBeat(lib, day, { ...beat, text: ["word ".repeat(200)] }, [], await picture(1600, 630))).error, /four lines/);
  assert.match((await buildBeat(lib, day, beat, [], await picture(1600, 630, true))).error, /blank/);
  assert.match((await buildBeat(lib, day, beat, [], await picture(640, 400))).error, /smaller/);
  assert.match((await buildBeat(lib, day, beat, [], new Uint8Array([0xff, 0xd8, 0xff, 0xe0]))).error, /not a PNG/);
});

test("buildBeat: the log shows the two previous beats, newest first", async () => {
  const { load, buildBeat } = await script();
  const earlier = [{ ...beat, start: "07:00" }, { ...beat, start: "10:00" }, { ...beat, start: "11:00", log: "x ".repeat(60) }];
  // the too-long third log line is the newest, so it shows and is refused
  assert.match((await buildBeat(load(buildDir), day, beat, earlier, await picture(1600, 630))).error, /log line/);
  assert.equal((await buildBeat(load(buildDir), day, beat, earlier.slice(0, 2), await picture(1600, 630))).error, undefined);
});
