const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const load = () => import(path.join(__dirname, "..", "scripts", "storybook.mjs"));

test("storybook: takes the inlined PNG and refuses a page without one", async () => {
  const { extractPng } = await load();
  const png = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3]);
  const page = `<body><img src="data:image/png;base64,${png.toString("base64")}" width="800"></body>`;
  assert.deepEqual(extractPng(page), png);
  assert.equal(extractPng("<body><p>This page could not be drawn.</p></body>"), null);
  assert.equal(extractPng(`<img src="data:image/png;base64,${Buffer.from("not a png").toString("base64")}">`), null);
});

test("storybook: Chicago file names, day numbers and DST-safe names", async () => {
  const { chicagoStamp, dayNumber, frameName } = await load();
  assert.deepEqual(chicagoStamp(new Date("2026-10-19T05:05:00Z")), { date: "2026-10-19", time: "00:05", stem: "0005" });
  assert.deepEqual(chicagoStamp(new Date("2026-12-01T05:59:00Z")), { date: "2026-11-30", time: "23:59", stem: "2359" });
  assert.equal(dayNumber("2026-10-17"), 1);
  assert.equal(dayNumber("2026-11-02"), 17);
  assert.equal(frameName("0130", new Set()), "0130.png");
  assert.equal(frameName("0130", new Set(["0130.png", "0130-2.png"])), "0130-3.png");
});

test("storybook: backfill steps every 10 minutes from Chicago midnight and stops at now", async () => {
  const { backfillInstants } = await load();
  const all = backfillInstants("2026-10-17", "2026-10-18", new Date("2027-01-01T00:00:00Z"));
  assert.equal(all.length, 288);
  assert.equal(all[0].toISOString(), "2026-10-17T05:00:00.000Z");
  assert.equal(all.at(-1).toISOString(), "2026-10-19T04:50:00.000Z");
  const partial = backfillInstants("2026-10-17", "2026-10-17", new Date("2026-10-17T06:00:00Z"));
  assert.equal(partial.length, 7);
  // The autumn DST day has 25 hours, so 150 steps.
  assert.equal(backfillInstants("2026-11-01", "2026-11-01", new Date("2027-01-01T00:00:00Z")).length, 150);
});

test("storybook: the live job fetches only when a new moment starts (or the moment is unknown)", async () => {
  const { needsFrame, annotateRows } = await load();
  const moment = { key: "2026-10-19T10:00", start: new Date("2026-10-19T15:00:00Z") };
  assert.equal(needsFrame([], moment), true);
  assert.equal(needsFrame([{ moment: "2026-10-19T07:12" }], moment), true);
  assert.equal(needsFrame([{ moment: "2026-10-19T07:12" }, { moment: "2026-10-19T10:00" }], moment), false);
  assert.equal(needsFrame([{ moment: "2026-10-19T10:00" }], null), true);
  const rows = await annotateRows([{ at: "2026-10-19T15:10:00.000Z", file: "1010.png" }], async () => moment);
  assert.deepEqual(rows, [{ at: "2026-10-19T15:10:00.000Z", file: "1010.png", moment: moment.key, moment_start: "2026-10-19T15:00:00.000Z" }]);
});

test("storybook: without the engine, a frame counts as new only when the picture below the top bar changed", async () => {
  const { pictureChanged } = await load();
  const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
  const { encodePNG1Bit } = require(path.join(buildDir, "src/png.js"));
  const frame = async (dots) => {
    const px = new Uint8Array(800 * 480);
    for (const [x, y] of dots) px[y * 800 + x] = 1;
    return Buffer.from(await encodePNG1Bit(px, 800, 480));
  };
  const base = await frame([[5, 100]]);
  assert.equal(pictureChanged(base, await frame([[5, 100], [300, 10]])), false);   // only the clock bar
  assert.equal(pictureChanged(base, await frame([[5, 100], [300, 200]])), true);
  assert.equal(pictureChanged(base, Buffer.from("not a png")), true);
});
