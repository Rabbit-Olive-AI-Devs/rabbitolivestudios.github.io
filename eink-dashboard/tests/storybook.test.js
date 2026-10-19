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
