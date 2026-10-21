const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const script = () => import(path.join(__dirname, "..", "scripts", "serial-mailbox.mjs"));
const lib = async () => (await import(path.join(__dirname, "..", "scripts", "serial-build.mjs"))).load(buildDir);

const beat = (n, over = {}) => ({ id: `2026-10-22-${n}`, start: `${String(6 + n * 3).padStart(2, "0")}:00`, title: "CHAPTER", text: [`Line number ${n} of the day.`], log: `Log ${n}`, scene: { focus: { fx: 0.5, fy: 0.4 } }, ...over });

test("beatText: the match variant only once the match is done", async () => {
  const { beatText } = await script();
  const b = beat(1, { live: "match", text_done: ["It ended {{us}}-{{them}} against {{opponent}}."] });
  assert.equal(beatText(b, null).variant, "default");
  assert.equal(beatText(b, { done: false, us: 0, them: 0, opponent: "PAL" }).variant, "default");
  const done = beatText(b, { done: true, us: 2, them: 1, opponent: "PAL" });
  assert.deepEqual(done, { text: ["It ended 2-1 against PAL."], variant: "done:2-1" });
  assert.equal(beatText(beat(2), { done: true, us: 2, them: 1, opponent: "PAL" }).variant, "default");
});

test("needsBuild: new picture, changed live text, or a review still owed", async () => {
  const { needsBuild } = await script();
  assert.equal(needsBuild(undefined, "a", "default"), true);
  assert.equal(needsBuild({ sha: "a", verdict: "PASS", variant: "default" }, "a", "default"), false);
  assert.equal(needsBuild({ sha: "a", verdict: "PASS", variant: "default" }, "b", "default"), true);
  assert.equal(needsBuild({ sha: "a", verdict: "PASS", variant: "default" }, "a", "done:2-1"), true);
  assert.equal(needsBuild({ sha: "a", verdict: "REVISE" }, "a", "default"), false);
  assert.equal(needsBuild({ sha: "a", verdict: "ERROR" }, "a", "default"), false);
  assert.equal(needsBuild({ sha: "a", verdict: "UNREVIEWED", tries: 1 }, "a", "default"), true);
  assert.equal(needsBuild({ sha: "a", verdict: "UNREVIEWED", tries: 3 }, "a", "default"), false);
});

test("writeThrough: the Sunday of the week starting 12 or more days ahead", async () => {
  const { writeThrough } = await script();
  assert.equal(writeThrough("2026-10-28"), "2026-11-15"); // Wed: +12 = Mon 9 Nov
  assert.equal(writeThrough("2026-10-21"), "2026-11-08"); // Wed: +12 = Mon 2 Nov
  assert.equal(writeThrough("2026-10-22"), "2026-11-15"); // Thu: +12 = Tue 3 Nov, next Monday 9 Nov
});

test("checkDay: accepts a good day, names each problem by beat id", async () => {
  const { checkDay } = await script();
  const l = await lib();
  const day = { date: "2026-10-22", beats: [beat(1), beat(2)] };
  assert.deepEqual(checkDay(l, day, new Set(), new Set()), []);
  const bad = { date: "2026-10-22", beats: [
    beat(1, { text: ["word ".repeat(200)] }),
    beat(2, { log: "café — dash" }),
    beat(3, { scene: {} }),
    beat(4, { text: ["Line number 2 of the day."] }),
  ] };
  const p = checkDay(l, bad, new Set(["log 1"]), new Set()).join("\n");
  assert.match(p, /2026-10-22-1: story text longer/);
  assert.match(p, /2026-10-22-2: no glyph/);
  assert.match(p, /2026-10-22-3: focus/);
  assert.match(p, /2026-10-22-4: a line repeats/);
  assert.match(p, /2026-10-22-1: a line repeats/); // its log is another day's
});

test("checkDay: a match beat must also fit with the widest score", async () => {
  const { checkDay } = await script();
  const b = beat(1, { live: "match", text_done: ["x ".repeat(150) + "{{opponent}}"] });
  assert.match(checkDay(await lib(), { date: "2026-10-22", beats: [b] }, new Set(), new Set()).join("\n"), /story text longer/);
});

test("standInDue: only near the start, only without a painting, only with a passed library place", async () => {
  const { standInDue } = await script();
  const b = beat(1, { start: "13:00", scene: { setting_id: "p1", focus: { fx: 0.5, fy: 0.5 } } });
  const ok = () => true;
  assert.equal(standInDue(b, 12 * 60 + 49, undefined, ok), false); // 11 min early
  assert.equal(standInDue(b, 12 * 60 + 50, undefined, ok), true);
  assert.equal(standInDue(b, 14 * 60, { verdict: "REVISE" }, ok), true);
  assert.equal(standInDue(b, 14 * 60, { verdict: "PASS" }, ok), false);
  assert.equal(standInDue(b, 14 * 60, { verdict: "STAND_IN" }, ok), false);
  assert.equal(standInDue(b, 14 * 60, undefined, () => false), false);
  assert.equal(standInDue(beat(1, { start: "13:00" }), 14 * 60, undefined, ok), false); // no setting
});

test("needsBuild: a stand-in waits for a new painting, never re-reviews a rejected one", async () => {
  const { needsBuild } = await script();
  assert.equal(needsBuild({ sha: "library:p1", verdict: "STAND_IN" }, "new", "default"), true);
  assert.equal(needsBuild({ sha: "rejected", verdict: "STAND_IN" }, "rejected", "default"), false);
});

test("checkDay: with a settings library, every beat names a known setting", async () => {
  const { checkDay } = await script();
  const l = await lib();
  const day = { date: "2026-10-22", beats: [beat(1, { scene: { setting_id: "p1", focus: { fx: 0.5, fy: 0.5 } } }), beat(2)] };
  const p = checkDay(l, day, new Set(), new Set(), { p1: {} });
  assert.deepEqual(p, ["2026-10-22-2: scene.setting_id is not in the season's settings.json"]);
});
