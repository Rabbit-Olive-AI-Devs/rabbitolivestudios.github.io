const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const { handleStorybook, handleStorybookFrame } = require(path.join(buildDir, "src/storybook.js"));

const objects = {
  "storybook/days.json": JSON.stringify([{ date: "2026-10-18", day: 2, frames: 2 }, { date: "2026-10-17", day: 1, frames: 1 }]),
  "storybook/days/2026-10-18.json": JSON.stringify({ date: "2026-10-18", day: 2, frames: [{ file: "a.png", caption: "07:00" }, { file: "b.png", caption: "<b>10:00</b>" }] }),
  "storybook/frames/2026-10-18/a.png": "PNGBYTES",
};
const env = (key = "k3y") => ({
  STORYBOOK_KEY: key,
  SERIES: { get: async (k) => (k in objects ? { body: objects[k], json: async () => JSON.parse(objects[k]) } : null) },
});
const req = (q) => new URL(`https://x/storybook?${q}`);

test("storybook: 404 without the key, with a wrong key, or while no key is set", async () => {
  assert.equal((await handleStorybook(env(), req(""))).status, 404);
  assert.equal((await handleStorybook(env(), req("key=nope"))).status, 404);
  assert.equal((await handleStorybook(env(undefined), req("key="))).status, 404);
  assert.equal((await handleStorybookFrame(env(), new URL("https://x/storybook/frame?day=2026-10-18&file=a.png"))).status, 404);
});

test("storybook: contents lists days in order, with the key carried in links", async () => {
  const html = await (await handleStorybook(env(), req("key=k3y"))).text();
  assert.ok(html.indexOf("Day 1") < html.indexOf("Day 2"));
  assert.match(html, /\?key=k3y&amp;day=2026-10-18/);
});

test("storybook: a day shows its frames in order, captions escaped, with navigation", async () => {
  const html = await (await handleStorybook(env(), req("key=k3y&day=2026-10-18"))).text();
  assert.ok(html.indexOf("file=a.png") < html.indexOf("file=b.png"));
  assert.match(html, /&lt;b&gt;10:00&lt;\/b&gt;/);
  assert.match(html, /Day before/);
  assert.equal((await handleStorybook(env(), req("key=k3y&day=2026-10-30"))).status, 404);
  assert.equal((await handleStorybook(env(), req("key=k3y&day=../x"))).status, 404);
});

test("storybook frame: streams the PNG; refuses odd names", async () => {
  const ok = await handleStorybookFrame(env(), new URL("https://x/storybook/frame?key=k3y&day=2026-10-18&file=a.png"));
  assert.equal(ok.status, 200);
  assert.equal(ok.headers.get("content-type"), "image/png");
  assert.equal((await handleStorybookFrame(env(), new URL("https://x/storybook/frame?key=k3y&day=2026-10-18&file=../x.png"))).status, 404);
  assert.equal((await handleStorybookFrame(env(), new URL("https://x/storybook/frame?key=k3y&day=2026-10-18&file=zz.png"))).status, 404);
});
