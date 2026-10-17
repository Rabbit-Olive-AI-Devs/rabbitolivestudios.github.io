const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { familyStyleFor, yearsAgo, familyCaption, storeFamily } = fromBuild("src/family.js");
const { momentStyleFor } = fromBuild("src/art-styles.js");
const { parseDateParts, pickSkylineStyle, DEFAULT_ROTATE_MIN } = fromBuild("src/skyline.js");
const { buildArtBrief } = fromBuild("src/art.js");
const { encodePNGRGB } = fromBuild("src/png.js");
const { familyCacheKey } = fromBuild("src/cache-keys.js");

/** A fake R2 bucket over a Map: get/head/put/delete as the Worker uses them. */
function fakeBucket(entries = {}) {
  const m = new Map(Object.entries(entries));
  const obj = (v) => ({
    arrayBuffer: async () => (typeof v === "string" ? new TextEncoder().encode(v).buffer : v),
    json: async () => JSON.parse(v),
  });
  return {
    m,
    get: async (k) => (m.has(k) ? obj(m.get(k)) : null),
    head: async (k) => (m.has(k) ? {} : null),
    put: async (k, v) => { m.set(k, v); },
    delete: async (k) => { m.delete(k); },
  };
}

async function fakeEnv(photos) {
  const png = await encodePNGRGB(new Uint8Array(800 * 480 * 3).fill(200), 800, 480);
  const kv = new Map();
  return {
    kv,
    PHOTOS: fakeBucket(photos),
    CACHE: { get: async (k) => kv.get(k) ?? null, put: async (k, v) => { kv.set(k, v); } },
    AI: { run: async () => { throw new Error("no AI in tests"); } },
    IMAGES: { input: () => ({ transform: () => ({ output: async () => ({ response: () => new Response(png) }) }) }) },
  };
}

test("the family style is never the day's moment or skyline style", () => {
  for (let i = 0; i < 400; i++) {
    const d = new Date(Date.UTC(2026, 9, 17 + i)).toISOString().slice(0, 10);
    const id = familyStyleFor(d).id;
    assert.notEqual(id, momentStyleFor(d).id, d);
    assert.notEqual(id, pickSkylineStyle(parseDateParts(d), { mode: "daily", rotateMin: DEFAULT_ROTATE_MIN, bucket: 0 }).key, d);
  }
});

test("years ago and the caption", () => {
  assert.equal(yearsAgo("2019-10-15", "2026-10-17"), 7);
  assert.equal(yearsAgo("2019-10-20", "2026-10-17"), 6);
  assert.equal(yearsAgo("2025-11-20", "2026-10-17"), 1, "at least 1");
  assert.deepEqual(familyCaption("2019-10-15", "2026-10-17"), { left: "15 Oct 2019", center: "Family Memory", right: "7 years ago" });
  assert.equal(familyCaption("2025-10-17", "2026-10-17").right, "1 year ago");
});

test("the brief carries the family picture only when the photo is up, never themed", async () => {
  const none = await buildArtBrief(await fakeEnv({}), "2026-11-26");
  assert.ok(!none.pictures.some((p) => p.kind === "family"));

  const env = await fakeEnv({ "family/2026-10-20/photo.jpg": "x", "family/2026-10-31/photo.jpg": "x" });
  const bday = await buildArtBrief(env, "2026-10-20");
  assert.deepEqual(bday.pictures.map((p) => p.kind), ["birthday", "skyline", "family"]);
  const fam = bday.pictures.find((p) => p.kind === "family");
  assert.equal(fam.file, "2026-10-20-family.png");
  assert.equal(fam.references, 1);

  const halloween = (await buildArtBrief(env, "2026-10-31")).pictures.find((p) => p.kind === "family");
  assert.doesNotMatch(halloween.prompt, /Halloween|jack-o|pumpkin/i);
});

test("storeFamily prefers Jarvis, falls back to the photo, cleans up after itself", async () => {
  const meta = JSON.stringify({ taken: "2019-10-15" });
  const both = await fakeEnv({ "family/2026-10-18/photo.jpg": "p", "family/2026-10-18/photo.json": meta, "art/2026-10-18/family": "a" });
  assert.equal(await storeFamily(both, "2026-10-18"), "jarvis");
  assert.ok(both.kv.get(familyCacheKey("2026-10-18")));
  assert.ok(both.PHOTOS.m.has("art/used/2026-10-18/family"));
  for (const k of ["family/2026-10-18/photo.jpg", "family/2026-10-18/photo.json", "art/2026-10-18/family"]) assert.ok(!both.PHOTOS.m.has(k), k);

  const photoOnly = await fakeEnv({ "family/2026-10-18/photo.jpg": "p", "family/2026-10-18/photo.json": meta });
  assert.equal(await storeFamily(photoOnly, "2026-10-18"), "photo");
  assert.ok(!photoOnly.PHOTOS.m.has("art/used/2026-10-18/family"));

  const nothing = await fakeEnv({ "family/2026-10-18/photo.jpg": "p" });
  assert.equal(await storeFamily(nothing, "2026-10-18"), null, "no photo.json, no caption date");
  assert.equal(nothing.kv.size, 0);
});

const { handleFamilyPage } = fromBuild("src/pages/family.js");
const { describeProblems } = fromBuild("src/alert.js");

function pageKV(entries) {
  const m = new Map(Object.entries(entries));
  return {
    get: async (k) => m.get(k) ?? null,
    put: async (k, v) => { m.set(k, v); },
    list: async ({ prefix }) => ({ keys: [...m.keys()].filter((k) => k.startsWith(prefix)).map((name) => ({ name })), list_complete: true }),
  };
}

test("the page serves today, then the latest earlier day, then the text page", async () => {
  const today = await (await handleFamilyPage({ CACHE: pageKV({ "family:v1:2026-10-18": "VE9EQVk=" }) }, "2026-10-18")).text();
  assert.match(today, /data:image\/png;base64,VE9EQVk=/);

  const stale = await (await handleFamilyPage({ CACHE: pageKV({ "family:v1:2026-10-16": "T0xE" }) }, "2026-10-18")).text();
  assert.match(stale, /data:image\/png;base64,T0xE/, "the mini missed an evening: yesterday's picture stays up");

  const none = await handleFamilyPage({ CACHE: pageKV({}) }, "2026-10-18");
  const html = await none.text();
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /Family Memory/);
  assert.equal(none.headers.get("Cache-Control"), "no-store");
});

test("the alert names a missed family picture without blaming Workers AI", () => {
  const problems = describeProblems({ dateStr: "2026-10-18", missingImages: [], aiBlocked: false, quietMachines: [], jarvisMissed: ["skyline", "family"] });
  assert.ok(problems.some((p) => /2026-10-18's skyline; Workers AI drew it/.test(p)));
  assert.ok(problems.some((p) => /2026-10-18's family picture; the photo itself is shown/.test(p)));
});
