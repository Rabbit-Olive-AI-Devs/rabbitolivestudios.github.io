const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const {
  findMostRecentCached,
  hasRecentCached,
  imageUnavailable,
  STALE_INDEX_PREFIX,
} = fromBuild("src/stale-cache.js");

/**
 * Minimal KV stub that counts operations. The whole point of the index is the
 * op count, so the counters are the assertion (DECISIONS #62).
 */
function makeKV(initial = {}) {
  const store = new Map(Object.entries(initial));
  const ops = { list: 0, get: 0, put: 0 };
  return {
    ops,
    store,
    async list({ prefix, cursor }) {
      ops.list++;
      const keys = [...store.keys()]
        .filter((k) => k.startsWith(prefix))
        .map((name) => ({ name }));
      return { keys, list_complete: true, cursor: undefined };
    },
    async get(key) {
      ops.get++;
      return store.has(key) ? store.get(key) : null;
    },
    async put(key, value) {
      ops.put++;
      store.set(key, value);
    },
    async delete(key) {
      store.delete(key);
    },
  };
}

const DAY = "2026-08-21";
const images = () => ({
  "fact4:v4:2026-08-18": "old",
  "fact4:v4:2026-08-19": "newest",
});

// --- the fix: one list, then none -------------------------------------------

test("findMostRecentCached lists once, then serves from the index", async () => {
  const CACHE = makeKV(images());
  const env = { CACHE };

  const a = await findMostRecentCached(env, "fact4:", DAY);
  assert.equal(a.key, "fact4:v4:2026-08-19");
  assert.equal(CACHE.ops.list, 1);

  // Simulating the next device poll during the same outage.
  for (let i = 0; i < 20; i++) {
    const hit = await findMostRecentCached(env, "fact4:", DAY);
    assert.equal(hit.key, "fact4:v4:2026-08-19");
  }
  assert.equal(CACHE.ops.list, 1, "21 lookups must still cost exactly one list");
});

test("the index is written once, not per request", async () => {
  const CACHE = makeKV(images());
  const env = { CACHE };
  for (let i = 0; i < 10; i++) await findMostRecentCached(env, "fact4:", DAY);
  const idxWrites = CACHE.ops.put;
  assert.equal(idxWrites, 1, "writes are the tighter budget — index must not rewrite per request");
  assert.ok(CACHE.store.has(STALE_INDEX_PREFIX + "fact4:"));
});

test("a cold index lists again", async () => {
  const CACHE = makeKV(images());
  const env = { CACHE };
  await findMostRecentCached(env, "fact4:", DAY);
  assert.equal(CACHE.ops.list, 1);
  CACHE.store.delete(STALE_INDEX_PREFIX + "fact4:"); // TTL expiry
  await findMostRecentCached(env, "fact4:", DAY);
  assert.equal(CACHE.ops.list, 2);
});

// --- correctness must survive a stale index ---------------------------------

test("a stale index naming a dead key still resolves the live entry", async () => {
  const CACHE = makeKV(images());
  const env = { CACHE };
  await findMostRecentCached(env, "fact4:", DAY); // index now names 08-19 and 08-18

  // The newest entry expires while the index still lists it.
  CACHE.store.delete("fact4:v4:2026-08-19");

  const hit = await findMostRecentCached(env, "fact4:", DAY);
  assert.ok(hit, "must not return null just because the index was stale");
  assert.equal(hit.key, "fact4:v4:2026-08-18");
});

test("a wholly stale index re-lists rather than reporting nothing", async () => {
  const CACHE = makeKV(images());
  const env = { CACHE };
  await findMostRecentCached(env, "fact4:", DAY);

  // Everything the index named is gone, but a newer entry has appeared.
  CACHE.store.delete("fact4:v4:2026-08-19");
  CACHE.store.delete("fact4:v4:2026-08-18");
  CACHE.store.set("fact4:v5:2026-08-20", "fresh");

  const hit = await findMostRecentCached(env, "fact4:", DAY);
  assert.ok(hit, "a fully stale index must trigger a re-list");
  assert.equal(hit.key, "fact4:v5:2026-08-20");
});

test("findMostRecentCached returns null when there is genuinely nothing", async () => {
  const CACHE = makeKV({});
  const env = { CACHE };
  assert.equal(await findMostRecentCached(env, "skyline:", DAY), null);
});

// --- hasRecentCached must verify, not trust the index ------------------------

test("hasRecentCached confirms the entry actually exists", async () => {
  const CACHE = makeKV(images());
  const env = { CACHE };
  assert.equal(await hasRecentCached(env, "fact4:", DAY), true);

  // Index still names both keys, but both are gone. Trusting the index here
  // would render an <img> that 404s — the exact glyph #58 exists to prevent.
  CACHE.store.delete("fact4:v4:2026-08-19");
  CACHE.store.delete("fact4:v4:2026-08-18");
  assert.equal(await hasRecentCached(env, "fact4:", DAY), false);
});

test("hasRecentCached ignores entries outside the lookback window", async () => {
  const CACHE = makeKV({ "fact4:v4:2026-01-01": "ancient" });
  const env = { CACHE };
  assert.equal(await hasRecentCached(env, "fact4:", DAY), false);
});

test("hasRecentCached honours the accept filter (bw vs colour skyline)", async () => {
  const CACHE = makeKV({
    "skyline:v3:2026-08-20:daily": "colour",
    "skyline:v3:2026-08-20:daily:bw": "bw",
  });
  const env = { CACHE };
  const bwOnly = { accept: (k) => k.endsWith(":bw") };
  const colourOnly = { accept: (k) => !k.endsWith(":bw") };
  assert.equal(await hasRecentCached(env, "skyline:", DAY, bwOnly), true);
  assert.equal(await hasRecentCached(env, "skyline:", DAY, colourOnly), true);
  CACHE.store.delete("skyline:v3:2026-08-20:daily:bw");
  assert.equal(await hasRecentCached(env, "skyline:", DAY, bwOnly), false);
  assert.equal(await hasRecentCached(env, "skyline:", DAY, colourOnly), true);
});

// --- the index must never be the reason a lookup fails -----------------------

test("a failing index write does not break the lookup", async () => {
  const CACHE = makeKV(images());
  CACHE.put = async () => { throw new Error("KV write quota exceeded"); };
  const env = { CACHE };
  const hit = await findMostRecentCached(env, "fact4:", DAY);
  assert.equal(hit.key, "fact4:v4:2026-08-19");
});

test("a corrupt index falls back to a real list", async () => {
  const CACHE = makeKV(images());
  CACHE.store.set(STALE_INDEX_PREFIX + "fact4:", "{not json");
  const env = { CACHE };
  const hit = await findMostRecentCached(env, "fact4:", DAY);
  assert.equal(hit.key, "fact4:v4:2026-08-19");
});

test("the index key is not itself picked up as a cache entry", async () => {
  const CACHE = makeKV(images());
  const env = { CACHE };
  await findMostRecentCached(env, "fact4:", DAY);
  const hit = await findMostRecentCached(env, "fact4:", DAY);
  assert.ok(!hit.key.startsWith(STALE_INDEX_PREFIX));
});

// --- imageUnavailable still short-circuits on a healthy day ------------------

test("imageUnavailable costs zero list ops when AI is not blocked", async () => {
  const CACHE = makeKV(images());
  const env = { CACHE };
  assert.equal(await imageUnavailable(env, "fact4:", DAY), false);
  assert.equal(CACHE.ops.list, 0, "a healthy day must not list at all");
});

test("imageUnavailable reports true only when blocked AND nothing cached", async () => {
  const blocked = JSON.stringify({
    source: "test", message: "4006", createdAt: 0, blockUntil: Date.now() + 3_600_000,
  });
  const withImages = makeKV({ ...images(), "ai-budget:v1:block": blocked });
  assert.equal(await imageUnavailable({ CACHE: withImages }, "fact4:", DAY), false);

  const without = makeKV({ "ai-budget:v1:block": blocked });
  assert.equal(await imageUnavailable({ CACHE: without }, "fact4:", DAY), true);
});

test("an empty family is memoised, not re-listed on every request", async () => {
  // `birthday:` is empty on all but a handful of days, and /fact checks it on
  // every request. Re-listing an empty index would defeat the memo entirely.
  const CACHE = makeKV(images());
  const env = { CACHE };
  for (let i = 0; i < 20; i++) {
    assert.equal(await findMostRecentCached(env, "birthday:", DAY), null);
  }
  assert.equal(CACHE.ops.list, 1, "20 lookups of an empty family must cost one list");
});

test("an index that named only dead entries still re-lists", async () => {
  // The other half of the rule: a demonstrably wrong index must not be trusted.
  const CACHE = makeKV(images());
  const env = { CACHE };
  await findMostRecentCached(env, "fact4:", DAY);
  const listsAfterWarm = CACHE.ops.list;
  CACHE.store.delete("fact4:v4:2026-08-19");
  CACHE.store.delete("fact4:v4:2026-08-18");
  CACHE.store.set("fact4:v5:2026-08-20", "fresh");
  const hit = await findMostRecentCached(env, "fact4:", DAY);
  assert.equal(hit.key, "fact4:v5:2026-08-20");
  assert.ok(CACHE.ops.list > listsAfterWarm, "must have paid for a re-list");
});
