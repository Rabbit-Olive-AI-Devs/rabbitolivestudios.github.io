const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { serveStaleWhileGenerating } = fromBuild("src/cache-guard.js");

/** ExecutionContext stub that records what was handed to waitUntil. */
function makeCtx() {
  const pending = [];
  return { pending, waitUntil(p) { pending.push(p); } };
}

const stalePNG = () =>
  new Response(new Uint8Array([1, 2, 3]), {
    headers: { "Content-Type": "image/png", "Cache-Control": "public, max-age=86400" },
  });

// --- serveStaleWhileGenerating (DECISIONS #64) ---

test("serves the stale image immediately and runs generation in the background", async () => {
  const ctx = makeCtx();
  let generated = false;
  let release;
  const gate = new Promise((r) => { release = r; });

  const res = await serveStaleWhileGenerating(ctx, "t", stalePNG, async () => {
    await gate;
    generated = true;
  });

  // Returned before generation finished, with the stale bytes.
  assert.ok(res);
  assert.equal(generated, false);
  assert.deepEqual([...new Uint8Array(await res.arrayBuffer())], [1, 2, 3]);
  assert.equal(res.headers.get("Content-Type"), "image/png");

  // Generation was handed to waitUntil and completes there.
  assert.equal(ctx.pending.length, 1);
  release();
  await ctx.pending[0];
  assert.equal(generated, true);
});

test("stale response is marked no-store so it never outlives the cold window downstream", async () => {
  const res = await serveStaleWhileGenerating(makeCtx(), "t", stalePNG, async () => {});
  assert.equal(res.headers.get("Cache-Control"), "no-store");
  assert.equal(res.headers.get("X-Stale-While-Generating"), "1");
});

test("returns null when nothing stale exists, without starting generation", async () => {
  const ctx = makeCtx();
  let generated = false;
  const res = await serveStaleWhileGenerating(ctx, "t", async () => null, async () => { generated = true; });
  assert.equal(res, null);
  assert.equal(ctx.pending.length, 0);
  assert.equal(generated, false);
});

test("returns null without an ExecutionContext (cron callers block as before)", async () => {
  let looked = false;
  const res = await serveStaleWhileGenerating(undefined, "t", async () => { looked = true; return stalePNG(); }, async () => {});
  assert.equal(res, null);
  assert.equal(looked, false);
});

test("a failing background generation is contained — waitUntil promise resolves", async () => {
  const ctx = makeCtx();
  const res = await serveStaleWhileGenerating(ctx, "t", stalePNG, async () => { throw new Error("FLUX down"); });
  assert.ok(res);
  await assert.doesNotReject(ctx.pending[0]);
});
