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

// --- getTodayEvents / getFact accept an explicit date (DECISIONS #64) ---

test("getTodayEvents fetches the requested date, not today's", async () => {
  const { getTodayEvents } = fromBuild("src/fact.js");
  const realFetch = globalThis.fetch;
  let url = "";
  globalThis.fetch = async (u) => { url = String(u); return new Response(JSON.stringify({ events: [{ year: 1969, text: "Moon" }] })); };
  try {
    const r = await getTodayEvents({}, "2026-12-25");
    assert.match(url, /onthisday\/events\/12\/25$/);
    assert.equal(r.dateStr, "2026-12-25");
    assert.equal(r.displayDate, "Dec 25");
    assert.deepEqual(r.events, [{ year: 1969, text: "Moon" }]);
  } finally {
    globalThis.fetch = realFetch;
  }
});

// --- Colour skyline caption survives dithering (DECISIONS #64) ---

test("caption rows stay solid black/white while the picture is dithered", () => {
  const { ditherPictureKeepCaption } = fromBuild("src/skyline-image.js");
  const { ditherFloydSteinberg } = fromBuild("src/dither-spectra6.js");
  const { SPECTRA6_PALETTE } = fromBuild("src/spectra6.js");
  const W = 800, H = 480, BAR = 24;

  // Bright, slightly off-palette "sky" (this is what dumps error into the bar), then a
  // black bar with a white text block in it.
  const rgb = new Uint8Array(W * H * 3);
  for (let i = 0; i < (H - BAR) * W; i++) { rgb[i * 3] = 250; rgb[i * 3 + 1] = 235; rgb[i * 3 + 2] = 120; }
  for (let y = H - BAR; y < H; y++) for (let x = 0; x < W; x++) {
    const white = y >= H - 16 && y < H - 8 && x >= 8 && x < 40;
    const o = (y * W + x) * 3; rgb[o] = rgb[o + 1] = rgb[o + 2] = white ? 255 : 0;
  }

  const fixed = ditherPictureKeepCaption(rgb);
  const naive = ditherFloydSteinberg(rgb, W, H, SPECTRA6_PALETTE);

  let fixedBad = 0, naiveBad = 0, whiteOk = 0;
  for (let y = H - BAR; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    const expect = (y >= H - 16 && y < H - 8 && x >= 8 && x < 40) ? 1 : 0;
    if (fixed[i] !== expect) fixedBad++;
    if (naive[i] !== expect) naiveBad++;
    if (expect === 1 && fixed[i] === 1) whiteOk++;
  }
  assert.equal(fixedBad, 0, "caption rows must map exactly to black/white");
  assert.equal(whiteOk, 8 * 32);
  assert.ok(naiveBad > 1000, `plain FS should corrupt the bar (got ${naiveBad}) — otherwise this test proves nothing`);
  // Picture rows still dithered (not a single flat colour).
  const pic = new Set(fixed.subarray(0, (H - BAR) * W));
  assert.ok(pic.size >= 2);
});
