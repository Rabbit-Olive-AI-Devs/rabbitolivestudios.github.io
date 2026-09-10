const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const {
  renderSkylineInlineHTML,
  skylinePageResponse,
  skylineBwPageResponse,
} = fromBuild("src/pages/skyline.js");

/** KV stub: empty store, so the AI budget is not blocked and nothing is cached. */
function makeEnv() {
  return {
    CACHE: {
      async get() { return null; },
      async list() { return { keys: [], list_complete: true }; },
      async put() {},
    },
  };
}

// PNG signature bytes; base64 of these four is "iVBORw==".
const PNG_BYTES = new Uint8Array([137, 80, 78, 71]);
const PNG_B64 = "iVBORw==";

const pngResponse = () =>
  new Response(PNG_BYTES, {
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "no-store",
      "X-Skyline-City": "havana",
      "X-Stale-While-Generating": "1",
    },
  });

// --- DECISIONS #65: the panel must need exactly one request -----------------

test("renderSkylineInlineHTML embeds the PNG as a data URI and references no image route", () => {
  const html = renderSkylineInlineHTML(PNG_B64, true);
  assert.match(html, /<img src="data:image\/png;base64,iVBORw=="/);
  assert.doesNotMatch(html, /skyline\.png/);
  assert.doesNotMatch(html, /<script/);
});

test("/skyline inlines the image the png handler returns", async () => {
  let calls = 0;
  const res = await skylinePageResponse(makeEnv(), async () => { calls++; return pngResponse(); });

  assert.equal(calls, 1);
  assert.equal(res.status, 200);
  assert.match(res.headers.get("Content-Type"), /text\/html/);
  assert.equal(res.headers.get("Cache-Control"), "no-store");
  const html = await res.text();
  assert.match(html, /data:image\/png;base64,iVBORw==/);
  assert.doesNotMatch(html, /src="\/skyline\.png/);
});

test("/skyline forwards the png handler's diagnostic headers", async () => {
  const res = await skylinePageResponse(makeEnv(), pngResponse);
  assert.equal(res.headers.get("X-Skyline-City"), "havana");
  assert.equal(res.headers.get("X-Stale-While-Generating"), "1");
});

test("/skyline serves the text page when the png handler cannot produce an image", async () => {
  const res = await skylinePageResponse(makeEnv(), async () =>
    new Response("Failed to generate skyline image", { status: 503 }));

  assert.equal(res.status, 200);
  const html = await res.text();
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /could not be generated/);
});

test("/skyline-bw inlines too, without Spectra 6 CSS", async () => {
  const res = await skylineBwPageResponse(makeEnv(), pngResponse);
  const html = await res.text();
  assert.match(html, /data:image\/png;base64,iVBORw==/);
  assert.doesNotMatch(html, /--s6-red/);
});
