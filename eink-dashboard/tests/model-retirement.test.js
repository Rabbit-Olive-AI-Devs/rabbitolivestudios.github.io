// Offline integration tests. Every image/reference is synthetic; no cloud calls or private files.
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");
const test = require("node:test");
const fromBuild = (p) => require(path.join(process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests", p));
const image = fromBuild("src/image.js");
const color = fromBuild("src/pages/color-moment.js");
const dates = fromBuild("src/date-utils.js");
const keys = fromBuild("src/cache-keys.js");
const portrait = fromBuild("src/birthday-image.js");
const birthdays = fromBuild("src/birthday.js");
const skyline = fromBuild("src/skyline-image.js");
const art = fromBuild("src/art.js");
const worker = fromBuild("src/index.js").default;
const { encodePNGRGB, pngToBase64 } = fromBuild("src/png.js");
const { decodePNG } = fromBuild("src/png-decode.js");
const FLUX = "@cf/black-forest-labs/flux-2-klein-9b";
const DATE = "2026-11-07";
const moment = { year: 1900, location: "Test Harbour", title: "Test Launch", scene: "A vessel", imagePrompt: "A vessel at sea" };
const person = { key: "synthetic-person", name: "Test Person", birthYear: 1990, month: 11, day: 7 };
let syntheticPNG;
test.before(async () => { syntheticPNG = await encodePNGRGB(new Uint8Array(800 * 480 * 3).fill(220), 800, 480); });

function fixture(t, { fail, refs = false, birthday = false, blocked = false } = {}) {
  t.mock.method(console, "log", () => {});
  t.mock.method(console, "error", () => {});
  t.mock.method(dates, "getChicagoDateParts", () => ({ year: "2026", month: "11", day: "07", dateStr: DATE }));
  t.mock.method(birthdays, "getBirthdayToday", () => birthday ? person : null);
  t.mock.method(globalThis, "fetch", async () => { throw new Error("unexpected network access"); });
  const values = new Map([[keys.momentCacheKey(DATE), JSON.stringify({ data: moment, timestamp: Date.now() })]]);
  if (blocked) values.set("ai-budget:v1:block", JSON.stringify({ source: "synthetic-test", blockUntil: Date.now() + 3600000 }));
  const calls = [], transforms = [], writes = [];
  const env = {
    CACHE: {
      async get(k, type) { const v = values.get(k) ?? null; return v && type === "json" ? JSON.parse(v) : v; },
      async put(k, v, opts) { assert.ok(opts.expirationTtl >= 60); writes.push(k); values.set(k, v); },
      async delete(k) { values.delete(k); },
      async list({ prefix }) { return { keys: [...values.keys()].filter(k => k.startsWith(prefix)).map(name => ({ name })), list_complete: true }; },
    },
    PHOTOS: {
      async get(k) { return refs && k.endsWith("_0.jpg") ? { async arrayBuffer() { return new Uint8Array([11, 22, 33]).buffer; } } : null; },
      async head() { return null; },
      async put() {}, async delete() {},
    },
    AI: { async run(model, input) {
      calls.push({ model, input });
      if (fail) throw fail;
      return { image: pngToBase64(syntheticPNG) };
    } },
    IMAGES: { input(stream) {
      assert.ok(stream instanceof ReadableStream, "Images must receive a stream, not raw bytes");
      const pipeline = {
        transform(opts) { transforms.push(opts); return pipeline; },
        async output() { return { response: () => new Response(syntheticPNG) }; },
      };
      return pipeline;
    } },
  };
  const pending = [];
  const ctx = { waitUntil(p) { pending.push(p); } };
  return { env, values, calls, transforms, writes, pending, ctx };
}
async function formOf(call) {
  assert.equal(call.model, FLUX);
  assert.deepEqual(Object.keys(call.input), ["multipart"]);
  return new Response(call.input.multipart.body, { headers: { "content-type": call.input.multipart.contentType } }).formData();
}
function colorEntry(title = "Saved launch") {
  return JSON.stringify({ imageB64: pngToBase64(syntheticPNG), moment: { ...moment, title }, displayDate: "Nov 7" });
}
async function request(f, route) { return worker.fetch(new Request(`https://test.invalid${route}`), f.env, f.ctx); }

test("legacy raw generation uses FLUX multipart, not the retired JSON contract", async t => {
  const f = fixture(t);
  assert.deepEqual(await image.generateAIImage(f.env, "synthetic scene"), syntheticPNG);
  assert.equal(f.calls.length, 1);
  const form = await formOf(f.calls[0]);
  assert.equal(form.get("prompt"), "synthetic scene");
  assert.equal(form.get("width"), "1024");
  assert.equal(form.get("height"), "768");
  assert.equal(form.has("num_steps"), false);
});

for (const [name, generate] of [
  ["legacy colour moment", f => image.generateMomentImage(f.env, moment, "Nov 7", DATE)],
  ["legacy 1-bit moment", f => image.generateMomentImage1Bit(f.env, moment, "Nov 7", DATE)],
  ["colour moment", async f => Uint8Array.from(Buffer.from((await color.generateColorMoment(f.env, moment, DATE)).base64, "base64"))],
]) {
  test(`${name}: one FLUX call and an 800x480 PNG`, async t => {
    const f = fixture(t);
    const png = await generate(f);
    assert.equal(f.calls.length, 1);
    await formOf(f.calls[0]);
    // Pipeline B intentionally emits bit-depth 1; the input decoder only accepts 8-bit.
    assert.equal(Buffer.from(png).readUInt32BE(16), 800);
    assert.equal(Buffer.from(png).readUInt32BE(20), 480);
    assert.equal(png[24], name === "legacy 1-bit moment" ? 1 : 8);
  });
  test(`${name}: failure does not retry the same model or call SDXL`, async t => {
    const error = new Error("synthetic image service down");
    const f = fixture(t, { fail: error });
    await assert.rejects(() => generate(f), /synthetic image service down/);
    assert.equal(f.calls.length, 1);
    assert.equal(f.calls[0].model, FLUX);
  });
}

for (const refs of [false, true]) {
  test(`skyline ${refs ? "with" : "without"} reference: FLUX multipart and correct reference contract`, async t => {
    const f = fixture(t, { refs });
    const result = await skyline.generateSkylineImage(f.env, "reinterpret image 0", "text-only city", { left: "City", center: "Skyline", right: "Nov 7" }, "color", "synthetic-city", 0);
    assert.equal(result.usedRef, refs);
    assert.equal(f.calls.length, 1);
    const form = await formOf(f.calls[0]);
    assert.equal(form.has("input_image_0"), refs);
    assert.equal(form.get("prompt"), refs ? "reinterpret image 0" : "text-only city");
    assert.deepEqual(f.transforms, [{ width: 800, height: 480, fit: "cover" }]);
  });
  test(`birthday ${refs ? "with" : "without"} reference: 1280x768, single FLUX call`, async t => {
    const f = fixture(t, { refs });
    const png = await portrait.generateBirthdayImage(f.env, person, 2026);
    const form = await formOf(f.calls[0]);
    assert.equal(form.get("width"), "1280"); assert.equal(form.get("height"), "768");
    assert.equal(form.has("input_image_0"), refs);
    assert.equal((await decodePNG(png)).width, 800);
    assert.equal(f.calls.length, 1);
  });
}

test("reference skyline failure propagates after one call (no text-only second charge)", async t => {
  const f = fixture(t, { refs: true, fail: new Error("synthetic reference failure") });
  await assert.rejects(() => skyline.generateSkylineImage(f.env, "ref", "text", { left: "", center: "", right: "" }, "bw", "synthetic-city", 0), /synthetic reference failure/);
  assert.equal(f.calls.length, 1);
});

test("colour birthday failure does not generate a text-only stranger or a replacement moment", async t => {
  const f = fixture(t, { birthday: true, refs: true, fail: new Error("synthetic portrait failure") });
  await assert.rejects(() => color.buildColorMomentEntry(f.env, DATE, person, 2026, "Nov 7"), /synthetic portrait failure/);
  assert.equal(f.calls.length, 1);
  assert.equal(f.writes.some(k => k.startsWith("color-birthday:")), false);
});

for (const route of ["/fact.png", "/fact1.png", "/skyline.png"]) {
  test(`${route}: same-date old-version cache survives a failed migration generation`, async t => {
    const f = fixture(t, { fail: new Error("synthetic unavailable") });
    const prefix = route === "/fact.png" ? "fact4" : route === "/fact1.png" ? "fact1" : "skyline";
    f.values.set(`${prefix}:v0:${DATE}${prefix === "skyline" ? ":daily" : ""}`, pngToBase64(syntheticPNG));
    // getTodayEvents itself never uses KV; stub that external boundary, not the image pipeline.
    t.mock.method(fromBuild("src/fact.js"), "getTodayEvents", async () => ({ events: [], displayDate: "Nov 7" }));
    const res = await request(f, route);
    assert.equal(res.status, 200);
    assert.equal(res.headers.get("cache-control"), "no-store");
    assert.deepEqual(new Uint8Array(await res.arrayBuffer()), syntheticPNG);
    await Promise.all(f.pending);
    assert.equal(f.calls.length, 1);
  });
}

test("colour page serves the same-date previous version immediately, even if processing fails", async t => {
  const f = fixture(t);
  f.values.set(`color-moment:v0:${DATE}:old`, colorEntry());
  t.mock.method(fromBuild("src/fact.js"), "getTodayEvents", async () => ({ events: [], displayDate: "Nov 7" }));
  t.mock.method(f.env.IMAGES, "input", () => { throw new Error("synthetic Images failure"); });
  const res = await color.handleColorMomentPage(f.env, new URL("https://test.invalid/color/moment"), f.ctx);
  assert.equal(res.status, 200);
  assert.match(await res.text(), /Saved launch/);
  assert.equal(res.headers.get("cache-control"), "no-store");
  await Promise.all(f.pending);
  assert.equal(f.calls.length, 1);
});

test("birthday page can reuse today's older-version portrait without confusing two birthdays", async t => {
  const f = fixture(t, { birthday: true, blocked: true });
  f.values.set(`color-birthday:v0:${DATE}`, JSON.stringify({ imageB64: pngToBase64(syntheticPNG), moment, displayDate: "Nov 7", birthdayInfo: { name: "Today's Test Person", age: 36, styleName: "Test" } }));
  const res = await color.handleColorMomentPage(f.env, new URL("https://test.invalid/color/moment"), f.ctx);
  assert.equal(res.status, 200); assert.match(await res.text(), /Today&#39;s Test Person/);
  assert.equal(f.calls.length, 0);
});

test("previous person's portrait is never substituted for today's birthday", async t => {
  const f = fixture(t, { birthday: true, blocked: true });
  f.values.set("color-birthday:v0:2026-11-06", JSON.stringify({ imageB64: pngToBase64(syntheticPNG), moment, displayDate: "Nov 6", birthdayInfo: { name: "Wrong Person", age: 10, styleName: "Test" } }));
  f.values.set("color-moment:v0:2026-11-05:old", colorEntry("A prior event"));
  const res = await color.handleColorMomentPage(f.env, new URL("https://test.invalid/color/moment"), f.ctx);
  const html = await res.text();
  assert.equal(res.status, 200); assert.match(html, /A prior event/); assert.doesNotMatch(html, /Wrong Person/);
  assert.equal(f.calls.length, 0);
});

test("cold colour page with a blocked budget returns an 800x480 text page, no broken image", async t => {
  const f = fixture(t, { blocked: true });
  const res = await color.handleColorMomentPage(f.env, new URL("https://test.invalid/color/moment"), f.ctx);
  assert.equal(res.status, 200);
  assert.match(res.headers.get("content-type"), /text\/html/);
  const html = await res.text();
  assert.match(html, /800px.*480px/); assert.doesNotMatch(html, /<img|<script/);
  assert.equal(f.calls.length, 0);
});

test("usage-limit error arms the existing block, serves last good image and prevents another AI call", async t => {
  const f = fixture(t, { fail: new Error("4006 synthetic neurons limit") });
  f.values.set("color-moment:v0:2026-11-06:old", colorEntry());
  t.mock.method(fromBuild("src/fact.js"), "getTodayEvents", async () => ({ events: [], displayDate: "Nov 7" }));
  const first = await color.handleColorMomentPage(f.env, new URL("https://test.invalid/color/moment"), f.ctx);
  assert.equal(first.status, 200);
  await Promise.all(f.pending);
  assert.ok(f.values.has("ai-budget:v1:block"));
  const second = await color.handleColorMomentPage(f.env, new URL("https://test.invalid/color/moment"), f.ctx);
  assert.equal(second.status, 200); assert.equal(f.calls.length, 1);
});

test("missing uploaded pictures return false without spending AI", async t => {
  const f = fixture(t);
  assert.equal(await art.storeJarvisColor(f.env, DATE, null, 2026, "Nov 7", moment), false);
  assert.equal(await art.storeJarvisSkyline(f.env, DATE), false);
  assert.equal(f.calls.length, 0);
});

test("delivered art is stored as Jarvis's picture without an AI call, even with quota blocked", async t => {
  const f = fixture(t, { blocked: true });
  t.mock.method(f.env.PHOTOS, "get", async k => k === `art/${DATE}/moment` ? { async arrayBuffer() { return syntheticPNG.buffer; } } : null);
  assert.equal(await art.storeJarvisColor(f.env, DATE, null, 2026, "Nov 7", moment), true);
  const saved = JSON.parse(f.values.get(keys.colorMomentCacheKey(DATE, color.getColorMomentStyle(DATE).id)));
  assert.equal(saved.source, "jarvis"); assert.equal(f.calls.length, 0);
});

async function warm(t, f) {
  for (const [module, names] of [
    ["weather", ["getWeather", "getWeatherForLocation"]], ["device", ["fetchDeviceData"]],
    ["planet", ["getPlanetOfDay"]], ["flamengo", ["getFlamengo"]], ["family", ["storeFamily"]],
    ["fact", ["getFact"]],
  ]) for (const name of names) t.mock.method(fromBuild(`src/${module}.js`), name, async () => null);
  t.mock.method(fromBuild("src/fact.js"), "getTodayEvents", async () => ({ events: [], displayDate: "Nov 7" }));
  t.mock.method(fromBuild("src/alert.js"), "runAlertCheck", async () => ({ checked: false }));
  t.mock.method(dates, "getChicagoHour", () => 1);
  await worker.scheduled({ cron: "35 4,5 * * *" }, f.env, f.ctx);
  await Promise.all(f.pending);
}

for (const uploads of [false, true]) {
  test(`daily warm: ${uploads ? "delivered art" : "missing artwork"} uses ${uploads ? "zero" : "two"} image calls`, async t => {
    const f = fixture(t);
    if (uploads) t.mock.method(f.env.PHOTOS, "get", async k => [`art/${DATE}/moment`, `art/${DATE}/skyline`].includes(k) ? { async arrayBuffer() { return syntheticPNG.buffer; } } : null);
    await warm(t, f);
    assert.equal(f.calls.length, uploads ? 0 : 2);
    assert.ok(f.values.has(keys.colorMomentCacheKey(DATE, color.getColorMomentStyle(DATE).id)));
    assert.ok(f.values.has(keys.skylineCacheKey(DATE, "daily", 15, 0)));
  });
}

test("daily warm respects an existing budget block while still importing delivered skyline", async t => {
  const f = fixture(t, { blocked: true });
  t.mock.method(f.env.PHOTOS, "get", async k => k === `art/${DATE}/skyline` ? { async arrayBuffer() { return syntheticPNG.buffer; } } : null);
  await warm(t, f);
  assert.equal(f.calls.length, 0);
  assert.ok(f.values.has(keys.skylineCacheKey(DATE, "daily", 15, 0)));
});

test("new quota failure stops image generation but does not strand the already-delivered skyline", async t => {
  const f = fixture(t, { fail: new Error("4006 synthetic neurons limit") });
  t.mock.method(f.env.PHOTOS, "get", async k => k === `art/${DATE}/skyline` ? { async arrayBuffer() { return syntheticPNG.buffer; } } : null);
  await warm(t, f);
  assert.equal(f.calls.length, 1);
  assert.ok(f.values.has("ai-budget:v1:block"));
  assert.ok(f.values.has(keys.skylineCacheKey(DATE, "daily", 15, 0)));
});

test("mono birthday failure falls back to a saved event, not a second image attempt", async t => {
  const f = fixture(t, { birthday: true, fail: new Error("synthetic portrait failure") });
  f.values.set(`fact4:v0:${DATE}`, pngToBase64(syntheticPNG));
  t.mock.method(fromBuild("src/fact.js"), "getTodayEvents", async () => ({ events: [], displayDate: "Nov 7" }));
  const res = await request(f, "/fact.png");
  assert.equal(res.status, 200);
  await Promise.all(f.pending);
  assert.equal(f.calls.length, 1);
});

for (const state of ["cold-budget-block", "last-good-event", "same-day-birthday", "birthday-event-fallback"]) {
  test(`panel HTML: ${state} stays self-contained at 800x480`, async t => {
    const birthday = state.includes("birthday");
    const f = fixture(t, { birthday, blocked: true });
    if (state === "same-day-birthday") {
      f.values.set(`color-birthday:v0:${DATE}`, JSON.stringify({ ...JSON.parse(colorEntry("Test Person")), birthdayInfo: { name: "Test Person", age: 36, styleName: "Test style" } }));
    } else if (state !== "cold-budget-block") f.values.set(`color-moment:v0:${DATE}:synthetic`, colorEntry());
    const res = await request(f, "/color/moment");
    const html = await res.text();
    assert.equal(res.status, 200);
    assert.match(res.headers.get("content-type"), /text\/html/);
    assert.doesNotMatch(html, /<script|src=["'](?:https?:|\/)/i);
    assert.equal(f.calls.length, 0);
    if (state === "cold-budget-block") assert.doesNotMatch(html, /<img/);
    else assert.match(html, /data:image\/png;base64/);
    if (state === "same-day-birthday") assert.match(html, /Happy Birthday!.*Test Person - 36 years.*Test style/);
    if (state === "birthday-event-fallback") assert.doesNotMatch(html, /Happy Birthday!/);
    if (process.env.EINK_UI_EVIDENCE_DIR) fs.writeFileSync(path.join(process.env.EINK_UI_EVIDENCE_DIR, `${state}.html`), html, { flag: "wx", mode: 0o600 });
  });
}

test("executable source has no retired SDXL model identifier", () => {
  const root = path.resolve(__dirname, "../src");
  for (const p of fs.readdirSync(root, { recursive: true })) {
    if (String(p).endsWith(".ts") && !String(p).endsWith(".d.ts")) {
      assert.doesNotMatch(fs.readFileSync(path.join(root, String(p)), "utf8"), /@cf\/stabilityai\/stable-diffusion-xl-base-1\.0/);
    }
  }
});
