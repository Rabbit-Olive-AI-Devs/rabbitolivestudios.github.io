const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const { storeFamily, familyStyleFor, familyCaption, yearsAgo } = fromBuild("src/family.js");
const { encodePNGRGB } = fromBuild("src/png.js");
const { familyCacheKey } = fromBuild("src/cache-keys.js");
const {
  sha256Hex, stagedPickOrder, claimsFor, stagedManifestKey, stagedPhotoKey, stagedRepaintKey,
} = fromBuild("src/family-staged.js");
const {
  isFamilyPreflightInstant, periodicFireAction, chicagoHourMinute, runFamilyPreflight, familyPreflightMessage,
  familyPreflightSubject, sendFamilyPreflightTest,
} = fromBuild("src/family-preflight.js");
const { describeProblems, alertFingerprint } = fromBuild("src/alert.js");

const D = "2026-11-22";
const HEALTH = "https://eink-dashboard.thiago-oliveira77.workers.dev";

function record(date, stagedAt, extra = {}) {
  return {
    key: stagedManifestKey(date),
    raw: {},
    httpEtag: `W/"${date}"`,
    manifest: {
      schema: 1, date, taken: "2019-05-04", place: "Sample Shore", stagedAt,
      photoSha256: "ab".repeat(32), repaintSha256: null,
      used: false, usedFor: null, usedAt: null, claimFor: null,
      ...extra,
    },
  };
}

test("staged pick order prefers the display date, then the newest stagedAt, then the later date", () => {
  const exact = record(D, "2026-11-01T00:00:00.000Z");
  const older = record("2026-11-02", "2026-11-18T00:00:00.000Z");
  const newer = record("2026-11-03", "2026-11-20T00:00:00.000Z");
  const tieEarly = record("2026-10-01", "2026-11-19T00:00:00.000Z");
  const tieLate = record("2026-10-02", "2026-11-19T00:00:00.000Z");
  const used = record("2026-12-01", "2026-11-21T00:00:00.000Z", { used: true, usedFor: "2026-11-01", usedAt: "2026-11-01T00:00:00.000Z" });
  const claimed = record("2026-12-02", "2026-11-21T01:00:00.000Z", { claimFor: "2026-11-09" });
  const order = stagedPickOrder([newer, used, tieLate, claimed, older, exact, tieEarly], D);
  assert.deepEqual(order.map((r) => r.manifest.date), [D, "2026-11-03", "2026-10-02", "2026-10-01", "2026-11-02"]);
  assert.deepEqual(claimsFor([claimed, exact], "2026-11-09").map((r) => r.manifest.date), ["2026-12-02"]);
  assert.equal(claimsFor([exact], D).length, 0);
});

function bucket() {
  let n = 1;
  const m = new Map();
  const api = {
    m,
    rewrite: null,
    seed(k, body) {
      m.set(k, { body, etag: `W/"${n++}"` });
    },
    body(k) { return m.get(k)?.body; },
    etag(k) { return m.get(k)?.etag; },
    async get(k) {
      const cur = m.get(k);
      if (!cur) return null;
      const body = cur.body;
      return {
        httpEtag: cur.etag,
        etag: cur.etag,
        json: async () => JSON.parse(typeof body === "string" ? body : new TextDecoder().decode(body)),
        arrayBuffer: async () => {
          if (typeof body === "string") return new TextEncoder().encode(body).buffer;
          return body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength);
        },
      };
    },
    async head(k) { return m.has(k) ? { httpEtag: m.get(k).etag } : null; },
    async put(k, v, opts) {
      const match = opts?.onlyIf?.etagMatches;
      if (api.rewrite && match && match === api.rewrite.etag && k === api.rewrite.key) {
        m.set(k, { body: api.rewrite.body, etag: `W/"${n++}"` });
        api.rewrite = null;
        return null;
      }
      if (match) {
        const cur = m.get(k);
        if (!cur || cur.etag !== match) return null;
      }
      const etag = `W/"${n++}"`;
      m.set(k, { body: v, etag });
      return { httpEtag: etag, etag };
    },
    async delete(k) { m.delete(k); },
    async list({ prefix }) {
      return { objects: [...m.keys()].filter((key) => key.startsWith(prefix)).map((key) => ({ key })), truncated: false };
    },
  };
  return api;
}

async function picture(date, opts = {}) {
  const photo = new TextEncoder().encode(opts.photo ?? `photo-${date}`);
  const repaint = opts.repaint == null ? null : new TextEncoder().encode(opts.repaint);
  return {
    photo,
    repaint,
    json: {
      schema: 1,
      date,
      taken: opts.taken ?? "2019-05-04",
      place: opts.place ?? "Sample Shore",
      stagedAt: opts.stagedAt ?? "2026-11-20T18:00:00.000Z",
      photoSha256: await sha256Hex(photo),
      repaintSha256: repaint ? await sha256Hex(repaint) : null,
      used: opts.used ?? false,
      usedFor: opts.usedFor ?? null,
      usedAt: opts.usedAt ?? null,
      claimFor: opts.claimFor ?? null,
    },
  };
}

function plant(photos, date, built) {
  photos.seed(stagedPhotoKey(date), built.photo);
  if (built.repaint) photos.seed(stagedRepaintKey(date), built.repaint);
  photos.seed(stagedManifestKey(date), JSON.stringify(built.json));
}

async function envWith(photos) {
  const png = await encodePNGRGB(new Uint8Array(800 * 480 * 3).fill(180), 800, 480);
  const kv = new Map();
  let ai = 0;
  return {
    kv, ai: () => ai, photos,
    PHOTOS: photos,
    CACHE: {
      get: async (k) => kv.get(k) ?? null,
      put: async (k, v, opts) => {
        if (String(k).startsWith("family:v1:")) assert.equal(opts.expirationTtl, 604800);
        kv.set(k, v);
      },
    },
    AI: { run: async () => { ai += 1; throw new Error("no AI"); } },
    IMAGES: { input: () => ({ transform: () => ({ output: async () => ({ response: () => new Response(png) }) }) }) },
  };
}

test("the warm uses the display date's staged repaint and never deletes or reuses it", async () => {
  const photos = bucket();
  const exact = await picture(D, { repaint: "repaint-bytes", stagedAt: "2026-11-01T00:00:00.000Z", taken: "2019-05-04" });
  const newer = await picture("2026-11-03", { stagedAt: "2026-11-21T00:00:00.000Z", photo: "other-photo" });
  plant(photos, D, exact);
  plant(photos, "2026-11-03", newer);
  const env = await envWith(photos);

  assert.equal(await storeFamily(env, D), "staged");
  assert.equal(env.ai(), 0);
  const saved = JSON.parse(env.kv.get(familyCacheKey(D)));
  assert.equal(saved.staged.entry, D);
  assert.equal(saved.staged.asset, "repaint");
  assert.equal(saved.date, "2019-05-04");
  assert.equal(saved.yearsAgo, yearsAgo("2019-05-04", D));
  assert.equal(saved.place, "Sample Shore");
  assert.equal(saved.style, familyStyleFor(D).id);
  assert.deepEqual(familyCaption("2019-05-04", D), { left: "4 May 2019", center: "Family Memory", right: `${saved.yearsAgo} years ago` });
  assert.ok(!photos.m.has(`art/used/${D}/family`));
  for (const key of [stagedManifestKey(D), stagedPhotoKey(D), stagedRepaintKey(D), stagedManifestKey("2026-11-03"), stagedPhotoKey("2026-11-03")]) {
    assert.ok(photos.m.has(key), key);
  }
  const used = JSON.parse(photos.body(stagedManifestKey(D)));
  assert.equal(used.used, true);
  assert.equal(used.usedFor, D);
  assert.equal(used.claimFor, D);
  assert.ok(used.usedAt);

  env.kv.delete(familyCacheKey(D));
  assert.equal(await storeFamily(env, D), "staged-photo");
  assert.equal(JSON.parse(env.kv.get(familyCacheKey(D))).staged.entry, "2026-11-03");
  assert.equal(JSON.parse(photos.body(stagedManifestKey(D))).used, true, "a used entry stays used");
  assert.equal(JSON.parse(photos.body(stagedManifestKey("2026-11-03"))).used, true);
});

test("a late repaint written before the claim is the asset that is stored", async () => {
  const photos = bucket();
  const first = await picture(D, { photo: "same-photo" });
  plant(photos, D, first);
  const next = await picture(D, { photo: "same-photo", repaint: "late-repaint" });
  photos.seed(stagedRepaintKey(D), next.repaint);
  photos.rewrite = { key: stagedManifestKey(D), etag: photos.etag(stagedManifestKey(D)), body: JSON.stringify(next.json) };
  const env = await envWith(photos);
  assert.equal(await storeFamily(env, D), "staged");
  const manifest = JSON.parse(photos.body(stagedManifestKey(D)));
  assert.equal(manifest.repaintSha256, next.json.repaintSha256);
  assert.equal(manifest.used, true);
  assert.ok(photos.m.has(stagedPhotoKey(D)));
  assert.ok(photos.m.has(stagedRepaintKey(D)));
});

test("a hash miss is skipped, a used or claimed entry is not taken, and the ordinary upload still wins", async () => {
  const photos = bucket();
  const bad = await picture("2026-11-20", { stagedAt: "2026-11-21T03:00:00.000Z" });
  bad.json.photoSha256 = "0".repeat(64);
  const used = await picture("2026-11-18", { stagedAt: "2026-11-21T04:00:00.000Z", used: true, usedFor: "2026-11-01", usedAt: "2026-11-01T00:00:00.000Z" });
  const claimed = await picture("2026-11-19", { stagedAt: "2026-11-21T05:00:00.000Z", claimFor: "2026-11-21" });
  const good = await picture("2026-11-04", { stagedAt: "2026-11-19T00:00:00.000Z", photo: "good-spare" });
  plant(photos, "2026-11-20", bad);
  plant(photos, "2026-11-18", used);
  plant(photos, "2026-11-19", claimed);
  plant(photos, "2026-11-04", good);
  photos.seed(`family/${D}/photo.json`, JSON.stringify({ taken: "nope" }));
  const badMeta = await envWith(photos);
  assert.equal(await storeFamily(badMeta, D), null, "a bad dated manifest fails closed");
  assert.equal(JSON.parse(photos.body(stagedManifestKey("2026-11-04"))).claimFor, null);

  photos.delete(`family/${D}/photo.json`);
  const env = await envWith(photos);
  assert.equal(await storeFamily(env, D), "staged-photo");
  assert.equal(JSON.parse(env.kv.get(familyCacheKey(D))).staged.entry, "2026-11-04");
  assert.equal(JSON.parse(photos.body(stagedManifestKey("2026-11-20"))).used, false);
  assert.equal(JSON.parse(photos.body(stagedManifestKey("2026-11-18"))).used, true);
  assert.equal(JSON.parse(photos.body(stagedManifestKey("2026-11-19"))).claimFor, "2026-11-21");

  const dated = bucket();
  const spare = await picture(D, { repaint: "should-not-win" });
  plant(dated, D, spare);
  dated.seed(`family/${D}/photo.json`, JSON.stringify({ taken: "2018-01-02", place: "Sample Shore" }));
  dated.seed(`family/${D}/photo.jpg`, new TextEncoder().encode("dated-photo"));
  const datedEnv = await envWith(dated);
  assert.equal(await storeFamily(datedEnv, D), "photo");
  assert.equal(JSON.parse(dated.body(stagedManifestKey(D))).used, false);
  assert.equal(JSON.parse(dated.body(stagedManifestKey(D))).claimFor, null);
  assert.ok(!dated.m.has(`family/${D}/photo.json`));
  assert.ok(dated.m.has(stagedPhotoKey(D)));
});

test("an interrupted claim is finished, and a staged cache does not adopt a later painting", async () => {
  const photos = bucket();
  const owned = await picture(D, { photo: "owned-photo", claimFor: D, stagedAt: "2026-11-01T00:00:00.000Z" });
  const other = await picture("2026-11-09", { photo: "newer-spare", stagedAt: "2026-11-21T00:00:00.000Z" });
  plant(photos, D, owned);
  plant(photos, "2026-11-09", other);
  const env = await envWith(photos);
  assert.equal(await storeFamily(env, D), "staged-photo");
  assert.equal(JSON.parse(env.kv.get(familyCacheKey(D))).staged.entry, D);
  assert.equal(JSON.parse(photos.body(stagedManifestKey(D))).used, true);
  assert.equal(JSON.parse(photos.body(stagedManifestKey("2026-11-09"))).used, false);

  const late = bucket();
  const cached = JSON.stringify({
    image: "abc", date: "2019-05-04", yearsAgo: 7, place: "Sample Shore", style: "not-the-schedule",
    staged: { entry: D, asset: "photo" },
  });
  plant(late, D, await picture(D, { claimFor: D }));
  late.seed(`art/${D}/family`, new TextEncoder().encode("late-painting"));
  const lateEnv = await envWith(late);
  lateEnv.kv.set(familyCacheKey(D), cached);
  assert.equal(await storeFamily(lateEnv, D), "staged-photo");
  assert.equal(lateEnv.kv.get(familyCacheKey(D)), cached);
  assert.ok(late.m.has(`art/${D}/family`));
  assert.ok(!late.m.has(`art/used/${D}/family`));
  assert.equal(JSON.parse(late.body(stagedManifestKey(D))).used, true);
});

test("21:05 America/Chicago is the only new periodic fire, in both offsets", () => {
  const cases = [
    ["2026-07-16T02:05:00Z", "21:05", "preflight"],
    ["2026-07-16T03:05:00Z", "22:05", "noop"],
    ["2026-01-16T03:05:00Z", "21:05", "preflight"],
    ["2026-01-16T02:05:00Z", "20:05", "noop"],
    ["2026-03-09T02:05:00Z", "21:05", "preflight"],
    ["2026-03-09T03:05:00Z", "22:05", "noop"],
    ["2026-11-02T03:05:00Z", "21:05", "preflight"],
    ["2026-11-02T02:05:00Z", "20:05", "noop"],
    ["2026-11-01T02:05:00Z", "21:05", "preflight"],
    ["2026-07-16T00:05:00Z", "19:05", "refresh"],
    ["2026-01-16T18:05:00Z", "12:05", "refresh"],
  ];
  for (const [iso, wall, action] of cases) {
    const ms = Date.parse(iso);
    const got = chicagoHourMinute(ms);
    assert.equal(`${String(got.hour).padStart(2, "0")}:${String(got.minute).padStart(2, "0")}`, wall, iso);
    assert.equal(isFamilyPreflightInstant(ms), wall === "21:05", iso);
    assert.equal(periodicFireAction(ms), action, iso);
  }
});

test("the periodic cron stays one expression and adds only the two preflight hours", () => {
  const toml = fs.readFileSync(path.join(__dirname, "../wrangler.toml"), "utf8");
  const line = toml.match(/crons = \[([^\]]+)\]/);
  assert.ok(line);
  assert.equal(line[1], '"35 4,5 * * *", "5 0,2,3,6,12,18 * * *"');
});

function preflightEnv(photos, opts = {}) {
  const kv = new Map();
  const sent = [];
  return {
    kv, sent,
    PHOTOS: photos,
    CACHE: {
      get: async (k) => kv.get(k) ?? null,
      put: async (k, v, optsPut) => {
        assert.equal(optsPut.expirationTtl, 604800);
        kv.set(k, v);
      },
    },
    SEND_EMAIL: opts.email ? { send: async (msg) => { sent.push(msg); } } : undefined,
    ALERT_TO: opts.email ? "alert-to" : undefined,
    ALERT_FROM: opts.email ? "alert-from" : undefined,
  };
}

test("the subject names the Chicago date the way the frame mail should", () => {
  assert.equal(familyPreflightSubject("2026-11-22"), "Frame: Nov 22 family photo missing at 9:05 PM");
  assert.equal(familyPreflightSubject("2026-01-05"), "Frame: Jan 5 family photo missing at 9:05 PM");
});

test("the preflight emails once, skips when mail is unset, and a spare pool does not count as ready", async () => {
  const photos = bucket();
  const spare = await picture("2026-11-28", { stagedAt: "2026-11-20T00:00:00.000Z" });
  plant(photos, "2026-11-28", spare);
  const when = Date.parse("2026-01-16T03:05:00Z");
  const quiet = preflightEnv(photos);
  assert.deepEqual(await runFamilyPreflight(quiet, when, HEALTH), { posted: false, skipped: "not-configured" });
  assert.equal(quiet.kv.size, 0);

  const env = preflightEnv(photos, { email: true });
  const first = await runFamilyPreflight(env, when, HEALTH);
  const second = await runFamilyPreflight(env, when, HEALTH);
  assert.deepEqual(first, { posted: true, skipped: "email" });
  assert.deepEqual(second, { posted: false, skipped: "deduped" });
  assert.equal(env.sent.length, 1);
  assert.equal(env.sent[0].from, "alert-from");
  assert.equal(env.sent[0].to, "alert-to");
  assert.equal(env.sent[0].subject, "Frame: Jan 16 family photo missing at 9:05 PM");
  const expected = familyPreflightMessage(
    "2026-01-16",
    "family/2026-01-16/photo.json, family/2026-01-16/photo.jpg, family/staged/2026-01-16/photo.json",
    1,
    `${HEALTH}/health-detailed`,
  );
  assert.equal(env.sent[0].text, expected);
  assert.match(expected, /staged unused: 1\. Health: /);
});

test("a ready upload does not mail, and the proof send does not mark the date", async () => {
  const photos = bucket();
  const when = Date.parse("2026-07-16T02:05:00Z");
  photos.seed("family/2026-07-16/photo.json", "{}");
  photos.seed("family/2026-07-16/photo.jpg", new TextEncoder().encode("jpg"));
  const env = preflightEnv(photos, { email: true });
  assert.deepEqual(await runFamilyPreflight(env, when, HEALTH), { posted: false, skipped: "ready" });
  assert.equal(env.sent.length, 0);

  const proof = await sendFamilyPreflightTest(env, when, HEALTH);
  assert.equal(proof.subject, "Frame: Jul 16 family photo missing at 9:05 PM");
  assert.equal(env.sent.length, 1);
  assert.match(env.sent[0].text, /Test of the 9:05 PM family preflight/);
  assert.equal(env.kv.size, 0, "a proof send must not suppress the real night");
});

test("the extra UTC fire does no work, and 21:05 does not run the six-hour warm", async () => {
  const worker = fromBuild("src/index.js").default;
  const noop = {
    CACHE: { get: async () => { throw new Error("noop touched KV"); }, put: async () => { throw new Error("noop wrote KV"); } },
    PHOTOS: { list: async () => { throw new Error("noop listed R2"); }, head: async () => { throw new Error("noop headed R2"); } },
  };
  let pending = Promise.resolve();
  const ctx = { waitUntil(p) { pending = p; } };
  await worker.scheduled({ cron: "5 0,2,3,6,12,18 * * *", scheduledTime: Date.parse("2026-07-16T03:05:00Z") }, noop, ctx);
  await pending;

  const puts = [];
  const preflight = {
    CACHE: { get: async () => null, put: async (k) => { puts.push(k); } },
    PHOTOS: {
      head: async () => null,
      get: async () => null,
      list: async () => ({ objects: [], truncated: false }),
    },
  };
  await worker.scheduled({ cron: "5 0,2,3,6,12,18 * * *", scheduledTime: Date.parse("2026-01-16T03:05:00Z") }, preflight, ctx);
  await pending;
  assert.deepEqual(puts, [], "unset mail means no marker and no weather writes");
});

test("a staged fill is degraded-but-filled and is not a missing daily image", () => {
  const staged = describeProblems({ dateStr: D, missingImages: [], aiBlocked: false, quietMachines: [], familyStaged: "staged" });
  assert.ok(staged.some((p) => p.includes("degraded but filled")));
  assert.equal(staged.some((p) => p.includes("Daily images missing")), false);
  assert.equal(staged.some((p) => p.includes("photo itself is shown")), false);
  const photo = describeProblems({ dateStr: D, missingImages: [], aiBlocked: false, quietMachines: [], familyStaged: "staged-photo" });
  assert.ok(photo.some((p) => p.includes("a staged photo")));
  assert.equal(alertFingerprint({ dateStr: D, missingImages: [], aiBlocked: false, familyStaged: "staged" }), "family:staged");
});
