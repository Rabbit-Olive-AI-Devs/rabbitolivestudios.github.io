const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const pl = fromBuild("src/planet.js");
const draw = fromBuild("src/hud/planet-draw.js");
const pp = fromBuild("src/hud/planet-page.js");
const cv = fromBuild("src/hud/canvas.js");
const parts = fromBuild("src/hud/parts.js");
const desk = fromBuild("src/pages/desk.js");

const ROW = {
  pl_name: "TRAPPIST-1 e", hostname: "TRAPPIST-1", sy_dist: 12.42988881, pl_rade: 0.92, pl_bmasse: 0.692,
  pl_orbper: 6.101013, pl_eqt: 249.7, disc_year: 2017, discoverymethod: "Transit", sy_snum: 1, sy_pnum: 7,
  st_teff: 2566.0, ra: 346.6263919, dec: -5.0434618,
};
const NAMES = ["51 Peg b", "Kepler-16 b", "Proxima Cen b", "TRAPPIST-1 e"].map((pl_name) => ({ pl_name }));
const YEARS = [{ disc_year: 1995, n: 1 }, { disc_year: 2016, n: 1500 }, { disc_year: 2017, n: 150 }];
const SIBLINGS = ["b", "c", "d", "e", "f", "g", "h"].map((l, i) => ({ pl_name: `TRAPPIST-1 ${l}`, pl_orbsmax: 0.01 + i * 0.01 }));

/** A stand-in for NASA's archive that answers by looking at the query text. */
function archive(log = []) {
  return async (adql) => {
    log.push(adql);
    if (/count\(\*\)/.test(adql)) return YEARS;
    if (/select pl_name from/.test(adql)) return NAMES;
    if (/pl_orbsmax/.test(adql)) return SIBLINGS;
    return [ROW];
  };
}

function kv(initial = {}) {
  const store = new Map(Object.entries(initial));
  const puts = [];
  return {
    store, puts,
    async get(k) { return store.has(k) ? store.get(k) : null; },
    async put(k, v, o) { puts.push({ k, o }); store.set(k, v); },
  };
}

// --- Turning an archive row into a planet -------------------------------------

test("a row becomes a planet with light years and plain fields", () => {
  const p = pl.toPlanet(ROW);
  assert.equal(p.name, "TRAPPIST-1 e");
  assert.equal(p.host, "TRAPPIST-1");
  assert.ok(Math.abs(p.distanceLy - 40.54) < 0.05);
  assert.equal(p.radius, 0.92);
  assert.equal(p.year, 2017);
  assert.equal(p.stars, 1);
  assert.equal(p.planets, 7);
});

test("missing optional fields stay null instead of becoming zero or NaN", () => {
  const p = pl.toPlanet({ ...ROW, pl_bmasse: null, pl_eqt: null, st_teff: null, disc_year: null, sy_dist: null });
  assert.equal(p.mass, null);
  assert.equal(p.tempK, null);
  assert.equal(p.starTempK, null);
  assert.equal(p.year, null);
  assert.equal(p.distanceLy, null);
  assert.equal(pl.toPlanet({ pl_name: "X" }), null, "a row with no radius is not usable");
  assert.equal(pl.toPlanet(null), null);
});

test("the archive query quotes names safely", () => {
  assert.match(pl.planetQuery("Kepler-16 b"), /where pl_name='Kepler-16 b'/);
  assert.match(pl.planetQuery("O'Brien b"), /pl_name='O''Brien b'/, "a quote cannot end the string early");
});

// --- Loading the day's planet --------------------------------------------------

test("loadPlanet picks one planet for the date and fills in its place in the system", async () => {
  const log = [];
  const day = await pl.loadPlanet("2026-10-12", archive(log));
  assert.equal(day.planet.name, "TRAPPIST-1 e");
  assert.equal(day.total, 4);
  assert.ok(day.index >= 0 && day.index < 4);
  assert.equal(day.slot, 3, "fourth planet out from its star");
  assert.deepEqual(day.years, [[1995, 1], [2016, 1500], [2017, 150]]);
  const again = await pl.loadPlanet("2026-10-12", archive());
  assert.equal(again.index, day.index, "the same date picks the same planet");
});

test("loadPlanet returns null when the archive fails or answers with nothing", async () => {
  assert.equal(await pl.loadPlanet("2026-10-12", async () => { throw new Error("down"); }), null);
  assert.equal(await pl.loadPlanet("2026-10-12", async () => []), null);
  assert.equal(await pl.loadPlanet("2026-10-12", async () => "not json rows"), null);
});

test("a cached day is served without asking the archive", async () => {
  const cached = await pl.loadPlanet("2026-10-12", archive());
  const env = { CACHE: kv({ [pl.planetCacheKey("2026-10-12")]: JSON.stringify(cached) }) };
  const day = await pl.getPlanetOfDay(env, "2026-10-12", { query: async () => { throw new Error("must not fetch"); } });
  assert.equal(day.planet.name, "TRAPPIST-1 e");
  assert.equal(env.CACHE.puts.length, 0);
});

test("a cold day is fetched once and stored for a week", async () => {
  const env = { CACHE: kv() };
  const day = await pl.getPlanetOfDay(env, "2026-10-12", { query: archive() });
  assert.equal(day.planet.name, "TRAPPIST-1 e");
  assert.equal(env.CACHE.puts.length, 1, "one KV write per day");
  assert.equal(env.CACHE.puts[0].k, "planet:v1:2026-10-12");
  assert.equal(env.CACHE.puts[0].o.expirationTtl, 604800);
});

test("with yesterday's planet cached, a cold day serves it at once and refreshes in the background", async () => {
  const yesterday = await pl.loadPlanet("2026-10-11", archive());
  const env = { CACHE: kv({ [pl.planetCacheKey("2026-10-11")]: JSON.stringify(yesterday) }) };
  const waits = [];
  let fetched = false;
  const slow = async (q) => { fetched = true; return archive()(q); };
  const day = await pl.getPlanetOfDay(env, "2026-10-12", { query: slow, ctx: { waitUntil: (p) => waits.push(p) } });
  assert.equal(day.planet.name, yesterday.planet.name);
  assert.equal(waits.length, 1, "the refresh is handed to waitUntil");
  await Promise.all(waits);
  assert.ok(fetched);
  assert.ok(env.CACHE.store.has("planet:v1:2026-10-12"), "today's key is filled for the next request");
});

test("when the archive is down and nothing is cached, a built-in planet is shown and nothing is written", async () => {
  const env = { CACHE: kv() };
  const day = await pl.getPlanetOfDay(env, "2026-10-12", { query: async () => { throw new Error("down"); } });
  assert.ok(day.planet.name.length > 0);
  assert.ok(day.planet.radius > 0);
  assert.equal(day.total, 0, "marked as not from the archive");
  assert.equal(env.CACHE.puts.length, 0);
});

test("a date other than today is never written to the cache", async () => {
  const env = { CACHE: kv() };
  await pl.getPlanetOfDay(env, "2031-01-01", { query: archive(), store: false });
  assert.equal(env.CACHE.puts.length, 0, "a ?date= visitor cannot spend KV writes");
});

// --- What the planet looks like --------------------------------------------------

const P = (over) => ({ ...pl.toPlanet(ROW), ...over });

test("size and temperature choose the kind of planet and its label", () => {
  assert.equal(draw.planetLook(P({ radius: 0.9, tempK: 250 })).kind, "ocean");
  assert.equal(draw.planetLook(P({ radius: 1.4, tempK: 2100 })).kind, "lava");
  assert.equal(draw.planetLook(P({ radius: 1.2, tempK: 60 })).kind, "ice");
  assert.equal(draw.planetLook(P({ radius: 1.2, tempK: 600 })).kind, "rock");
  assert.equal(draw.planetLook(P({ radius: 1.2, tempK: null })).kind, "rock");
  assert.equal(draw.planetLook(P({ radius: 4, tempK: 700 })).kind, "neptune");
  assert.equal(draw.planetLook(P({ radius: 13, tempK: 1300 })).kind, "gas");
  assert.equal(draw.planetType(P({ radius: 13, tempK: 1300 })), "HOT GAS GIANT");
  assert.equal(draw.planetType(P({ radius: 0.9, tempK: 250 })), "TEMPERATE ROCKY");
  assert.equal(draw.planetType(P({ radius: 2, tempK: null })), "SUPER-EARTH");
  assert.equal(draw.planetType(P({ radius: 4, tempK: 90 })), "COLD NEPTUNE-LIKE");
});

test("a planet with two suns is lit from two sides", () => {
  assert.equal(draw.planetLook(P({ stars: 1 })).lights.length, 1);
  assert.equal(draw.planetLook(P({ stars: 2 })).lights.length, 2);
});

test("the same planet always draws the same, and different planets draw differently", () => {
  const render = (p) => { const c = cv.createCanvas(260, 260); draw.drawPlanet(c, 130, 130, 96, p); return Buffer.from(c.px).toString("base64"); };
  const a = P({ name: "Kepler-10 b", radius: 1.4, tempK: 2100 });
  assert.equal(render(a), render(a));
  assert.notEqual(render(a), render(P({ name: "Kepler-11 b", radius: 1.4, tempK: 2100 })), "the name sets the pattern");
  assert.notEqual(render(a), render(P({ name: "Kepler-10 b", radius: 12, tempK: 2100 })), "the data sets the kind");
});

test("the planet is a shaded disc: lit on one side, dark on the other, nothing outside its box", () => {
  const c = cv.createCanvas(300, 300);
  draw.drawPlanet(c, 150, 150, 96, P({ radius: 0.9, tempK: 600, name: "Test b" }));
  const black = (x0, y0, x1, y1) => { let n = 0; for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (c.px[y * 300 + x] === 0) n++; return n / ((x1 - x0) * (y1 - y0)); };
  assert.ok(black(90, 90, 130, 130) < black(170, 170, 210, 210) - 0.25, "upper left is lit, lower right is in shadow");
  assert.equal(black(0, 0, 300, 40), 0, "nothing above the planet");
  assert.equal(black(0, 0, 40, 300), 0, "nothing left of it");
});

test("drawing is bounded even when asked for an absurd radius", () => {
  const c = cv.createCanvas(100, 100);
  const t0 = Date.now();
  assert.doesNotThrow(() => draw.drawPlanet(c, 50, 50, 1e9, P({})));
  assert.doesNotThrow(() => draw.drawPlanet(c, 50, 50, NaN, P({})));
  assert.doesNotThrow(() => parts.disc(c, 50, 50, 1e9, true));
  assert.ok(Date.now() - t0 < 500);
});

test("disc draws a filled or an open circle", () => {
  const f = cv.createCanvas(40, 40);
  parts.disc(f, 20, 20, 8, true);
  assert.equal(f.px[20 * 40 + 20], 0);
  const o = cv.createCanvas(40, 40);
  parts.disc(o, 20, 20, 8, false);
  assert.equal(o.px[20 * 40 + 20], 1, "open circle has a white centre");
  assert.equal(o.px[20 * 40 + 28], 0, "and an edge");
});

// --- The page ---------------------------------------------------------------------

test("numbers are written the way a person reads them", () => {
  assert.equal(pp.formatDistance(40.54), "40.5 LY");
  assert.equal(pp.formatDistance(21525.4), "21,525 LY");
  assert.equal(pp.formatDistance(null), "UNKNOWN");
  assert.equal(pp.formatPeriod(6.101), "6.1 DAYS");
  assert.equal(pp.formatPeriod(228.776), "229 DAYS");
  assert.equal(pp.formatPeriod(3285), "9.0 YEARS");
  assert.equal(pp.formatTemp(249.7), "-23 C");
  assert.equal(pp.formatTemp(null), "UNKNOWN");
  assert.equal(pp.formatMass(0.692), "0.69 EARTHS");
  assert.equal(pp.formatMass(193.9), "194 EARTHS");
  assert.equal(pp.formatMass(null), "UNKNOWN");
  assert.equal(pp.formatRa(346.6263919), "RA 23H 07M");
  assert.equal(pp.formatDec(-5.0434618), "DEC -05.0");
  assert.equal(pp.formatDec(20.7690958), "DEC +20.8");
});

test("the planet page is a full panel for small, huge, long-named and sparse planets", async () => {
  const base = await pl.loadPlanet("2026-10-12", archive());
  const cases = [
    base,
    { ...base, planet: { ...base.planet, name: "OGLE-2005-BLG-390L b", radius: 14.1, mass: 3000, stars: 3, planets: 9 }, slot: 8 },
    { ...base, planet: { ...base.planet, mass: null, tempK: null, starTempK: null, year: null, distanceLy: null, ra: null, dec: null }, years: [], total: 0, slot: -1 },
    { ...base, tag: "REAL-LIFE TATOOINE" },
  ];
  for (const day of cases) {
    const c = pp.drawPlanetPage(day, "2026-10-12");
    assert.deepEqual([c.w, c.h], [800, 480]);
    const ratio = cv.countBlack(c) / (800 * 480);
    assert.ok(ratio > 0.05 && ratio < 0.4, `${day.planet.name}: black ratio ${ratio}`);
  }
});

test("a tag changes the page", async () => {
  const base = await pl.loadPlanet("2026-10-12", archive());
  const b64 = (c) => Buffer.from(c.px).toString("base64");
  assert.notEqual(b64(pp.drawPlanetPage(base, "2026-10-12")), b64(pp.drawPlanetPage({ ...base, tag: "REAL-LIFE TATOOINE" }, "2026-10-12")));
});

test("/desk/planet serves one inlined page and survives a dead archive", async () => {
  const env = { CACHE: kv() };
  const res = await desk.handleDeskPlanet(env, new URL("https://x/desk/planet"), undefined, async () => { throw new Error("down"); });
  assert.equal(res.status, 200);
  const html = await res.text();
  assert.match(html, /src="data:image\/png;base64,/);
  assert.doesNotMatch(html, /<script/i);
});
