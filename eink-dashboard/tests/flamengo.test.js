const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");
const zlib = require("node:zlib");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const fla = fromBuild("src/flamengo.js");
const page = fromBuild("src/hud/flamengo-page.js");
const crest = fromBuild("src/hud/crest.js");
const cv = fromBuild("src/hud/canvas.js");
const type = fromBuild("src/hud/type.js");
const desk = fromBuild("src/pages/desk.js");

const HOUR = 3_600_000;
const NOW = Date.parse("2026-10-14T02:00:00Z"); // 13/10 21:00 in Chicago

const FLA = { id: "819", abbreviation: "FLA", displayName: "Flamengo", shortDisplayName: "Flamengo" };
const club = (id, abbr, name) => ({ id, abbreviation: abbr, displayName: name, shortDisplayName: name });
const SERIE_A = { name: "Brazilian Serie A", isTournament: false };
const LIBERTADORES = { name: "CONMEBOL Libertadores", isTournament: true };

/** One event in the shape ESPN's team schedule uses. `score` null means not played. */
function event(id, date, league, home, away, score, extra = {}) {
  const side = (team, homeAway, goals) => ({
    homeAway, team,
    ...(goals === null ? {} : { score: { value: goals, displayValue: String(goals) }, winner: false }),
  });
  return {
    id, date, league, seasonType: { name: extra.stage ?? "2026 Brasileiro Serie A " },
    competitions: [{
      status: { type: { state: score ? "post" : "pre", completed: Boolean(score), ...(extra.status ?? {}) } },
      venue: extra.venue === undefined ? { fullName: "Estadio do Maracana" } : extra.venue,
      competitors: [side(home, "home", score ? score[0] : null), side(away, "away", score ? score[1] : null)],
    }],
  };
}

const REMO = club("4936", "REMO", "Remo");
const IDV = club("17086", "IDV", "Ind. del Valle");
const SANTOS = club("2674", "SAN", "Santos");
const FLU = club("3445", "FLU", "Fluminense");
const EST = club("8", "EST", "Estudiantes");
const BAHIA = club("9967", "BAH", "Bahia");

const RESULTS = {
  events: [
    // newest first, as ESPN sends them
    event("r5", "2026-10-11T20:30Z", SERIE_A, FLA, FLU, [1, 0]),
    event("r4", "2026-10-08T22:30Z", SERIE_A, SANTOS, FLA, [2, 2], { venue: { fullName: "Estádio Vila Belmiro (Urbano Caldeira)" } }),
    event("r3", "2026-09-18T00:30Z", LIBERTADORES, FLA, IDV, [1, 1], { stage: "Quarterfinals" }),
    event("r2", "2026-09-11T00:30Z", LIBERTADORES, IDV, FLA, [0, 2], { stage: "Quarterfinals" }),
    event("r1", "2026-09-06T19:00Z", SERIE_A, REMO, FLA, [1, 0]),
    event("r0", "2026-09-02T22:30Z", SERIE_A, FLA, club("9169", "MIR", "Mirassol"), [2, 0]),
  ],
};
const FIXTURES = {
  events: [
    event("f1", "2026-10-16T00:30Z", LIBERTADORES, EST, FLA, null, { stage: "Semifinals", venue: { fullName: "Estadio Jorge Luis Hirschi" } }),
    event("f2", "2026-10-18T21:30Z", SERIE_A, BAHIA, FLA, null, { venue: null }),
  ],
};

function tableOf(names) {
  const entries = names.map(([id, name], i) => ({
    team: { id, shortDisplayName: name },
    stats: [
      ["rank", i + 1], ["gamesPlayed", 30], ["wins", 19 - i], ["ties", 7], ["losses", 4 + i],
      ["pointDifferential", 33 - i * 4], ["points", 64 - i * 2],
    ].map(([name, value]) => ({ name, value, displayValue: String(value) })),
  }));
  return { children: [{ standings: { entries: entries.reverse() } }] };
}
const TWENTY = [["819", "Flamengo"], ["2029", "Palmeiras"], ["3458", "Athletico-PR"], ["9967", "Bahia"],
  ["7632", "Atlético-MG"], ["2026", "São Paulo"], ["6273", "Grêmio"],
  ...Array.from({ length: 13 }, (_, i) => [`9${i}`, `Club ${i}`])];
const TABLE = tableOf(TWENTY);

/** Stand-ins for ESPN that count what was asked for. */
function sources(over = {}) {
  const asked = [];
  return {
    asked,
    async json(url) {
      asked.push(url);
      if (/standings/.test(url)) return "table" in over ? over.table : TABLE;
      if (/fixture=true/.test(url)) return "fixtures" in over ? over.fixtures : FIXTURES;
      if (/schedule/.test(url)) return "results" in over ? over.results : RESULTS;
      if (/teams\/8$/.test(url)) return { team: { standingSummary: "1st in Argentine Liga Profesional de Fútbol" } };
      return null;
    },
    async crest(id) {
      asked.push(`crest:${id}`);
      return "crest" in over ? over.crest : `bits-${id}`;
    },
  };
}

function kv(initial = {}) {
  const store = new Map(Object.entries(initial));
  const puts = [];
  return {
    store, puts,
    async get(k) { return store.has(k) ? store.get(k) : null; },
    async put(k, v, o) { puts.push([k, o]); store.set(k, v); },
  };
}

function ctx() {
  const pending = [];
  return { pending, waitUntil(p) { pending.push(p); } };
}

const KEY = "flamengo:v1:state";

// --- Reading ESPN ---------------------------------------------------------------

test("a schedule becomes matches in date order, with scores, stage and a tidy venue", () => {
  const m = fla.parseSchedule(RESULTS);
  assert.deepEqual(m.map((x) => x.id), ["r0", "r1", "r2", "r3", "r4", "r5"]);
  const santos = m[4];
  assert.deepEqual([santos.home.name, santos.away.name, santos.homeScore, santos.awayScore, santos.done], ["Santos", "Flamengo", 2, 2, true]);
  assert.equal(santos.venue, "Estádio Vila Belmiro", "the bracketed second name is dropped");
  assert.equal(santos.stage, undefined, "a league match has no round");
  assert.equal(m[2].stage, "Quarterfinals");
  assert.equal(m[2].league, "CONMEBOL Libertadores");
});

test("fixtures have no score and are not results; a postponed match is not a result either", () => {
  const f = fla.parseSchedule(FIXTURES);
  assert.deepEqual([f[0].done, f[0].homeScore, f[0].awayScore], [false, null, null]);
  assert.equal(f[1].venue, undefined);
  const postponed = event("p", "2026-08-01T20:00Z", SERIE_A, FLA, FLU, null, { status: { state: "post", completed: false } });
  assert.equal(fla.parseSchedule({ events: [postponed] })[0].done, false);
});

test("scores are read whether they come as objects, strings or numbers", () => {
  const e = event("x", "2026-10-11T20:30Z", SERIE_A, FLA, FLU, [3, 1]);
  e.competitions[0].competitors[0].score = "3";
  e.competitions[0].competitors[1].score = 1;
  const [m] = fla.parseSchedule({ events: [e] });
  assert.deepEqual([m.homeScore, m.awayScore, m.done], [3, 1, true]);
});

test("malformed answers give nothing rather than throwing", () => {
  for (const junk of [null, undefined, 42, "x", {}, { events: "no" }, { events: [null, {}, { id: "a" }] }]) {
    assert.deepEqual(fla.parseSchedule(junk), []);
    assert.deepEqual(fla.parseTable(junk), []);
    assert.equal(fla.parseStanding(junk), null);
  }
});

test("the table comes out top to bottom with a signed goal difference", () => {
  const t = fla.parseTable(TABLE);
  assert.equal(t.length, 20);
  assert.deepEqual(t[0], { pos: 1, id: "819", name: "Flamengo", p: 30, w: 19, d: 7, l: 4, gd: "+33", pts: 64 });
  assert.equal(t[19].pos, 20);
  assert.equal(t[19].gd, "-43");
  const level = tableOf([["1", "A"]]);
  level.children[0].standings.entries[0].stats.find((s) => s.name === "pointDifferential").value = 0;
  assert.equal(fla.parseTable(level)[0].gd, "0");
});

// --- Loading and the cache ------------------------------------------------------

test("a load gathers the table, results, fixtures, and crest and league place for the next match only", async () => {
  const src = sources();
  const s = await fla.loadFlamengo(NOW, null, src);
  assert.equal(s.fetchedAt, NOW);
  assert.equal(s.table.length, 20);
  assert.deepEqual(s.results.map((m) => m.id), ["r0", "r1", "r2", "r3", "r4", "r5"]);
  assert.deepEqual(s.fixtures.map((m) => m.id), ["f1", "f2"]);
  assert.deepEqual(s.crests, { 8: "bits-8", 819: "bits-819" });
  assert.deepEqual(s.standing, { 8: "1st in Argentine Liga Profesional de Fútbol" }, "Flamengo's place comes from the table");
  assert.ok(!src.asked.some((u) => /teams\/819$/.test(u)));
  assert.ok(!src.asked.includes("crest:9967"), "the match after next needs no crest yet");
});

test("a later load reuses crests it already has", async () => {
  const first = await fla.loadFlamengo(NOW, null, sources());
  const src = sources();
  await fla.loadFlamengo(NOW + HOUR, first, src);
  assert.ok(!src.asked.some((u) => u.startsWith("crest:")));
});

test("a feed that fails keeps what was known; all three failing gives null", async () => {
  const first = await fla.loadFlamengo(NOW, null, sources());
  const s = await fla.loadFlamengo(NOW + HOUR, first, sources({ table: null, results: null }));
  assert.equal(s.table.length, 20);
  assert.equal(s.results.length, 6);
  assert.equal(await fla.loadFlamengo(NOW, first, sources({ table: null, results: null, fixtures: null })), null);
  const thrower = { json: async () => { throw new Error("boom"); }, crest: async () => null };
  assert.equal(await fla.loadFlamengo(NOW, null, thrower), null, "never throws");
});

test("a match that shows up finished in the results is no longer a fixture", async () => {
  const played = { events: [...RESULTS.events, event("f1", "2026-10-16T00:30Z", LIBERTADORES, EST, FLA, [0, 1], { stage: "Semifinals" })] };
  const s = await fla.loadFlamengo(Date.parse("2026-10-16T03:00:00Z"), null, sources({ results: played }));
  assert.deepEqual(s.fixtures.map((m) => m.id), ["f2"]);
  assert.equal(s.results[s.results.length - 1].id, "f1");
});

test("the entry is reread every six hours, and every half hour around a match", async () => {
  const s = await fla.loadFlamengo(NOW, null, sources());
  assert.equal(fla.needsRefresh(s, NOW + 5 * HOUR), false);
  assert.equal(fla.needsRefresh(s, NOW + 6 * HOUR), true);
  const kickoff = Date.parse("2026-10-16T00:30Z");
  const duringMatch = { ...s, fetchedAt: kickoff + 10 * 60_000 };
  assert.equal(fla.needsRefresh(duringMatch, kickoff + 30 * 60_000), false);
  assert.equal(fla.needsRefresh(duringMatch, kickoff + 41 * 60_000), true);
  const beforeMatch = { ...s, fetchedAt: kickoff - 2 * HOUR };
  assert.equal(fla.needsRefresh(beforeMatch, kickoff - HOUR), false, "the short refresh starts at kickoff, not before");
});

test("a fresh entry is served with no fetch and no write", async () => {
  const state = await fla.loadFlamengo(NOW, null, sources());
  const env = { CACHE: kv({ [KEY]: JSON.stringify(state) }) };
  const src = sources();
  const got = await fla.getFlamengo(env, { now: NOW + HOUR, sources: src });
  assert.equal(got.fetchedAt, NOW);
  assert.equal(src.asked.length, 0);
  assert.equal(env.CACHE.puts.length, 0);
});

test("an aged entry is served at once and refreshed behind the response, with one write", async () => {
  const state = await fla.loadFlamengo(NOW, null, sources());
  const env = { CACHE: kv({ [KEY]: JSON.stringify(state) }) };
  const c = ctx();
  const got = await fla.getFlamengo(env, { now: NOW + 7 * HOUR, sources: sources(), ctx: c });
  assert.equal(got.fetchedAt, NOW, "the panel is not kept waiting");
  await Promise.all(c.pending);
  assert.equal(env.CACHE.puts.length, 1);
  assert.deepEqual(env.CACHE.puts[0], [KEY, { expirationTtl: 604800 }]);
  assert.equal(JSON.parse(env.CACHE.store.get(KEY)).fetchedAt, NOW + 7 * HOUR);
});

test("with nothing cached it waits for ESPN; if ESPN is silent it gives null and writes nothing", async () => {
  const env = { CACHE: kv() };
  const got = await fla.getFlamengo(env, { now: NOW, sources: sources() });
  assert.equal(got.table.length, 20);
  assert.equal(env.CACHE.puts.length, 1);

  const empty = { CACHE: kv() };
  const dead = sources({ table: null, results: null, fixtures: null });
  assert.equal(await fla.getFlamengo(empty, { now: NOW, sources: dead }), null);
  assert.equal(empty.CACHE.puts.length, 0);
});

test("an aged entry outlives a dead ESPN, and a corrupt entry is treated as none", async () => {
  const state = await fla.loadFlamengo(NOW, null, sources());
  const env = { CACHE: kv({ [KEY]: JSON.stringify(state) }) };
  const dead = sources({ table: null, results: null, fixtures: null });
  const got = await fla.getFlamengo(env, { now: NOW + 30 * HOUR, sources: dead });
  assert.equal(got.fetchedAt, NOW);
  assert.equal(env.CACHE.puts.length, 0);

  const corrupt = { CACHE: kv({ [KEY]: "{not json" }) };
  assert.equal((await fla.getFlamengo(corrupt, { now: NOW, sources: sources() })).table.length, 20);
});

// --- The view ---------------------------------------------------------------------

test("the view names the next match, the one after, the last result and the form", async () => {
  const v = fla.flamengoView(await fla.loadFlamengo(NOW, null, sources()), NOW);
  assert.equal(v.pill, "1ST  /  64 PTS");
  assert.deepEqual(v.played, { done: 30, of: 38 });
  assert.deepEqual(
    [v.next.home.name, v.next.away.name, v.next.where, v.next.league, v.next.stage, v.next.venue, v.next.live],
    ["Estudiantes", "Flamengo", "AWAY", "COPA LIBERTADORES", "SEMIFINALS", "Estadio Jorge Luis Hirschi", false],
  );
  assert.equal(v.next.home.rank, "1ST IN ARGENTINE LIGA PROFESIONAL DE FÚTBOL");
  assert.equal(v.next.away.rank, "1ST IN SERIE A");
  assert.equal(v.next.home.crest, "bits-8");
  assert.equal(v.countdown, "1D 22H");
  assert.deepEqual([v.then.opponent, v.then.where], ["Bahia", "AWAY"]);
  assert.deepEqual([v.last.score, v.last.outcome, v.last.opponent, v.last.where, v.last.league], ["1-0", "W", "Fluminense", "HOME", "SERIE A"]);
  assert.deepEqual(v.form, [
    { outcome: "L", opponent: "REMO" }, { outcome: "W", opponent: "IDV" }, { outcome: "D", opponent: "IDV" },
    { outcome: "D", opponent: "SAN" }, { outcome: "W", opponent: "FLU" },
  ]);
  assert.equal(v.unbeaten, 4, "counted back to the loss at Remo");
  assert.equal(v.staleDays, 0);
});

test("an away win reads as a win although the score is printed home first", async () => {
  const s = await fla.loadFlamengo(NOW, null, sources({ results: { events: [event("a", "2026-09-06T19:00Z", SERIE_A, REMO, FLA, [0, 1])] } }));
  const v = fla.flamengoView(s, NOW);
  assert.deepEqual([v.last.score, v.last.outcome, v.last.where], ["0-1", "W", "AWAY"]);
});

test("the countdown shortens as kickoff nears, then the match is live, then it moves on", async () => {
  const s = await fla.loadFlamengo(NOW, null, sources());
  const kickoff = Date.parse("2026-10-16T00:30Z");
  assert.equal(fla.flamengoView(s, kickoff - 5 * HOUR - 12 * 60_000).countdown, "5H 12M");
  assert.equal(fla.flamengoView(s, kickoff - 45 * 60_000).countdown, "45M");
  const during = fla.flamengoView(s, kickoff + HOUR);
  assert.deepEqual([during.countdown, during.next.live], ["LIVE", true]);
  const after = fla.flamengoView(s, kickoff + 5 * HOUR);
  assert.equal(after.next.home.name, "Bahia", "an unrefreshed entry still stops showing a match long over");
  assert.equal(after.then, null);
});

test("nothing known gives an empty but whole view, and old data says how old", async () => {
  const v = fla.flamengoView(null, NOW);
  assert.deepEqual([v.pill, v.next, v.last, v.then, v.played, v.countdown, v.unbeaten], ["FLAMENGO", null, null, null, null, "", 0]);
  assert.deepEqual([v.table, v.form], [[], []]);
  const s = await fla.loadFlamengo(NOW, null, sources());
  assert.equal(fla.flamengoView(s, NOW + 50 * HOUR).staleDays, 2);
});

test("ordinals and league names", () => {
  assert.deepEqual([1, 2, 3, 4, 11, 12, 13, 20, 21, 22].map(fla.ordinal),
    ["1ST", "2ND", "3RD", "4TH", "11TH", "12TH", "13TH", "20TH", "21ST", "22ND"]);
  assert.equal(fla.leagueName("Brazilian Serie A"), "SERIE A");
  assert.equal(fla.leagueName("Copa do Brasil"), "COPA DO BRASIL");
});

// --- Crests ---------------------------------------------------------------------

/** A 96x96 RGBA PNG: an opaque black square in the middle of full transparency. */
function crestPng(size = 96) {
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const inside = x >= 32 && x < 64 && y >= 32 && y < 64;
      // transparent pixels are black underneath, as real crests often are
      raw.writeUInt32BE(inside ? 0x000000ff : 0x00000000, y * (size * 4 + 1) + 1 + x * 4);
    }
  }
  const chunk = (type, data) => {
    const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
    const out = Buffer.alloc(body.length + 8);
    out.writeUInt32BE(data.length, 0);
    body.copy(out, 4);
    out.writeUInt32BE(zlib.crc32(body) >>> 0, body.length + 4);
    return out;
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4); ihdr[8] = 8; ihdr[9] = 6;
  return new Uint8Array(Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0)),
  ]));
}

test("a crest becomes a 1-bit bitmap: transparent is paper, however dark it is underneath", async () => {
  const bits = await crest.crestBits(crestPng());
  const c = cv.createCanvas();
  assert.equal(crest.drawCrest(c, bits, 100, 100), true);
  assert.equal(cv.countBlack(c), 32 * 32, "only the opaque square is inked");
  assert.equal(c.px[(100 + 40) * c.w + 100 + 40], 0);
  assert.equal(c.px[(100 + 5) * c.w + 100 + 5], 1);
});

test("anything that is not a usable crest is refused, and drawing nothing draws nothing", async () => {
  assert.equal(await crest.crestBits(crestPng(64)), null, "wrong size");
  assert.equal(await crest.crestBits(new Uint8Array([1, 2, 3])), null);
  const c = cv.createCanvas();
  for (const bad of [undefined, "", "bits-8", "!!!", btoa("short")]) assert.equal(crest.drawCrest(c, bad, 10, 10), false);
  assert.equal(cv.countBlack(c), 0);
});

// --- The page -------------------------------------------------------------------

test("names keep their accents, and a letter the faces lack falls back instead of vanishing", () => {
  assert.equal(page.printable("Atlético-MG"), "ATLÉTICO-MG");
  assert.equal(page.printable("São Paulo"), "SÃO PAULO");
  assert.equal(page.printable("Grêmio"), "GRÊMIO");
  assert.equal(page.printable("Fenerbahçe  Şk"), "FENERBAHÇE SK");
  assert.ok(type.measureType("plex12m", "É") > 0, "the text faces carry accented capitals");
});

test("text too long for its place is cut back to whole words, never mid-word or on a joiner", () => {
  const rank = page.fitWords("plex11m", "1st in Argentine Liga Profesional de Fútbol", 201);
  assert.equal(rank, "1ST IN ARGENTINE LIGA");
  assert.equal(page.fitWords("plex11m", "Short", 201), "SHORT");
  const one = page.fitWords("plex12m", "Supercalifragilistic", 70);
  assert.ok(one.length > 0 && type.measureType("plex12m", one) <= 70);
});

test("the page draws all twenty clubs and both crests, and survives having nothing to show", async () => {
  const state = await fla.loadFlamengo(NOW, null, sources({ crest: await crest.crestBits(crestPng()) }));
  const full = page.drawFlamengoPage(fla.flamengoView(state, NOW), new Date(NOW));
  const row = (i) => { let n = 0; for (let x = 12; x < 330; x++) for (let y = 91 + i * 19; y < 108 + i * 19; y++) n += full.px[y * full.w + x] === 0; return n; };
  for (let i = 0; i < 20; i++) assert.ok(row(i) > 40, `row ${i + 1} is drawn`);
  assert.ok(row(0) > row(1) * 2, "Flamengo's row is the inverted one");
  let last = 0;
  for (let y = 0; y < full.h; y++) for (let x = 0; x < 330; x++) if (full.px[y * full.w + x] === 0) last = y;
  assert.ok(last < 478, `the table ends inside the panel (deepest ink at ${last})`);

  const empty = page.drawFlamengoPage(fla.flamengoView(null, NOW), new Date(NOW));
  assert.ok(cv.countBlack(empty) > 500);
  assert.ok(cv.countBlack(full) > cv.countBlack(empty) * 2);

  const noCrest = await fla.loadFlamengo(NOW, null, sources({ crest: null }));
  assert.ok(cv.countBlack(page.drawFlamengoPage(fla.flamengoView(noCrest, NOW), new Date(NOW))) > 500);
});

test("/desk/flamengo answers with one self-contained page, even when ESPN is silent", async () => {
  const ok = await desk.handleDeskFlamengo({ CACHE: kv() }, undefined, sources());
  assert.equal(ok.status, 200);
  assert.match(await ok.text(), /src="data:image\/png;base64,/);
  const dead = sources({ table: null, results: null, fixtures: null });
  const silent = await desk.handleDeskFlamengo({ CACHE: kv() }, undefined, dead);
  assert.equal(silent.status, 200);
  assert.match(await silent.text(), /src="data:image\/png;base64,/);
});
