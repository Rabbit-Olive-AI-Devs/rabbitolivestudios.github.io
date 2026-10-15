const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const clock = fromBuild("src/series/clock.js");
const keys = fromBuild("src/series/keys.js");
const types = fromBuild("src/series/types.js");

/** A Chicago wall-clock time as a Date. October 2026 is CDT (UTC-5); January is CST (UTC-6). */
function chi(iso, offsetHours) {
  return new Date(Date.parse(iso + "Z") + offsetHours * 3600 * 1000);
}
const CDT = 5, CST = 6;

const SCHEDULE = {
  seasons: [
    { number: 0, show: "THE LONG DRIFT", title: "PROLOGUE", writer: "Claude", premiere: "2026-10-26", chapters: 10, approved: [1, 2], pending: [3] },
    { number: 1, show: "THE LONG DRIFT", title: "S1", writer: "OpenAI", premiere: "2027-01-04", chapters: 13, approved: [1], pending: [] },
  ],
};

test("prologue has 10 chapters, other seasons 13", () => {
  assert.equal(types.plannedChapters(0), 10);
  assert.equal(types.plannedChapters(1), 13);
});

test("weekday scenes change at 08, 10, 12, 14, 16 Chicago", () => {
  const at = (hhmm) => clock.seriesSlot(chi(`2026-10-26T${hhmm}:00`, CDT), SCHEDULE);
  assert.deepEqual(at("08:00"), { kind: "scene", season: 0, chapter: 1, scene: 1, replay: false });
  assert.deepEqual(at("09:59"), { kind: "scene", season: 0, chapter: 1, scene: 1, replay: false });
  assert.deepEqual(at("10:00"), { kind: "scene", season: 0, chapter: 1, scene: 2, replay: false });
  assert.deepEqual(at("16:00"), { kind: "scene", season: 0, chapter: 1, scene: 5, replay: false });
  assert.deepEqual(at("23:59"), { kind: "scene", season: 0, chapter: 1, scene: 5, replay: false });
});

test("Tuesday before 08:00 still shows Monday's last scene; Friday 16:00 is scene 25", () => {
  assert.deepEqual(clock.seriesSlot(chi("2026-10-27T07:59:00", CDT), SCHEDULE),
    { kind: "scene", season: 0, chapter: 1, scene: 5, replay: false });
  assert.deepEqual(clock.seriesSlot(chi("2026-10-30T16:00:00", CDT), SCHEDULE),
    { kind: "scene", season: 0, chapter: 1, scene: 25, replay: false });
});

test("weekend replays the week, then the teaser from Sunday 19:00 to Monday 08:00", () => {
  // 1 Nov 2026 is the clock change: Saturday is CDT, Sunday after 02:00 is CST.
  const at = (iso) => clock.seriesSlot(chi(iso, iso.startsWith("2026-11-01") ? CST : CDT), SCHEDULE);
  assert.deepEqual(at("2026-10-31T07:59:00"), { kind: "scene", season: 0, chapter: 1, scene: 25, replay: false });
  assert.deepEqual(at("2026-10-31T08:00:00"), { kind: "scene", season: 0, chapter: 1, scene: 1, replay: true });
  assert.deepEqual(at("2026-10-31T21:00:00"), { kind: "scene", season: 0, chapter: 1, scene: 14, replay: true });
  assert.deepEqual(at("2026-10-31T23:30:00"), { kind: "scene", season: 0, chapter: 1, scene: 14, replay: true });
  assert.deepEqual(at("2026-11-01T07:00:00"), { kind: "scene", season: 0, chapter: 1, scene: 14, replay: true });
  assert.deepEqual(at("2026-11-01T08:00:00"), { kind: "scene", season: 0, chapter: 1, scene: 15, replay: true });
  assert.deepEqual(at("2026-11-01T18:59:00"), { kind: "scene", season: 0, chapter: 1, scene: 25, replay: true });
  assert.deepEqual(at("2026-11-01T19:00:00"), { kind: "teaser", season: 0, chapter: 1 });
});

test("Monday 00:00-07:59 still shows the previous chapter's teaser", () => {
  // 1 Nov 2026 is the DST change (02:00 CDT -> 01:00 CST); Monday 2 Nov is CST.
  assert.deepEqual(clock.seriesSlot(chi("2026-11-02T07:59:00", CST), SCHEDULE), { kind: "teaser", season: 0, chapter: 1 });
  assert.deepEqual(clock.seriesSlot(chi("2026-11-02T08:00:00", CST), SCHEDULE),
    { kind: "scene", season: 0, chapter: 2, scene: 1, replay: false });
});

test("the DST change night itself holds replay scene 14 on every instant", () => {
  for (const utc of ["2026-11-01T06:30:00Z", "2026-11-01T07:30:00Z", "2026-11-01T08:30:00Z"]) {
    assert.deepEqual(clock.seriesSlot(new Date(utc), SCHEDULE), { kind: "scene", season: 0, chapter: 1, scene: 14, replay: true });
  }
});

test("before a premiere and in a gap the slot is between seasons", () => {
  assert.deepEqual(clock.seriesSlot(chi("2026-10-20T12:00:00", CDT), SCHEDULE), { kind: "between", next: 0, prev: null });
  // Premiere Monday before 08:00 belongs to Sunday night: still before the premiere.
  assert.deepEqual(clock.seriesSlot(chi("2026-10-26T07:00:00", CDT), SCHEDULE), { kind: "between", next: 0, prev: null });
  const late = { seasons: [SCHEDULE.seasons[0]] };
  assert.deepEqual(clock.seriesSlot(chi("2027-01-04T12:00:00", CST), late), { kind: "between", next: null, prev: 0 });
});

test("a season's last Sunday evening is its finale teaser, and the next Monday 08:00 starts the next season", () => {
  assert.deepEqual(clock.seriesSlot(chi("2027-01-03T20:00:00", CST), SCHEDULE), { kind: "teaser", season: 0, chapter: 10 });
  assert.deepEqual(clock.seriesSlot(chi("2027-01-04T08:00:00", CST), SCHEDULE),
    { kind: "scene", season: 1, chapter: 1, scene: 1, replay: false });
});

test("an unapproved chapter falls back to the last approved chapter's final scene", () => {
  const slot = { kind: "scene", season: 0, chapter: 3, scene: 7, replay: false };
  assert.deepEqual(clock.resolveSlot(slot, SCHEDULE, false), { kind: "scene", season: 0, chapter: 2, scene: 25, replay: false });
  assert.deepEqual(clock.resolveSlot(slot, SCHEDULE, true), slot); // previews see pending chapters
  const none = { kind: "teaser", season: 1, chapter: 4 };
  assert.deepEqual(clock.resolveSlot({ kind: "scene", season: 0, chapter: 1, scene: 3, replay: false },
    { seasons: [{ ...SCHEDULE.seasons[0], approved: [] }] }, false), { kind: "between", next: 0, prev: null });
  assert.deepEqual(clock.resolveSlot(none, SCHEDULE, false), { kind: "scene", season: 1, chapter: 1, scene: 25, replay: false });
});

test("scene time labels", () => {
  assert.equal(clock.sceneTimeLabel(1), "MON 08:00");
  assert.equal(clock.sceneTimeLabel(25), "FRI 16:00");
  assert.equal(clock.nextSceneLabel(1), "10:00");
  assert.equal(clock.nextSceneLabel(5), "TUE 08:00");
  assert.equal(clock.nextSceneLabel(25), "MON 08:00");
});

test("R2 keys", () => {
  assert.equal(keys.SCHEDULE_KEY, "series/schedule.json");
  assert.equal(keys.sceneKey(0, 3, 14, false, "desk"), "series/s00/c03/s14-desk.png");
  assert.equal(keys.sceneKey(0, 3, 14, true, "office"), "series/s00/c03/r14-office.png");
  assert.equal(keys.teaserKey(1, 13, "office"), "series/s01/c13/teaser-office.png");
  assert.equal(keys.cardKey(0, "before", "desk"), "series/s00/before-desk.png");
  assert.equal(keys.slotKey({ kind: "between", next: 1, prev: 0 }, "desk"), "series/s01/before-desk.png");
  assert.equal(keys.slotKey({ kind: "between", next: null, prev: 0 }, "desk"), "series/s00/after-desk.png");
  assert.equal(keys.slotKey({ kind: "between", next: null, prev: null }, "desk"), null);
});

const tpl = fromBuild("src/series/templates.js");
const art = fromBuild("src/series/art.js");
const conv = fromBuild("src/series/convert.js");

function solid(w, h, [r, g, b]) {
  const rgb = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) { rgb[i * 3] = r; rgb[i * 3 + 1] = g; rgb[i * 3 + 2] = b; }
  return { w, h, rgb };
}

test("every template panel lies inside the page and below the top bar", () => {
  for (const t of Object.values(tpl.TEMPLATES)) {
    for (const p of t.panels) {
      assert.ok(p.x >= 0 && p.y >= 37 && p.x + p.w <= 800 && p.y + p.h <= 480, `${t.id} panel out of page`);
      if (t.strip) assert.ok(p.y + p.h <= tpl.STRIP_TOP, `${t.id} panel runs into the strip`);
    }
  }
  assert.deepEqual(tpl.TEMPLATES.wide.panels, [{ x: 12, y: 46, w: 776, h: 346 }]);
  assert.equal(tpl.TEMPLATES.split.panels.length, 2);
  assert.equal(tpl.TEMPLATES.strip.panels.length, 3);
});

test("transparent pixels become white paper", () => {
  const rgb = new Uint8Array([0, 0, 0, 0, 0, 0]);
  const alpha = new Uint8Array([0, 255]);
  assert.deepEqual([...art.flattenAlpha(rgb, alpha)], [255, 255, 255, 0, 0, 0]);
  assert.deepEqual([...art.flattenAlpha(rgb, undefined)], [0, 0, 0, 0, 0, 0]);
});

test("cropToFill keeps the focus point and maps points into the panel", () => {
  // 300x100 image: left half black, right half white; crop to a 100x100 panel around x=250.
  const img = solid(300, 100, [0, 0, 0]);
  for (let y = 0; y < 100; y++) for (let x = 150; x < 300; x++) img.rgb.fill(255, (y * 300 + x) * 3, (y * 300 + x) * 3 + 3);
  const { image, map } = art.cropToFill(img, 100, 100, [250, 50]);
  assert.equal(image.w, 100);
  assert.equal(image.h, 100);
  assert.equal(image.rgb[(50 * 100 + 50) * 3], 255); // centre is white: the focus was kept
  assert.deepEqual(map([250, 50]), [50, 50]);
  assert.deepEqual(map([150, 0]), [-50, 0]); // left of the crop: outside the panel
});

test("cropToFill shrinks a larger picture to cover the panel", () => {
  const { image, map } = art.cropToFill(solid(1920, 1280, [200, 10, 10]), 776, 346);
  assert.equal(image.w, 776);
  assert.equal(image.h, 346);
  assert.equal(image.rgb[0], 200);
  const [x, y] = map([960, 640]);
  assert.ok(Math.abs(x - 388) <= 1 && Math.abs(y - 173) <= 1);
});

test("a nearly blank picture is detected", () => {
  assert.equal(art.isNearlyBlank(solid(50, 50, [250, 250, 250])), true);
  const busy = solid(50, 50, [0, 0, 0]);
  for (let i = 0; i < busy.rgb.length; i += 6) busy.rgb[i] = busy.rgb[i + 1] = busy.rgb[i + 2] = 255;
  assert.equal(art.isNearlyBlank(busy), false);
});

test("desk conversion is pure black and white", () => {
  const gray = new Uint8Array(64 * 64).map((_, i) => i % 256);
  const bits = conv.toDeskBits(gray, 64, 64);
  assert.ok([...bits].every((b) => b === 0 || b === 1));
  assert.equal(conv.toHomeGray, undefined, "the four-grey version is gone: both E1001s show the desk version");
});

test("office conversion keeps greys black-and-white and saturated red red", () => {
  const grey = solid(32, 32, [90, 90, 90]);
  const idx = conv.toOfficeIndices(grey.rgb, 32, 32);
  assert.ok([...idx].every((k) => k === 0 || k === 1), "a grey used a coloured ink");
  const red = solid(32, 32, [178, 19, 24]);
  const ridx = conv.toOfficeIndices(red.rgb, 32, 32);
  assert.ok([...ridx].every((k) => k === 2));
});

const letter = fromBuild("src/series/letter.js");
const cvs = fromBuild("src/hud/canvas.js");

const PANEL = { x: 12, y: 46, w: 776, h: 346 };
const blackCount = (c) => [...c.px].filter((v) => v === 0).length;

test("a balloon is sized to its text and every letter lies inside it", () => {
  const L = letter.newLayers();
  const r = letter.placeBalloon(L, { speaker: "BOLT", text: "I have calculated our odds. I will not be sharing them.", anchor: [600, 260] }, PANEL, [], [[600, 260]]);
  assert.equal(typeof r, "object", r);
  assert.ok(r.w < 400 && r.h < 140, "balloon not sized to its text");
  assert.ok(r.x >= PANEL.x && r.y >= PANEL.y && r.x + r.w <= PANEL.x + PANEL.w && r.y + r.h <= PANEL.y + PANEL.h);
  // letters landed on paper (placeBalloon already checked them against the ellipse)
  // Black pixels inside the ellipse are letters; the outline sits just outside it.
  const cx = r.x + r.w / 2, cy = r.y + r.h / 2, rx = r.w / 2, ry = r.h / 2;
  let lettersOffPaper = 0;
  for (let y = r.y; y < r.y + r.h; y++) for (let x = r.x; x < r.x + r.w; x++) {
    const i = y * 800 + x;
    const inEllipse = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1;
    if (inEllipse && L.ink.px[i] === 0 && L.paper.px[i] !== 0) lettersOffPaper++;
  }
  assert.equal(lettersOffPaper, 0);
  assert.ok(blackCount(L.ink) > 200);
});

test("a balloon never covers a speaker or a keep-clear box", () => {
  const L = letter.newLayers();
  const keep = { x: 300, y: 46, w: 488, h: 346 }; // everything right of x=300 is a character
  const r = letter.placeBalloon(L, { speaker: "VESS", text: "Hold still.", anchor: [250, 300] }, PANEL, [keep], [[250, 300]]);
  assert.equal(typeof r, "object", r);
  assert.ok(r.x + r.w <= keep.x - 6, "balloon overlaps the keep-clear box");
  const dx = 250 - (r.x + r.w / 2), dy = 300 - (r.y + r.h / 2);
  assert.ok(((dx / (r.w / 2)) ** 2 + (dy / (r.h / 2)) ** 2) > 1, "balloon covers the speaker's head");
});

test("a line too long for four lines is refused with the speaker's name", () => {
  const L = letter.newLayers();
  const long = "word ".repeat(80).trim();
  const r = letter.placeBalloon(L, { speaker: "FIGURE", text: long, anchor: [400, 300] }, PANEL, [], [[400, 300]]);
  assert.match(r, /^FIGURE: line too long for any balloon/);
});

test("no room for a balloon is refused", () => {
  const L = letter.newLayers();
  const r = letter.placeBalloon(L, { speaker: "BOLT", text: "Hello there, friend.", anchor: [400, 200] }, PANEL, [PANEL], [[400, 200]]);
  assert.match(r, /^BOLT: no room for the balloon/);
});

test("an off-picture voice gets a square box with the speaker's name", () => {
  const L = letter.newLayers();
  const r = letter.placeBalloon(L, { speaker: "PATROL", text: "Sweep the district again.", anchor: null }, PANEL, [], []);
  assert.equal(typeof r, "object", r);
  // the name bar is solid black across the top of the box
  assert.equal(L.ink.px[(r.y + 3) * 800 + r.x + 4], 0);
});

test("caption boxes and text blocks refuse what does not fit", () => {
  const L = letter.newLayers();
  assert.equal(typeof letter.captionBox(L, 22, 56, "MERREK PRIME. THE SALT DOCKS.", 760, "caption"), "object");
  assert.match(letter.captionBox(L, 22, 56, "X".repeat(200), 300, "caption"), /^caption: too long/);
  const c = cvs.createCanvas();
  assert.equal(letter.textBlock(c, "plex15m", 16, 420, "Short narration.", 548, 2, 21, "narration"), null);
  assert.match(letter.textBlock(c, "plex15m", 16, 420, "word ".repeat(60), 548, 2, 21, "narration"), /^narration: needs \d+ lines, room for 2/);
});

const pkgv = fromBuild("src/series/package.js");

const artFor = (p) => new Set(p.chapters.flatMap((c) => c.scenes.flatMap((s) => s.panels.map((x) => x.art))).filter((a) => a !== "hold"));

test("the sample package is valid", () => {
  const p = pkgv.samplePackage();
  const { pkg, problems } = pkgv.validatePackage(p, artFor(p), []);
  assert.deepEqual(problems, []);
  assert.equal(pkg.chapters[0].scenes.length, 25);
});

test("structure problems are listed by chapter and scene", () => {
  const p = pkgv.samplePackage();
  p.season.premiere = "2026-10-27"; // a Tuesday
  p.chapters[0].scenes[0].template = "wide"; // scene 1 must be a splash
  p.chapters[0].scenes[3].template = "comic";
  p.chapters[0].scenes.pop(); // 24 scenes
  const { problems } = pkgv.validatePackage(p, artFor(p), []);
  assert.ok(problems.some((x) => /premiere 2026-10-27 is not a Monday/.test(x)));
  assert.ok(problems.some((x) => /chapter 1: 24 scenes, needs 25/.test(x)));
  assert.ok(problems.some((x) => /chapter 1 scene 1: must be a splash/.test(x)));
  assert.ok(problems.some((x) => /chapter 1 scene 4: unknown template "comic"/.test(x)));
});

test("typographic characters are named with their place", () => {
  const p = pkgv.samplePackage();
  p.chapters[0].scenes[1].narration = "She said “hello” — then left…";
  const { problems } = pkgv.validatePackage(p, artFor(p), []);
  const hit = problems.find((x) => x.startsWith("chapter 1 scene 2 narration"));
  assert.ok(hit, problems.join("\n"));
  assert.match(hit, /U\+201C/);
  assert.match(hit, /U\+2014/);
  assert.match(hit, /U\+2026/);
});

test("banned names are refused, built-in and extra", () => {
  const p = pkgv.samplePackage();
  p.chapters[0].scenes[2].panels[0].balloons = [{ speaker: "VESS", text: "That ship belongs to Han Solo.", anchor: [100, 100] }];
  p.chapters[0].scenes[3].narration = "Grogu was not there.";
  p.chapters[0].scenes[4].caption = "SECRET BASE";
  const { problems } = pkgv.validatePackage(p, artFor(p), ["secret base"]);
  assert.ok(problems.some((x) => /scene 3 .*"Han Solo"/.test(x)));
  assert.ok(problems.some((x) => /scene 4 .*"Grogu"/.test(x)));
  assert.ok(problems.some((x) => /scene 5 .*"secret base"/.test(x)));
});

test("art: missing files, hold after another template, and a weekday with no new picture", () => {
  const p = pkgv.samplePackage();
  for (let s = 6; s <= 10; s++) p.chapters[0].scenes[s - 1].panels[0].art = "hold"; // Tuesday all held
  p.chapters[0].scenes[1].panels[0].art = "hold"; // scene 2 cannot hold the splash
  p.chapters[0].scenes[2].panels[0].art = "missing.png";
  const files = artFor(pkgv.samplePackage());
  const { problems } = pkgv.validatePackage(p, files, []);
  assert.ok(problems.some((x) => /chapter 1 scene 3: art file "missing.png" not in art\//.test(x)));
  assert.ok(problems.some((x) => /chapter 1 TUE: no new picture/.test(x)));
  assert.ok(problems.some((x) => /chapter 1 scene 2: "hold" after a splash/.test(x)));
});

test("too many balloons and chapters past the season's length are refused", () => {
  const p = pkgv.samplePackage();
  const b = { speaker: "A", text: "Hi.", anchor: [1, 1] };
  p.chapters[0].scenes[1].panels[0].balloons = [b, b, b];
  p.season.number = 0;
  p.chapters = Array.from({ length: 11 }, (_, i) => ({ ...p.chapters[0], number: i + 1 }));
  const { problems } = pkgv.validatePackage(p, artFor(p), []);
  assert.ok(problems.some((x) => /scene 2: 3 balloons in panel 1, room for 2/.test(x)));
  assert.ok(problems.some((x) => /11 chapters, season 0 has 10/.test(x)));
});

const render = fromBuild("src/series/render.js");
const { decodePNG } = fromBuild("src/png-decode.js");

/** Width and height from a PNG's IHDR chunk; works for every bit depth (decodePNG reads no 1-bit). */
function pngSize(bytes) {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  assert.equal(String.fromCharCode(...bytes.slice(12, 16)), "IHDR");
  return { width: v.getUint32(16), height: v.getUint32(20) };
}

function panelArt(w, h) {
  const img = solid(w, h, [40, 40, 40]);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if ((x >> 4) % 2) img.rgb.fill(178, (y * w + x) * 3, (y * w + x) * 3 + 1);
  return img;
}

test("a wide scene renders four 800x480 pages: two versions, live and replay", async () => {
  const p = pkgv.samplePackage();
  const sc = p.chapters[0].scenes[1];
  const { pages, problems } = await render.renderScene(p.season, p.chapters[0], sc,
    [{ image: panelArt(776, 346), anchors: [[612, 246]], clear: [] }]);
  assert.deepEqual(problems, []);
  assert.deepEqual(Object.keys(pages).sort(), ["r-desk", "r-office", "s-desk", "s-office"]);
  for (const bytes of Object.values(pages)) {
    const d = pngSize(bytes);
    assert.equal(d.width, 800);
    assert.equal(d.height, 480);
  }
  assert.notDeepEqual(pages["s-desk"], pages["r-desk"], "replay page must differ (REPLAY mark)");
});

test("lettering problems surface from renderScene instead of a page", async () => {
  const p = pkgv.samplePackage();
  const sc = { ...p.chapters[0].scenes[1], narration: "word ".repeat(60) };
  const { pages, problems } = await render.renderScene(p.season, p.chapters[0], sc,
    [{ image: panelArt(776, 346), anchors: [[612, 246]], clear: [] }]);
  assert.ok(problems.some((x) => /narration: needs \d+ lines/.test(x)));
  assert.deepEqual(pages, {});
});

test("the splash title card letters the chapter automatically", async () => {
  const p = pkgv.samplePackage();
  const { pages, problems } = await render.renderScene(p.season, p.chapters[0], p.chapters[0].scenes[0],
    [{ image: panelArt(800, 443), anchors: [], clear: [] }]);
  assert.deepEqual(problems, []);
  assert.ok(pages["s-office"].length > 1000);
});

test("teaser and cards are 800x480 PNGs; pills name the prologue", async () => {
  const p = pkgv.samplePackage();
  for (const bytes of [await render.renderTeaser(p.season, p.chapters[0], false), await render.renderCard(p.season, "before"), await render.renderCard(p.season, "after")]) {
    const d = pngSize(bytes);
    assert.equal(d.width, 800);
    assert.equal(d.height, 480);
  }
  assert.equal(render.pillFor(p.season, 3), "P / CH 03");
  assert.equal(render.pillFor({ ...p.season, number: 2 }, 11), "S2 / CH 11");
});

const build = fromBuild("src/series/build.js");

const artSource = (w = 1200, h = 800) => (name) => (name.endsWith(".png") ? panelArt(w, h) : null);

test("a chapter builds 102 files under the right keys: two versions, live and replay", async () => {
  const p = pkgv.samplePackage();
  const { files, problems } = await build.buildChapter(p, 1, artSource());
  assert.deepEqual(problems, []);
  assert.equal(files.size, 25 * 4 + 2);
  assert.ok(files.has("series/s00/c01/s01-desk.png"));
  assert.ok(files.has("series/s00/c01/r25-office.png"));
  assert.ok(files.has("series/s00/c01/teaser-desk.png"));
  assert.ok(![...files.keys()].some((k) => k.includes("-home")), "no third (grey) version");
});

test("hold reuses the previous scene's picture", async () => {
  const p = pkgv.samplePackage();
  p.chapters[0].scenes[2].panels[0].art = "hold";
  const seen = [];
  const { problems } = await build.buildChapter(p, 1, (n) => { seen.push(n); return panelArt(1200, 800); });
  assert.deepEqual(problems, []);
  assert.ok(!seen.includes("hold"));
  assert.ok(!seen.includes("c01-s03.png"));
});

test("a picture smaller than its panel is refused", async () => {
  const p = pkgv.samplePackage();
  const { problems } = await build.buildChapter(p, 1, artSource(400, 200));
  assert.ok(problems.some((x) => /chapter 1 scene 2: picture c01-s02.png is 400x200, smaller than the 776x346 panel/.test(x)));
});

test("an anchor cropped out of the panel is refused", async () => {
  const p = pkgv.samplePackage();
  // 3:2 picture in a wide panel loses top and bottom; y=5 is cut away
  p.chapters[0].scenes[1].panels[0].balloons[0].anchor = [600, 5];
  const { problems } = await build.buildChapter(p, 1, artSource(1920, 1280));
  assert.ok(problems.some((x) => /chapter 1 scene 2: balloon 1 anchor \[600, 5\] is cropped out of the panel/.test(x)));
});

test("a nearly blank picture is refused", async () => {
  const p = pkgv.samplePackage();
  const { problems } = await build.buildChapter(p, 1, () => solid(1200, 800, [250, 250, 250]));
  assert.ok(problems.some((x) => /picture c01-s02.png is nearly blank/.test(x)));
});

const sp = fromBuild("src/pages/series.js");

/** An in-memory stand-in for an R2 bucket. */
function bucket(objects) {
  return {
    async get(key) {
      if (!(key in objects)) return null;
      const v = objects[key];
      const bytes = typeof v === "string" ? new TextEncoder().encode(v) : v;
      return { arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), text: async () => new TextDecoder().decode(bytes) };
    },
  };
}
const PNG = new Uint8Array([137, 80, 78, 71, 1, 2, 3]);

test("a panel route serves the slot's page inline as one document", async () => {
  const env = { SERIES: bucket({ "series/schedule.json": JSON.stringify(SCHEDULE), "series/s00/c01/s02-office.png": PNG }) };
  const res = await sp.handleSeries(env, new URL("https://x/color/series"), "office", chi("2026-10-26T10:30:00", CDT));
  assert.equal(res.status, 200);
  const html = await res.text();
  assert.match(html, /data:image\/png;base64,iVBORw/);
  assert.doesNotMatch(html, /<script/);
});

test("missing or broken schedule, or a missing page, gives the text page, never a 500", async () => {
  for (const objects of [{}, { "series/schedule.json": "{not json" }, { "series/schedule.json": JSON.stringify(SCHEDULE) }]) {
    const res = await sp.handleSeries({ SERIES: bucket(objects) }, new URL("https://x/series"), "desk", chi("2026-10-26T10:30:00", CDT));
    assert.equal(res.status, 200);
    const html = await res.text();
    assert.doesNotMatch(html, /data:image/);
    assert.match(html, /<h1>/);
  }
});

test("previews with ?at= need SERIES_KEY when it is set, and see pending chapters", async () => {
  const objects = { "series/schedule.json": JSON.stringify(SCHEDULE), "series/s00/c03/s01-desk.png": PNG };
  const env = { SERIES: bucket(objects), SERIES_KEY: "k" };
  const denied = await sp.handleSeries(env, new URL("https://x/desk/series?at=2026-11-09T08:30"), "desk");
  assert.equal(denied.status, 404);
  const ok = await sp.handleSeries(env, new URL("https://x/desk/series?at=2026-11-09T08:30&key=k"), "desk");
  assert.match(await ok.text(), /data:image\/png/);
});

test("the proof page needs the key and lists every scene of a chapter", async () => {
  const objects = { "series/schedule.json": JSON.stringify(SCHEDULE) };
  for (let s = 1; s <= 25; s++) objects[`series/s00/c03/s${String(s).padStart(2, "0")}-office.png`] = PNG;
  const env = { SERIES: bucket(objects), SERIES_KEY: "k" };
  assert.equal((await sp.handleSeriesProof(env, new URL("https://x/series/proof?season=0&chapter=3"))).status, 404);
  const html = await (await sp.handleSeriesProof(env, new URL("https://x/series/proof?season=0&chapter=3&key=k"))).text();
  assert.equal((html.match(/<img /g) || []).length, 26); // 25 scenes + teaser, linked not inlined
  assert.match(html, /PENDING/);
});

const watch = fromBuild("src/series/watch.js");

test("an unapproved chapter airing within 7 days is a problem; further out is not", () => {
  // Chapter 3 of the prologue airs Mon 9 Nov; it is only pending.
  assert.deepEqual(watch.scheduleProblems(SCHEDULE, chi("2026-11-01T12:00:00", CST)), []);
  const p = watch.scheduleProblems(SCHEDULE, chi("2026-11-03T12:00:00", CST));
  assert.equal(p.length, 1);
  assert.match(p[0], /chapter 3 of the Prologue airs 09\/11 and is waiting for approval/);
});

test("a season ending within 14 days with no next season is a problem", () => {
  const only = { seasons: [{ ...SCHEDULE.seasons[0], approved: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], pending: [] }] };
  assert.deepEqual(watch.scheduleProblems(only, chi("2026-12-10T12:00:00", CST)), []);
  assert.match(watch.scheduleProblems(only, chi("2026-12-22T12:00:00", CST))[0], /the Prologue ends 01\/01 and no next season is imported/);
});

test("no schedule at all is not an alert before anything was ever imported", () => {
  assert.deepEqual(watch.scheduleProblems(null, new Date()), []);
});

test("end to end: PNG pictures decode, build and give valid pages", async () => {
  const { encodePNGIndexed } = fromBuild("src/png.js");
  const { SPECTRA6_PALETTE } = fromBuild("src/spectra6.js");
  const idx = new Uint8Array(1200 * 800).map((_, i) => ((i % 1200) >> 5) % 6);
  const png = await encodePNGIndexed(idx, 1200, 800, SPECTRA6_PALETTE);
  const d = await decodePNG(png);
  const pic = { w: d.width, h: d.height, rgb: art.flattenAlpha(d.rgb, d.alpha) };
  const { files, problems } = await build.buildChapter(pkgv.samplePackage(), 1, () => pic);
  assert.deepEqual(problems, []);
  const page = await decodePNG(files.get("series/s00/c01/s02-office.png"));
  assert.equal(page.width, 800);
});

// --- Final review fixes ---

const sched = fromBuild("src/series/schedule.js");

test("review: an unapproved chapter keeps alerting through its airing week", () => {
  // Chapter 3 airs Mon 9 Nov and is only pending; Wednesday 11 Nov the panels show chapter 2.
  const p = watch.scheduleProblems(SCHEDULE, chi("2026-11-11T12:00:00", CST));
  assert.ok(p.some((x) => /chapter 3 of the Prologue/.test(x)), p.join("\n"));
});

test("review: a failing series check reports a problem instead of throwing", async () => {
  const env = { SERIES: { get: async () => { throw new Error("R2 down"); } } };
  const p = await watch.seriesProblems(env, new Date());
  assert.equal(p.length, 1);
  assert.match(p[0], /^The series: check failed: R2 down/);
});

test("review: a schedule season with the wrong shape gives the text page, not a 500", async () => {
  const bad = { seasons: [{ number: 0, premiere: "2026-10-26", chapters: 10 }] }; // no approved/pending
  const env = { SERIES: bucket({ "series/schedule.json": JSON.stringify(bad) }) };
  const res = await sp.handleSeries(env, new URL("https://x/series"), "desk", chi("2026-10-26T10:30:00", CDT));
  assert.equal(res.status, 200);
  assert.match(await res.text(), /<h1>/);
  assert.equal(sched.isSchedule(bad), false);
  assert.equal(sched.isSchedule(SCHEDULE), true);
});

test("review: a held panel's anchor outside the panel is refused", async () => {
  const p = pkgv.samplePackage();
  p.chapters[0].scenes[2].panels[0].art = "hold";
  p.chapters[0].scenes[2].panels[0].balloons[0].anchor = [900, 100]; // panel is 776 wide
  const { problems } = await build.buildChapter(p, 1, artSource());
  assert.ok(problems.some((x) => /chapter 1 scene 3: balloon 1 anchor \[900, 100\] is outside the held panel/.test(x)), problems.join("\n"));
});

test("review: the proof page links its images instead of inlining them", async () => {
  const objects = { "series/schedule.json": JSON.stringify(SCHEDULE) };
  for (let s = 1; s <= 25; s++) objects[`series/s00/c03/s${String(s).padStart(2, "0")}-office.png`] = PNG;
  const env = { SERIES: { ...bucket(objects), async head(k) { return k in objects ? {} : null; } }, SERIES_KEY: "k" };
  const html = await (await sp.handleSeriesProof(env, new URL("https://x/series/proof?season=0&chapter=3&key=k"))).text();
  assert.doesNotMatch(html, /data:image/);
  assert.equal((html.match(/<img src="\/series\/proof\/img\?/g) || []).length, 26); // 25 scenes + teaser
  const img = await sp.handleSeriesProofImage(env, new URL("https://x/series/proof/img?k=series/s00/c03/s01-office.png&key=k"));
  assert.equal(img.status, 200);
  assert.equal(img.headers.get("Content-Type"), "image/png");
  assert.equal((await sp.handleSeriesProofImage(env, new URL("https://x/series/proof/img?k=series/s00/c03/s01-office.png"))).status, 404);
  assert.equal((await sp.handleSeriesProofImage(env, new URL("https://x/series/proof/img?k=series/schedule.json&key=k"))).status, 404);
});

test("review: uploading does not replace approved chapters unless asked", () => {
  const info = { number: 0, show: "THE LONG DRIFT", title: "PROLOGUE", writer: "Claude", premiere: "2026-10-26" };
  const r = sched.mergeUpload(SCHEDULE, info, [1, 2, 3, 4], false);
  assert.deepEqual(r.upload, [3, 4]);
  assert.deepEqual(r.skipped, [1, 2]);
  const s0 = r.schedule.seasons.find((s) => s.number === 0);
  assert.deepEqual(s0.approved, [1, 2]);
  assert.deepEqual(s0.pending, [3, 4]);
  assert.equal(r.schedule.seasons.length, 2, "other seasons are kept");
  const all = sched.mergeUpload(SCHEDULE, info, [1, 2], true);
  assert.deepEqual(all.upload, [1, 2]);
  const fresh = sched.mergeUpload({ seasons: [] }, { ...info, number: 2, premiere: "2027-04-05" }, [1], false);
  assert.deepEqual(fresh.schedule.seasons[0], { ...info, number: 2, premiere: "2027-04-05", chapters: 13, approved: [], pending: [1] });
});

test("office conversion: a tinted grey field barely changes the area after it", () => {
  // The same cream area dithered on its own and right after a bluish near-grey field (which is
  // printed in black and white only). Carrying the grey's tint forward used to shift ~128 dots.
  const w = 200, h = 60;
  const field = (before) => {
    const rgb = new Uint8Array(w * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) rgb.set(x < 140 ? before : [235, 215, 170], (y * w + x) * 3);
    const idx = conv.toOfficeIndices(rgb, w, h);
    const counts = [0, 0, 0, 0, 0, 0];
    for (let y = 0; y < h; y++) for (let x = 144; x < w; x++) counts[idx[y * w + x]]++;
    return counts;
  };
  const after = field([60, 70, 110]), alone = field([235, 215, 170]);
  const shifted = after.reduce((sum, v, k) => sum + Math.abs(v - alone[k]), 0);
  assert.ok(shifted < 40, `${shifted} dots shifted in the cream area by the grey before it`);
});

// --- Mailbox watcher ---

const mb = fromBuild("src/series/mailbox.js");

test("mailbox: APPROVE, DELIVERY and STYLE TEST first lines are parsed", () => {
  assert.deepEqual(mb.parseMessage("APPROVE season=0 chapter=1\n\nLooks good."), { kind: "APPROVE", season: 0, chapters: [1] });
  assert.deepEqual(mb.parseMessage("\napprove season=0 chapters=1,2"), { kind: "APPROVE", season: 0, chapters: [1, 2] });
  assert.deepEqual(mb.parseMessage("DELIVERY package=season-01-chapter-01.zip\nnotes"), { kind: "DELIVERY", package: "season-01-chapter-01.zip", replaceApproved: false });
  assert.deepEqual(mb.parseMessage("DELIVERY package=s01.zip replace-approved"), { kind: "DELIVERY", package: "s01.zip", replaceApproved: true });
  assert.deepEqual(mb.parseMessage("STYLE TEST folder=openai-1"), { kind: "STYLE TEST", folder: "openai-1" });
});

test("mailbox: other kinds go to Claude, broken commands are explained", () => {
  assert.deepEqual(mb.parseMessage("QUESTION\nCan a balloon point off-panel?"), { kind: "QUESTION" });
  assert.deepEqual(mb.parseMessage("FYI\n# Notion ready"), { kind: "FYI" });
  assert.deepEqual(mb.parseMessage("Hello Claude"), { kind: "OTHER" });
  assert.match(mb.parseMessage("APPROVE chapter=1").reason, /season=/);
  assert.match(mb.parseMessage("DELIVERY\nzip is in packages").reason, /package=/);
  assert.match(mb.parseMessage("DELIVERY package=../../etc/passwd").reason, /file name/);
  assert.match(mb.parseMessage("STYLE TEST folder=a/b").reason, /folder name/);
});

test("mailbox: Claude's next chapter is due 19 days before it airs, one at a time", () => {
  const s = { seasons: [
    { number: 0, show: "X", title: "P", writer: "Claude", premiere: "2026-10-26", chapters: 10, approved: [1], pending: [2] },
    { number: 1, show: "X", title: "S1", writer: "OpenAI", premiere: "2027-01-04", chapters: 13, approved: [], pending: [] },
  ] };
  assert.deepEqual(mb.dueClaudeChapter(s, "2026-10-20"), null);           // ch3 airs 9 Nov: 20 days away
  assert.deepEqual(mb.dueClaudeChapter(s, "2026-10-21"), { season: 0, chapter: 3, airs: "2026-11-09" });
  const done = { seasons: [{ ...s.seasons[0], approved: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], pending: [] }] };
  assert.equal(mb.dueClaudeChapter(done, "2026-12-20"), null);            // season complete
});

// --- Style-test address ---

test("style-test pages live under series/tests/ and only take plain names", () => {
  assert.equal(keys.testKey("trial-zoe-1", "source", "office"), "series/tests/trial-zoe-1/source-office.png");
  assert.equal(keys.testKey("../x", "source", "desk"), null);
  assert.equal(keys.testKey("ok", "a/b", "desk"), null);
});

test("the test address shows a style test inline, only with the key", async () => {
  const objects = { "series/tests/trial-zoe-1/source-desk.png": PNG };
  const env = { SERIES: bucket(objects), SERIES_KEY: "k" };
  const ok = await sp.handleSeriesTest(env, new URL("https://x/desk/series/test?folder=trial-zoe-1&image=source&key=k"), "desk");
  assert.equal(ok.status, 200);
  assert.match(await ok.text(), /data:image\/png;base64,iVBORw/);
  assert.equal((await sp.handleSeriesTest(env, new URL("https://x/desk/series/test?folder=trial-zoe-1&image=source"), "desk")).status, 404);
  assert.equal((await sp.handleSeriesTest(env, new URL("https://x/desk/series/test?folder=../x&image=source&key=k"), "desk")).status, 404);
  const missing = await sp.handleSeriesTest(env, new URL("https://x/series/test?folder=trial-zoe-1&image=source&key=k"), "office");
  assert.equal(missing.status, 200);
  assert.doesNotMatch(await missing.text(), /data:image/);
});

test("the colour test address can send the original picture as-is (original=1)", async () => {
  assert.equal(keys.testOriginalKey("trial-zoe-1", "source"), "series/tests/trial-zoe-1/source-original.jpg");
  assert.equal(keys.testOriginalKey("../x", "source"), null);
  const JPG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3]);
  const env = { SERIES: bucket({ "series/tests/trial-zoe-1/source-original.jpg": JPG }), SERIES_KEY: "k" };
  const html = await (await sp.handleSeriesTest(env, new URL("https://x/color/series/test?folder=trial-zoe-1&image=source&original=1&key=k"), "office")).text();
  assert.match(html, /data:image\/jpeg;base64,\/9j\//);
});

test("the test address can also send the four-grey version (grey=1), for comparing on a real E1001", async () => {
  assert.equal(keys.testGreyKey("trial-zoe-1", "source"), "series/tests/trial-zoe-1/source-grey.png");
  assert.equal(keys.testGreyKey("a/b", "source"), null);
  const env = { SERIES: bucket({ "series/tests/trial-zoe-1/source-grey.png": PNG }), SERIES_KEY: "k" };
  const html = await (await sp.handleSeriesTest(env, new URL("https://x/desk/series/test?folder=trial-zoe-1&image=source&grey=1&key=k"), "desk")).text();
  assert.match(html, /data:image\/png;base64,iVBORw/);
});
