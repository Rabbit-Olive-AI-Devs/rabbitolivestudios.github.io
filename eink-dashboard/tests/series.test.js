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
  assert.equal(keys.teaserKey(1, 13, "home"), "series/s01/c13/teaser-home.png");
  assert.equal(keys.cardKey(0, "before", "desk"), "series/s00/before-desk.png");
  assert.equal(keys.slotKey({ kind: "between", next: 1, prev: 0 }, "home"), "series/s01/before-home.png");
  assert.equal(keys.slotKey({ kind: "between", next: null, prev: 0 }, "home"), "series/s00/after-home.png");
  assert.equal(keys.slotKey({ kind: "between", next: null, prev: null }, "home"), null);
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

test("desk conversion is pure black and white; home has four greys", () => {
  const gray = new Uint8Array(64 * 64).map((_, i) => i % 256);
  const bits = conv.toDeskBits(gray, 64, 64);
  assert.ok([...bits].every((b) => b === 0 || b === 1));
  const g4 = conv.toHomeGray(gray);
  assert.deepEqual([...new Set(g4)].sort((a, b) => a - b), [0, 85, 170, 255]);
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
