const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");

const buildDir = process.env.EINK_TEST_BUILD_DIR || "/tmp/eink-dashboard-tests";
const fromBuild = (p) => require(path.join(buildDir, p));

const cv = fromBuild("src/hud/canvas.js");

const at = (c, x, y) => c.px[y * c.w + x];

test("canvas starts all white at panel size", () => {
  const c = cv.createCanvas();
  assert.equal(c.w, 800);
  assert.equal(c.h, 480);
  assert.equal(c.px.length, 800 * 480);
  assert.equal(cv.countBlack(c), 0);
});

test("fillRect paints exactly its area", () => {
  const c = cv.createCanvas(20, 10);
  cv.fillRect(c, 2, 3, 4, 5);
  assert.equal(cv.countBlack(c), 20);
  assert.equal(at(c, 2, 3), 0);
  assert.equal(at(c, 5, 7), 0);
  assert.equal(at(c, 6, 3), 1);
  assert.equal(at(c, 2, 8), 1);
});

test("drawing outside the canvas clips and never wraps to another row", () => {
  const c = cv.createCanvas(10, 10);
  cv.fillRect(c, -5, -5, 8, 8);          // only the 3x3 corner is inside
  assert.equal(cv.countBlack(c), 9);
  cv.fillRect(c, 8, 4, 50, 1);            // runs off the right edge
  assert.equal(at(c, 9, 4), 0);
  assert.equal(at(c, 0, 5), 1, "must not wrap onto the next row");
  assert.doesNotThrow(() => cv.line(c, -100, -100, 500, 500, 3));
});

test("strokeRect draws a hollow box of the given thickness", () => {
  const c = cv.createCanvas(20, 20);
  cv.strokeRect(c, 0, 0, 10, 10, 2);
  assert.equal(at(c, 0, 0), 0);
  assert.equal(at(c, 9, 9), 0);
  assert.equal(at(c, 1, 5), 0);
  assert.equal(at(c, 2, 5), 1, "inside stays white");
  assert.equal(cv.countBlack(c), 10 * 10 - 6 * 6);
});

test("invertRect flips black and white inside the area only", () => {
  const c = cv.createCanvas(10, 10);
  cv.fillRect(c, 0, 0, 5, 10);
  cv.invertRect(c, 0, 0, 10, 10);
  assert.equal(at(c, 0, 0), 1);
  assert.equal(at(c, 9, 9), 0);
  assert.equal(cv.countBlack(c), 50);
});

test("line joins its endpoints and thickness widens it", () => {
  const c = cv.createCanvas(20, 20);
  cv.line(c, 2, 2, 12, 2);
  assert.equal(cv.countBlack(c), 11);
  const d = cv.createCanvas(20, 20);
  cv.line(d, 2, 5, 12, 5, 3);
  assert.equal(at(d, 7, 4), 0);
  assert.equal(at(d, 7, 6), 0);
  const e = cv.createCanvas(20, 20);
  cv.line(e, 0, 0, 9, 9);
  for (let i = 0; i < 10; i++) assert.equal(at(e, i, i), 0);
});


// --- Real fonts as 1-bit glyph bitmaps (DECISIONS #67) -----------------------

const ty = fromBuild("src/hud/type.js");
const { FACES } = fromBuild("src/hud/fonts-data.js");

const inkBox = (c) => {
  let x0 = Infinity, x1 = -1, y0 = Infinity, y1 = -1;
  for (let y = 0; y < c.h; y++) for (let x = 0; x < c.w; x++) if (c.px[y * c.w + x] === 0) {
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  return { x0, x1, y0, y1 };
};

test("every generated face carries the characters the pages need", () => {
  const upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  for (const [key, face] of Object.entries(FACES)) {
    assert.ok(face.height > 0 && face.baseline > 0 && face.baseline <= face.height, key);
    for (const ch of upper) assert.ok(face.glyphs[ch], `${key} lacks ${ch}`);
    if (key.startsWith("plex")) for (const ch of "0123456789/.:- ") assert.ok(face.glyphs[ch], `${key} lacks "${ch}"`);
    for (const g of Object.values(face.glyphs)) {
      assert.equal(Buffer.from(g.d, "base64").length, Math.ceil((g.a * face.height) / 8), `${key} bitmap size`);
    }
  }
});

test("measureType adds advances and spacing, and ignores characters with no glyph", () => {
  const a = FACES.plex11m.glyphs.A.a;
  assert.equal(ty.measureType("plex11m", "AB"), a * 2, "Plex Mono is monospaced");
  assert.equal(ty.measureType("plex11m", "ABC", 2), a * 3 + 4);
  assert.equal(ty.measureType("plex11m", ""), 0);
  assert.equal(ty.measureType("af50", "Aé1"), FACES.af50.glyphs.A.a, "no glyph, no width");
  assert.ok(ty.measureType("af50", "I") < ty.measureType("af50", "A"), "Aurebesh AF is proportional");
});

test("drawType puts ink inside the text box, on the baseline it was given", () => {
  const c = cv.createCanvas(300, 80);
  const w = ty.drawType(c, "plex15s", 40, 50, "HELLO");
  assert.equal(w, ty.measureType("plex15s", "HELLO"));
  const b = inkBox(c);
  assert.ok(b.x0 >= 40 && b.x1 < 40 + w, `x ${b.x0}..${b.x1}`);
  assert.ok(b.y0 >= 50 - FACES.plex15s.baseline, "nothing above the cell");
  assert.ok(b.y1 <= 50, "capitals sit on the baseline, nothing below it");
  assert.ok(50 - b.y0 >= 8, "capitals are a real height, not a sliver");
});

test("drawType aligns right and centre from the same x", () => {
  const w = ty.measureType("plex15s", "HELLO");
  const r = cv.createCanvas(300, 40);
  ty.drawType(r, "plex15s", 200, 30, "HELLO", { align: "right" });
  assert.ok(inkBox(r).x1 < 200 && inkBox(r).x0 >= 200 - w);
  const m = cv.createCanvas(300, 40);
  ty.drawType(m, "plex15s", 150, 30, "HELLO", { align: "center" });
  const b = inkBox(m);
  assert.ok(Math.abs((b.x0 + b.x1) / 2 - 150) <= 3, "centred on x");
});

test("drawType clips at the canvas edge and skips unknown characters", () => {
  const c = cv.createCanvas(40, 20);
  assert.doesNotThrow(() => ty.drawType(c, "af112", -30, 15, "WWWW"));
  assert.doesNotThrow(() => ty.drawType(c, "plex11m", 5, 9999, "OFF"));
  const d = cv.createCanvas(100, 40);
  ty.drawType(d, "af50", 10, 35, "é\u{1F680}5");
  assert.equal(cv.countBlack(d), 0);
});

test("white type shows on a black block", () => {
  const c = cv.createCanvas(120, 30);
  cv.fillRect(c, 0, 0, 120, 30);
  ty.drawType(c, "plex12s", 10, 20, "041/120", { white: true });
  const white = c.px.length - cv.countBlack(c);
  assert.ok(white > 40, "letters are cut out of the block");
  assert.equal(at(c, 0, 0), 0, "the block itself stays black");
});

// --- Parts ---------------------------------------------------------------------

const parts = fromBuild("src/hud/parts.js");

test("brackets mark only the corners", () => {
  const c = cv.createCanvas(200, 200);
  parts.brackets(c, 20, 20, 100, 80, 10);
  assert.equal(at(c, 20, 20), 0);
  assert.equal(at(c, 119, 99), 0);
  assert.equal(at(c, 70, 20), 1, "the middle of an edge is open");
  assert.equal(at(c, 20, 60), 1);
});

test("hatch draws diagonal stripes inside its box only, however large it is asked to be", () => {
  const c = cv.createCanvas(60, 40);
  parts.hatch(c, 10, 10, 40, 20);
  const black = cv.countBlack(c);
  assert.ok(black > 0 && black < 40 * 20);
  assert.equal(at(c, 5, 15), 1);
  assert.equal(at(c, 55, 15), 1);
  const t0 = Date.now();
  parts.hatch(c, -1e6, -1e6, 4e6, 4e6);
  assert.ok(Date.now() - t0 < 200, "work is bounded by the canvas, not by the request");
});

test("dotGrid leaves the keep-out box empty and is bounded by the canvas", () => {
  const c = cv.createCanvas(200, 120);
  parts.dotGrid(c, 0, 0, 200, 120, 12, { x: 60, y: 30, w: 80, h: 60 });
  assert.ok(cv.countBlack(c) > 20);
  for (let y = 30; y < 90; y++) for (let x = 60; x < 140; x++) assert.equal(at(c, x, y), 1);
  const t0 = Date.now();
  parts.dotGrid(c, -1e7, -1e7, 3e7, 3e7, 12);
  assert.ok(Date.now() - t0 < 200);
});

test("ruler draws a baseline with ticks and refuses a zero step", () => {
  const c = cv.createCanvas(200, 40);
  parts.ruler(c, 10, 30, 100);
  assert.equal(at(c, 60, 30), 0);
  assert.equal(at(c, 10, 25), 0, "major tick rises above the line");
  assert.doesNotThrow(() => parts.ruler(c, 10, 30, 100, 0));
});

test("plus is a small cross centred on its point", () => {
  const c = cv.createCanvas(20, 20);
  parts.plus(c, 10, 10);
  assert.equal(cv.countBlack(c), 17);
  assert.equal(at(c, 10, 6), 0);
  assert.equal(at(c, 14, 10), 0);
});

// --- Page wrapper ---------------------------------------------------------------

const page = fromBuild("src/hud/page.js");
const card = fromBuild("src/hud/test-card.js");
const desk = fromBuild("src/pages/desk.js");

/** Width, height and bit depth from a PNG's IHDR chunk. */
function pngHeader(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { sig: [...bytes.slice(0, 4)], w: dv.getUint32(16), h: dv.getUint32(20), depth: bytes[24] };
}

const inlinedPng = (html) => {
  const m = html.match(/src="data:image\/png;base64,([^"]+)"/);
  assert.ok(m, "page must inline a PNG data URI");
  return new Uint8Array(Buffer.from(m[1], "base64"));
};

test("canvasToPng produces an 800x480 1-bit PNG", async () => {
  const h = pngHeader(await page.canvasToPng(cv.createCanvas()));
  assert.deepEqual(h.sig, [137, 80, 78, 71]);
  assert.deepEqual([h.w, h.h, h.depth], [800, 480, 1]);
});

test("hudPageResponse is one request: inline image, no script, no other URLs", async () => {
  const res = await page.hudPageResponse("Desk Test", () => cv.createCanvas());
  assert.equal(res.status, 200);
  assert.match(res.headers.get("Content-Type"), /^text\/html/);
  assert.equal(res.headers.get("Cache-Control"), "no-store");
  const html = await res.text();
  assert.equal(pngHeader(inlinedPng(html)).w, 800);
  assert.doesNotMatch(html, /<script/i);
  assert.doesNotMatch(html, /src="\//, "no second request to the Worker");
  assert.doesNotMatch(html, /https?:\/\//, "no external URL");
});

test("a draw function that throws becomes the text page, not an error", async () => {
  const res = await page.hudPageResponse("Desk Test", () => { throw new Error("boom"); });
  assert.equal(res.status, 200);
  const html = await res.text();
  assert.match(html, /Desk Test/);
  assert.doesNotMatch(html, /<img/i);
  assert.doesNotMatch(html, /boom/, "the error text is not shown on the panel");
});

const blackRatio = (c) => cv.countBlack(c) / (c.w * c.h);

test("the calibration card is a full panel with a moderate amount of ink", () => {
  const c = card.drawTestCard();
  assert.deepEqual([c.w, c.h], [800, 480]);
  assert.ok(blackRatio(c) > 0.02 && blackRatio(c) < 0.5, `black ratio ${blackRatio(c)}`);
});

test("/desk/test serves the card as an inlined page", async () => {
  const res = await desk.handleDeskTest();
  assert.equal(res.status, 200);
  assert.equal(pngHeader(inlinedPng(await res.text())).h, 480);
});

// --- Aurebesh --------------------------------------------------------------------

const words = fromBuild("src/hud/aurebesh-words.js");

test("every word is 3-9 capital letters and the list has no duplicates", () => {
  assert.ok(words.WORDS.length >= 100);
  for (const w of words.WORDS) assert.match(w, /^[A-Z]{3,9}$/, w);
  assert.equal(new Set(words.WORDS).size, words.WORDS.length);
});

test("dayOfYear handles the ends of the year and leap day", () => {
  assert.equal(words.dayOfYear("2026-01-01"), 1);
  assert.equal(words.dayOfYear("2026-12-31"), 365);
  assert.equal(words.dayOfYear("2028-02-29"), 60);
  assert.equal(words.dayOfYear("2028-12-31"), 366);
});

test("pickWord is stable per date, valid on day 366, and changes day to day", () => {
  assert.equal(words.pickWord("2026-10-12"), words.pickWord("2026-10-12"));
  assert.ok(words.WORDS.includes(words.pickWord("2028-12-31")));
  assert.notEqual(words.pickWord("2026-10-12"), words.pickWord("2026-10-13"));
});

test("parseDateParam accepts only real calendar dates", () => {
  assert.equal(words.parseDateParam("2026-10-12"), "2026-10-12");
  assert.equal(words.parseDateParam("2028-02-29"), "2028-02-29");
  for (const bad of [null, "", "yesterday", "2026-13-45", "2026-02-30", "2026-1-5", "20261012", "2026-10-12T00:00"]) {
    assert.equal(words.parseDateParam(bad), null, String(bad));
  }
});

test("cleanWord reduces any input to A-Z, capped at 9 letters", () => {
  assert.equal(words.cleanWord("hello"), "HELLO");
  assert.equal(words.cleanWord("R2-D2 droid!"), "RDDROID");
  assert.equal(words.cleanWord("supercalifragilistic"), "SUPERCALI");
  assert.equal(words.cleanWord("café \u{1F680}"), "CAF");
  for (const empty of [null, "", "   ", "1234", "\u{1F680}"]) assert.equal(words.cleanWord(empty), null);
});

test("every word belongs to a named group, and the groups match the list", () => {
  for (let i = 0; i < words.WORDS.length; i++) assert.match(words.groupOf(i), /^[A-Z]{3,9}$/, `index ${i}`);
  const g = (w) => words.groupOf(words.WORDS.indexOf(w));
  assert.equal(g("FORCE"), "CORE");
  assert.equal(g("TATOOINE"), "PLANETS");
  assert.equal(g("EXEGOL"), "PLANETS");
  assert.equal(g("HOPE"), "VIRTUES");
  assert.equal(g("MOTHER"), "KIN");
  assert.equal(g("TWILIGHT"), "TERRAIN");
  assert.equal(words.groupOf(-1), "CUSTOM", "a ?word= that is not in the list");
});

test("every letter has an Aurebesh name", () => {
  for (const ch of "ABCDEFGHIJKLMNOPQRSTUVWXYZ") assert.match(words.LETTER_NAMES[ch], /^[A-Z]{3,6}$/, ch);
  assert.equal(words.LETTER_NAMES.A, "AUREK");
  assert.equal(words.LETTER_NAMES.B, "BESH");
});

const ap = fromBuild("src/hud/aurebesh-page.js");
const aur = fromBuild("src/hud/aurebesh.js");

test("the word is set in the largest face that fits its box", () => {
  for (const w of ["I", "EWO", "JEDI", "MANDALORE", "WWWWWWWWW", "KASHYYYK"]) {
    const face = ap.wordFace(w);
    assert.match(face, /^af\d+$/);
    assert.ok(ty.measureType(face, w) <= 480, `${w} overflows in ${face}`);
  }
  assert.equal(ap.wordFace("JEDI"), "af112");
  assert.ok(FACES[ap.wordFace("MANDALORE")].size < FACES[ap.wordFace("JEDI")].size);
});

test("the word page stays inside the panel and the word stays inside its brackets", () => {
  for (const w of ["EWO", "MANDALORE", "WWWWWWWWW", "KASHYYYK"]) {
    const c = ap.drawAurebeshPage(w, "2026-10-12");
    assert.deepEqual([c.w, c.h], [800, 480]);
    assert.ok(blackRatio(c) > 0.03 && blackRatio(c) < 0.4, `${w}: black ratio ${blackRatio(c)}`);
    // the two bracket columns must be clear of word ink between the corner marks
    for (let y = 140; y <= 232; y++) for (const x of [238, 239, 240, 736, 737, 738]) {
      assert.equal(c.px[y * 800 + x], 1, `${w}: word touches the bracket at ${x},${y}`);
    }
  }
});

test("different words and different dates draw different pages", () => {
  const b64 = (c) => Buffer.from(c.px).toString("base64");
  assert.notEqual(b64(ap.drawAurebeshPage("JEDI", "2026-10-12")), b64(ap.drawAurebeshPage("REBEL", "2026-10-12")));
  assert.notEqual(b64(ap.drawAurebeshPage("JEDI", "2026-10-12")), b64(ap.drawAurebeshPage("JEDI", "2026-10-13")));
});

test("the check sheet shows all 26 letters on one panel", () => {
  const c = aur.drawAurebeshSheet();
  assert.deepEqual([c.w, c.h], [800, 480]);
  assert.ok(cv.countBlack(c) > 6000);
});

const pageBytes = async (qs) => {
  const res = await desk.handleDeskAurebesh(new URL(`https://x/desk/aurebesh${qs}`));
  assert.equal(res.status, 200);
  return Buffer.from(inlinedPng(await res.text())).toString("base64");
};

test("?word and ?date select the page; bad input falls back instead of failing", async () => {
  const jedi = await pageBytes("?word=jedi&date=2026-10-12");
  assert.equal(await pageBytes("?word=JEDI!!&date=2026-10-12"), jedi, "cleaned to the same word");
  assert.notEqual(await pageBytes("?word=rebel&date=2026-10-12"), jedi);
  const daily = await pageBytes("?date=2026-10-12");
  assert.equal(await pageBytes("?word=1234&date=2026-10-12"), daily, "unusable word -> daily word");
  assert.equal(await pageBytes("?word=&date=2026-10-12"), daily);
  assert.notEqual(await pageBytes("?date=2026-10-13"), daily);
  const today = await pageBytes("");
  assert.equal(await pageBytes("?date=2026-13-45"), today, "impossible date -> today");
  assert.equal(await pageBytes("?date=yesterday&word=%F0%9F%9A%80"), today);
  assert.notEqual(await pageBytes("?sheet"), daily, "?sheet shows the check sheet");
});

test("consecutive days do not walk the themed word list in order", () => {
  const n = words.WORDS.length;
  const idx = (d) => words.WORDS.indexOf(words.pickWord(d));
  const days = ["2026-10-12", "2026-10-13", "2026-10-14", "2026-10-15", "2026-12-31", "2027-01-01", "2027-01-02"];
  for (let i = 0; i < days.length - 1; i++) {
    const step = (idx(days[i + 1]) - idx(days[i]) + n) % n;
    assert.ok(step > 10 && step < n - 10, `${days[i]} -> ${days[i + 1]} moved ${step} places`);
  }
  const year = new Set();
  for (let d = 0; d < n; d++) year.add(words.pickWord(new Date(Date.UTC(2026, 0, 1 + d)).toISOString().slice(0, 10)));
  assert.equal(year.size, n, "every word still appears once per cycle");
});
