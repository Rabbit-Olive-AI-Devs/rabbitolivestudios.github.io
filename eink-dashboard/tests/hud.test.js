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
  assert.doesNotThrow(() => cv.circle(c, 5, 5, 400, 2));
  assert.doesNotThrow(() => cv.text(c, -20, 9999, "OUT", 3));
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

test("circle is symmetric and hollow", () => {
  const c = cv.createCanvas(41, 41);
  cv.circle(c, 20, 20, 10);
  assert.equal(at(c, 30, 20), 0);
  assert.equal(at(c, 10, 20), 0);
  assert.equal(at(c, 20, 30), 0);
  assert.equal(at(c, 20, 10), 0);
  assert.equal(at(c, 20, 20), 1);
});

test("text draws black pixels and textWidth matches the font metrics", () => {
  const c = cv.createCanvas(100, 20);
  cv.text(c, 0, 0, "A", 2);
  assert.ok(cv.countBlack(c) > 0);
  assert.equal(cv.textWidth("AB", 2), 2 * 16 + 2);
});
const st = fromBuild("src/hud/stroke.js");

test("parseGlyph splits polylines and points", () => {
  assert.deepEqual(st.parseGlyph("0,0 4,0|2,0 2,6"), [[[0, 0], [4, 0]], [[2, 0], [2, 6]]]);
  assert.deepEqual(st.parseGlyph("2,6"), [[[2, 6]]]);
});

test("LATIN covers A-Z and 0-9 with points inside its grid", () => {
  const need = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.:/%".split("");
  for (const ch of need) {
    const g = st.LATIN.glyphs[ch];
    assert.ok(g, `missing glyph ${ch}`);
    for (const poly of st.parseGlyph(g)) {
      for (const [x, y] of poly) {
        assert.ok(x >= 0 && x <= st.LATIN.w, `${ch} x=${x}`);
        assert.ok(y >= 0 && y <= st.LATIN.h, `${ch} y=${y}`);
      }
    }
  }
});

test("strokeTextWidth counts advances without a trailing gap", () => {
  // 3 glyphs of width 4 plus 2 gaps of 2 = 16 units
  assert.equal(st.strokeTextWidth(st.LATIN, "ABC", 5), 16 * 5);
  assert.equal(st.strokeTextWidth(st.LATIN, "", 5), 0);
});

test("drawStrokeText draws ink, skips unknown characters, and stays in its box", () => {
  const c = cv.createCanvas(200, 100);
  st.drawStrokeText(c, st.LATIN, 10, 10, "L", 10, 2);
  assert.ok(cv.countBlack(c) > 0);
  // L is "0,0 0,6 4,6": nothing above y=10 or left of x=10 (minus stamp offset 0 for t=2)
  for (let x = 0; x < 200; x++) assert.equal(c.px[5 * 200 + x], 1);
  const d = cv.createCanvas(200, 100);
  assert.doesNotThrow(() => st.drawStrokeText(d, st.LATIN, 10, 10, "é~", 10, 2));
  assert.equal(cv.countBlack(d), 0, "unknown characters draw nothing");
});

test("a space advances without drawing", () => {
  const c = cv.createCanvas(300, 100);
  st.drawStrokeText(c, st.LATIN, 0, 0, "I I", 10, 1);
  const cols = new Set();
  for (let y = 0; y < 100; y++) for (let x = 0; x < 300; x++) if (c.px[y * 300 + x] === 0) cols.add(x);
  assert.ok([...cols].some((x) => x < 50));
  assert.ok([...cols].some((x) => x >= 120), "second I sits two advances to the right");
});
const parts = fromBuild("src/hud/parts.js");
const style = fromBuild("src/hud/style.js");

test("frame draws a border inset by the margin and leaves the centre white", () => {
  const c = cv.createCanvas();
  parts.frame(c);
  const m = style.MARGIN;
  assert.equal(at(c, m, m), 0);
  assert.equal(at(c, 800 - m - 1, 480 - m - 1), 0);
  assert.equal(at(c, 400, m), 0);
  assert.equal(at(c, 0, 0), 1, "outside the margin stays white");
  assert.equal(at(c, 400, 240), 1);
});

test("brackets mark only the corners", () => {
  const c = cv.createCanvas(200, 200);
  parts.brackets(c, 20, 20, 100, 80, 10, 2);
  assert.equal(at(c, 20, 20), 0);
  assert.equal(at(c, 119, 99), 0);
  assert.equal(at(c, 70, 20), 1, "the middle of an edge is open");
  assert.equal(at(c, 20, 60), 1);
});

test("dotFill density follows the level and stays in its box", () => {
  const box = 40 * 40;
  const fill = (level) => {
    const c = cv.createCanvas(60, 60);
    parts.dotFill(c, 10, 10, 40, 40, level);
    assert.equal(at(c, 5, 5), 1);
    assert.equal(at(c, 55, 55), 1);
    return cv.countBlack(c) / box;
  };
  assert.equal(fill(0), 0);
  assert.equal(fill(1), 1);
  assert.ok(Math.abs(fill(0.5) - 0.5) < 0.07);
  assert.ok(fill(0.25) < fill(0.75));
});

test("hatch draws diagonal stripes inside its box only", () => {
  const c = cv.createCanvas(60, 40);
  parts.hatch(c, 10, 10, 40, 20, 6);
  const black = cv.countBlack(c);
  assert.ok(black > 0 && black < 40 * 20);
  assert.equal(at(c, 5, 15), 1);
  assert.equal(at(c, 55, 15), 1);
});

test("accentBlock is a solid block with white text and reports its width", () => {
  const c = cv.createCanvas(300, 60);
  const w = parts.accentBlock(c, 10, 10, "KEY", 2);
  assert.equal(w, cv.textWidth("KEY", 2) + 12);
  assert.equal(at(c, 10, 10), 0, "block corner is black");
  let white = 0;
  for (let y = 10; y < 10 + 24; y++) for (let x = 10; x < 10 + w; x++) if (at(c, x, y) === 1) white++;
  assert.ok(white > 0, "text shows as white inside the block");
});

test("ruler, crosshair, titleBar and bigNumber all draw without throwing", () => {
  const c = cv.createCanvas();
  parts.ruler(c, 40, 100, 300);
  parts.crosshair(c, 400, 240, 60);
  parts.titleBar(c, "LEFT / TITLE", "2026-10-12");
  parts.bigNumber(c, 40, 300, "14", "VSL");
  assert.ok(cv.countBlack(c) > 500);
});
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
  const png = await page.canvasToPng(cv.createCanvas());
  const h = pngHeader(png);
  assert.deepEqual(h.sig, [137, 80, 78, 71]);
  assert.equal(h.w, 800);
  assert.equal(h.h, 480);
  assert.equal(h.depth, 1);
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

test("the calibration card fills the panel with a moderate amount of ink", () => {
  const c = card.drawTestCard();
  assert.equal(c.w, 800);
  assert.equal(c.h, 480);
  const ratio = cv.countBlack(c) / (800 * 480);
  assert.ok(ratio > 0.02 && ratio < 0.5, `black ratio ${ratio}`);
});

test("/desk/test serves the card as an inlined page", async () => {
  const res = await desk.handleDeskTest();
  assert.equal(res.status, 200);
  assert.equal(pngHeader(inlinedPng(await res.text())).h, 480);
});
