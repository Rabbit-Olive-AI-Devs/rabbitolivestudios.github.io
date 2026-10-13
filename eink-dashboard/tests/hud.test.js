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
