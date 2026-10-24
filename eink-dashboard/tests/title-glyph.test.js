const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const buildDir = process.env.EINK_TEST_BUILD_DIR || '/tmp/eink-dashboard-tests';
const {FACES} = require(path.join(buildDir, 'src/hud/fonts-data.js'));
const {measureType, drawType} = require(path.join(buildDir, 'src/hud/type.js'));
const {createCanvas, countBlack} = require(path.join(buildDir, 'src/hud/canvas.js'));

test('series title face includes a nonblank ASCII apostrophe from the real font', () => {
  const glyph = FACES.plex28r.glyphs["'"];
  assert.ok(glyph, 'plex28r must retain authored ASCII apostrophes');
  assert.ok(glyph.a > 0);
  assert.ok(Buffer.from(glyph.d, 'base64').some(byte => byte !== 0));
});

test('series title measurement includes the apostrophe advance and spacing', () => {
  const withMark = measureType('plex28r', "PILOT'S RETURN", 1);
  const withoutMark = measureType('plex28r', 'PILOTS RETURN', 1);
  assert.ok(withMark > withoutMark, 'an authored title character must not disappear from measurement');
  assert.equal(withMark - withoutMark, FACES.plex28r.glyphs["'"].a + 1);
});

test('series title apostrophe draws real black pixels at native size', () => {
  const canvas = createCanvas(40, 48);
  const width = drawType(canvas, 'plex28r', 4, 36, "'");
  assert.ok(countBlack(canvas) > 0, 'native title apostrophe must not render blank');
  assert.equal(width, FACES.plex28r.glyphs["'"].a);
});
