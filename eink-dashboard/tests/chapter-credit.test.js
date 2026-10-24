const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const root = process.env.EINK_TEST_BUILD_DIR;
if (!root) throw new Error('isolated compiled build required');
const load = n => require(path.join(root, 'src/series', n));
const { samplePackage, validatePackage } = load('package.js');
const { buildChapter, buildCards } = load('build.js');
const { renderScene, renderTeaser } = load('render.js');
const artNames = p => new Set(p.chapters.flatMap(c => c.scenes.flatMap(s => s.panels.map(p => p.art))));
const image = { w: 800, h: 443, rgb: Uint8Array.from({ length: 800 * 443 * 3 }, (_, i) => (i * 17 + Math.floor(i / 2400)) % 256) };
function fixture() {
  const p = samplePackage();
  const c = p.chapters[0];
  c.title = 'CREDIT TEST'; c.teaser = 'Next week: another test.';
  c.scenes = [{ number: 1, template: 'splash', panels: [{ art: 'test.png' }] }];
  return p;
}
async function png(html) {
  const m = Buffer.from(html).toString('utf8').match(/data:image\/png;base64,([^"\s<>]+)/);
  assert.ok(m); return Buffer.from(m[1], 'base64');
}
test('explicit chapter writer credits actual live and replay pages without changing the season', async () => {
  const p = fixture(); p.chapters[0].writer = 'Jarvis';
  const before = JSON.stringify(p);
  const built = await buildChapter(p, 1, () => image);
  assert.deepEqual(built.problems, []);
  const expected = await renderScene({ ...p.season, writer: 'Jarvis' }, p.chapters[0], p.chapters[0].scenes[0], [{ image, anchors: [], clear: [] }]);
  assert.deepEqual(expected.problems, []);
  for (const variant of ['s', 'r']) {
    const actual = await png(built.files.get(`series/s00/c01/${variant}01.html`));
    assert.deepEqual(actual, Buffer.from(expected.pages[variant]));
  }
  assert.equal(JSON.stringify(p), before);
  assert.equal(p.season.writer, 'Claude');
});
test('chapter credit changes live and replay credits but preserves the existing creditless teaser', async () => {
  const p = fixture();
  const before = await buildChapter(p, 1, () => image);
  const expected = Buffer.from(await renderTeaser(p.season, p.chapters[0], false));
  assert.deepEqual(before.problems, []);
  assert.deepEqual(await png(before.files.get('series/s00/c01/teaser.html')), expected);
  for (const writer of ['Jarvis', 'Jarvis & Volt']) {
    p.chapters[0].writer = writer;
    const built = await buildChapter(p, 1, () => image);
    assert.deepEqual(built.problems, []);
    assert.deepEqual(built.files.get('series/s00/c01/teaser.html'), before.files.get('series/s00/c01/teaser.html'));
    assert.deepEqual(await png(built.files.get('series/s00/c01/teaser.html')), expected);
    for (const v of ['s', 'r']) assert.notDeepEqual(built.files.get(`series/s00/c01/${v}01.html`), before.files.get(`series/s00/c01/${v}01.html`));
  }
  p.chapters[0].teaser = 'A different test teaser.';
  const changed = await buildChapter(p, 1, () => image);
  assert.deepEqual(changed.problems, []);
  assert.notDeepEqual(changed.files.get('series/s00/c01/teaser.html'), before.files.get('series/s00/c01/teaser.html'));
});
test('omitted chapter writer retains exact legacy live and replay page bytes', async () => {
  const p = fixture();
  const built = await buildChapter(p, 1, () => image);
  const expected = await renderScene(p.season, p.chapters[0], p.chapters[0].scenes[0], [{ image, anchors: [], clear: [] }]);
  assert.deepEqual(built.problems, []);
  for (const v of ['s', 'r']) assert.deepEqual(await png(built.files.get(`series/s00/c01/${v}01.html`)), Buffer.from(expected.pages[v]));
});
test('chapter credit does not alter the season cards or original season object', async () => {
  const p = fixture(); const season = JSON.stringify(p.season);
  const before = await buildCards(p.season);
  p.chapters[0].writer = 'Jarvis';
  await buildChapter(p, 1, () => image);
  const after = await buildCards(p.season);
  assert.deepEqual(after, before);
  assert.equal(JSON.stringify(p.season), season);
});
test('valid explicit writer and omitted writer both pass package validation', () => {
  for (const writer of [undefined, 'Jarvis', 'Jarvis & Volt']) {
    const p = samplePackage();
    if (writer !== undefined) p.chapters[0].writer = writer;
    assert.deepEqual(validatePackage(p, artNames(p), []).problems, []);
  }
});
for (const [label, value] of Object.entries({ empty: '', blank: '   ', null: null, number: 3, boolean: true, list: ['Jarvis'], object: {}, nonascii: 'Jarvís', newline: 'Jarvis\n', banned: 'Han Solo' })) {
  test(`invalid chapter writer is refused: ${label}`, () => {
    const p = samplePackage(); p.chapters[0].writer = value;
    const r = validatePackage(p, artNames(p), []);
    assert.equal(r.pkg, null);
    assert.ok(r.problems.some(x => /^chapter 1 writer:/.test(x)), 'must name the chapter credit problem');
  });
}
