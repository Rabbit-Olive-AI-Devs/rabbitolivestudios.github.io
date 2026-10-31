const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const book = require(path.join(process.env.EINK_TEST_BUILD_DIR, 'src/series/book.js'));
const schedule = {seasons: [
  {number: 0, title: 'Prologue', show: 'The Long Drift', writer: 'Claude', premiere: '2026-10-26', chapters: 10, approved: [1, 2], pending: [3]},
  {number: 1, title: 'Future season', show: 'The Long Drift', writer: 'OpenAI', premiere: '2027-01-04', chapters: 13, approved: [1], pending: []}
]};
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAyAAAAHgCAIAAAA=', 'base64');
const objects = {'series/schedule.json': JSON.stringify(schedule), 'series/s00/c01/s01.html': `<html><img src="data:image/png;base64,${png.toString('base64')}"></html>`};
function env(data = objects) { return {SERIES: {get: async k => k in data ? {text: async () => data[k]} : null}}; }
const url = q => new URL('https://example.com/series/book' + q);
const now = new Date('2026-10-26T14:59:00Z'); // 09:59 Chicago.

test('contents lists only aired chapters and links to the scrolling chapter', async () => {
  const res = await book.handleSeriesBook(env(), url(''), now);
  const html = await res.text();
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('cache-control'), 'no-store');
  assert.match(html, /The Long Drift/);
  assert.match(html, /c=s00%2Fc01/);
  assert.doesNotMatch(html, /c02|c03|Future season|series\/s00/);
});
test('reader has exact aired-image link, contents and no future next link', async () => {
  const res = await book.handleSeriesBook(env(), url('?p=s00/c01/s01'), now);
  const html = await res.text();
  assert.match(html, /\/series\/book\/image\?p=s00%2Fc01%2Fs01/);
  assert.match(html, /Contents/);
  assert.doesNotMatch(html, /s02|rel="next"/);
});
test('next scene becomes readable at its exact Chicago airing time', async () => {
  const at = new Date('2026-10-26T15:00:00Z');
  const html = await (await book.handleSeriesBook(env(), url('?p=s00/c01/s01'), at)).text();
  assert.match(html, /rel="next"/);
  assert.match(html, /p=s00%2Fc01%2Fs02/);
});
test('future, pending, replay and traversal selectors never produce a reader or image', async () => {
  for (const p of ['s00/c01/s02','s00/c03/s01','s00/c01/r01','../schedule.json','s01/c01/s01']) {
    const u = url('?p=' + encodeURIComponent(p) + '&at=2030-01-01T20:00&key=anything');
    assert.equal((await book.handleSeriesBook(env(), u, now)).status, 404);
    assert.equal((await book.handleSeriesBookImage(env(), u, now)).status, 404);
  }
  assert.equal((await book.handleSeriesBookImage(env(), url('?p=s00/c03/s01'), new Date('2027-06-01T12:00:00Z'))).status, 404);
});
test('aired image is byte-identical to the stored panel PNG and never cached as public', async () => {
  const res = await book.handleSeriesBookImage(env(), url('?p=s00/c01/s01'), now);
  assert.equal(res.status, 200);
  assert.equal(res.headers.get('content-type'), 'image/png');
  assert.equal(res.headers.get('cache-control'), 'no-store');
  assert.deepEqual(Buffer.from(await res.arrayBuffer()), png);
});
test('unreadable schedule and missing image fail closed with useful text', async () => {
  const bad = env({'series/schedule.json': 'not json'});
  const res = await book.handleSeriesBook(bad, url(''), now);
  assert.equal(res.status, 503);
  assert.match(await res.text(), /unavailable/i);
  assert.equal((await book.handleSeriesBookImage(bad, url('?p=s00/c01/s01'), now)).status, 404);
  assert.equal((await book.handleSeriesBookImage(env({'series/schedule.json': JSON.stringify(schedule)}), url('?p=s00/c01/s01'), now)).status, 503);
});
test('chapter reader stacks every aired page in order with contents at both ends', async () => {
  const html = await (await book.handleSeriesBook(env(), url('?c=s00/c01'), new Date('2026-10-26T15:00:00Z'))).text();
  assert.equal((html.match(/<figure/g) || []).length, 2);
  assert.equal((html.match(/>Contents<\/a>/g) || []).length, 2);
  assert.ok(html.indexOf('p=s00%2Fc01%2Fs01') < html.indexOf('p=s00%2Fc01%2Fs02'));
  assert.doesNotMatch(html, /s03|Next chapter/);
  assert.match(html, /Open full-size page/);
});
test('chapter navigation is limited to chapters with actually aired pages', async () => {
  const at = new Date('2026-11-02T14:00:00Z');
  const first = await (await book.handleSeriesBook(env(), url('?c=s00/c01'), at)).text();
  const second = await (await book.handleSeriesBook(env(), url('?c=s00/c02'), at)).text();
  assert.match(first, /rel="next" href="\/series\/book\?c=s00%2Fc02"/);
  assert.match(second, /rel="prev" href="\/series\/book\?c=s00%2Fc01"/);
  assert.doesNotMatch(second, /c03|s01%2Fc01|Future season/);
  assert.equal((second.match(/<figure/g) || []).length, 1);
});
test('chapter selectors never unlock future, pending or arbitrary content', async () => {
  for (const c of ['s00/c02', 's00/c03', '../schedule.json', 's01/c01', 's00/c01/s02', '']) {
    assert.equal((await book.handleSeriesBook(env(), url('?c=' + encodeURIComponent(c) + '&at=2030-01-01&key=anything'), now)).status, 404);
  }
  assert.equal((await book.handleSeriesBook(env(), url('?c=s00/c03'), new Date('2027-06-01T12:00:00Z'))).status, 404);
});
test('before premiere no chapter or unaired season title appears', async () => {
  const html = await (await book.handleSeriesBook(env(), url(''), new Date('2026-10-26T12:59:00Z'))).text();
  assert.match(html, /No scenes have aired yet/);
  assert.doesNotMatch(html, /Prologue|Future season|p=s00/);
});
