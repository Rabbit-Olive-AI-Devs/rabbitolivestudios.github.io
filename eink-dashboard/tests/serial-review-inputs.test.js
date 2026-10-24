const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const test = require('node:test');
const repo = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(repo, 'scripts/serial-mailbox.mjs'), 'utf8');
const start = source.indexOf('function review(beat, pagePng, picture) {');
const end = source.indexOf('\nfunction reply(', start);
assert(start >= 0 && end > start);
const body = source.slice(start, end);
const sha = b => crypto.createHash('sha256').update(b).digest('hex');
const beat = { id: 'SYNTHETIC-ONLY', text: ['A test cube.'], scene: { focus: { fx: 0.5, fy: 0.5 } } };
const digest = sha(JSON.stringify(beat, null, 2));
const healthy = stdout => ({ status: 0, signal: null, stdout, stderr: '' });
function run(result) {
  const logs = [];
  const fake = args => {
    assert.equal(args.write, undefined);
    assert.equal(args.web, undefined);
    assert.deepEqual(args.images, ['/synthetic/picture.png', '/synthetic/page.png']);
    assert(args.prompt.includes('beat.json'));
    return result;
  };
  const review = new Function('fs', 'os', 'path', 'crypto', 'REPO', 'runAgent', 'log', body + '\nreturn review;')(fs, os, path, crypto, repo, fake, m => logs.push(m));
  return { verdict: review(beat, '/synthetic/page.png', '/synthetic/picture.png'), logs };
}
test('actual missing-Read-tool false PASS is rejected', () => {
  const r = run(healthy('The painting and page show no visible faults; the subjects remain visible and the page text is readable. I couldn’t verify beat.json because the requested Read tool is unavailable.\n\nVERDICT: PASS'));
  assert.equal(r.verdict, null);
  assert(r.logs.every(x => !x.includes('The painting')));
});
test('legacy bare PASS without exact beat read is rejected', () => assert.equal(run(healthy('VERDICT: PASS')).verdict, null));
test('legacy bare REVISE without exact beat read is rejected', () => assert.equal(run(healthy('VERDICT: REVISE: synthetic issue')).verdict, null));
test('exact beat digest and final PASS are accepted', () => assert.deepEqual(run(healthy(`BEAT-SHA256: ${digest}\nVERDICT: PASS`)).verdict, { verdict: 'PASS', reason: '' }));
test('exact beat digest and final REVISE preserve reason', () => assert.deepEqual(run(healthy(`BEAT-SHA256: ${digest}\nVERDICT: REVISE: synthetic lettering issue`)).verdict, { verdict: 'REVISE', reason: 'synthetic lettering issue' }));
test('wrong beat digest is rejected', () => assert.equal(run(healthy(`BEAT-SHA256: ${'0'.repeat(64)}\nVERDICT: PASS`)).verdict, null));
test('duplicate digest lines are rejected', () => assert.equal(run(healthy(`BEAT-SHA256: ${digest}\nBEAT-SHA256: ${digest}\nVERDICT: PASS`)).verdict, null));
test('multiple verdict lines are rejected', () => assert.equal(run(healthy(`BEAT-SHA256: ${digest}\nVERDICT: REVISE: synthetic\nVERDICT: PASS`)).verdict, null));
test('text after final verdict is rejected', () => assert.equal(run(healthy(`BEAT-SHA256: ${digest}\nVERDICT: PASS\nI could not read the beat.`)).verdict, null));
for (const result of [{ status: 8 }, { signal: 'SIGTERM' }, { error: new Error('timeout') }]) {
  test('native failure never accepts a complete-looking response ' + JSON.stringify(result), () => assert.equal(run({ ...healthy(`BEAT-SHA256: ${digest}\nVERDICT: PASS`), ...result }).verdict, null));
}
test('review instructions allow bounded read-only command reads and digest calculation', () => {
  const prompt = fs.readFileSync(path.join(repo, 'scripts/series-agent/serial-review.md'), 'utf8');
  assert(!prompt.includes('Use only the Read tool.'));
  assert(prompt.includes('read-only'));
  assert(prompt.includes('BEAT-SHA256:'));
  assert(prompt.includes('REVIEW INCOMPLETE'));
});
