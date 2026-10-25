const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const test = require('node:test');

const script = path.resolve(__dirname, '../scripts/series-mailbox.mjs');
const date = '2026-10-25';
const picture = (kind, extra = {}) => ({ kind, file: `${date}-${kind}.png`, style: { name: 'test' }, prompt: `original ${kind}`, ...extra });

function fixture(t, previous, incoming, missingReference = false) {
  const root = fs.mkdtempSync(path.join(process.env.TMPDIR || os.tmpdir(), 'eink-art-retry-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const home = path.join(root, 'home');
  const mailbox = path.join(root, 'mailbox');
  const out = path.join(mailbox, 'from-claude');
  const art = path.join(out, 'art');
  const bin = path.join(root, 'bin');
  fs.mkdirSync(path.join(home, '.config/eink'), { recursive: true });
  fs.mkdirSync(art, { recursive: true });
  fs.mkdirSync(bin);
  fs.writeFileSync(path.join(home, '.config/eink/family-shown.json'), JSON.stringify([{ date }]));
  if (previous) fs.writeFileSync(path.join(art, `${date}-brief.json`), JSON.stringify(previous));
  const loader = path.join(root, 'fetch.cjs');
  fs.writeFileSync(loader, `globalThis.fetch = async () => new Response(${JSON.stringify(JSON.stringify(incoming))}, { status: 200, headers: { 'content-type': 'application/json' } });`);
  fs.writeFileSync(path.join(bin, 'npx'), `#!${process.execPath}\nconst fs=require('node:fs');const a=process.argv.slice(2);const i=a.indexOf('--file');if(i<0)process.exit(2);${missingReference ? '' : "fs.writeFileSync(a[i+1], Buffer.from([255,216,255,217]));"}\n`, { mode: 0o700 });
  const run = () => spawnSync(process.execPath, ['--require', loader, script, 'art-brief', date], {
    encoding: 'utf8', timeout: 15000,
    env: { ...process.env, HOME: home, SERIES_MAILBOX: mailbox, PATH: `${bin}:${process.env.PATH}`, ART_FORCE: '' },
  });
  const read = () => JSON.parse(fs.readFileSync(path.join(art, `${date}-brief.json`), 'utf8'));
  const notices = () => fs.readdirSync(out).filter(f => f.endsWith('.md') && f.includes('-art-brief-'));
  return { run, read, notices, art };
}

const initial = () => ({ date, deliverBy: 'test deadline', pictures: [picture('moment'), picture('skyline')] });

test('a late family photo augments an already complete brief without replacing its original art', t => {
  const old = initial();
  const fresh = { ...initial(), pictures: [picture('moment', { prompt: 'DO NOT REPLACE' }), picture('skyline'), picture('family', { references: 1 })] };
  const f = fixture(t, old, fresh);
  const r = f.run(); assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(f.read().pictures.map(p => p.kind), ['moment', 'skyline', 'family']);
  assert.deepEqual(f.read().pictures[0], old.pictures[0]);
  assert.deepEqual(f.read().pictures[2].referenceFiles, [`from-claude/art/${date}-family-ref.jpg`]);
  const once = fs.readFileSync(path.join(f.art, `${date}-brief.json`), 'utf8');
  const notices = f.notices();
  const again = f.run(); assert.equal(again.status, 0, again.stderr);
  assert.equal(fs.readFileSync(path.join(f.art, `${date}-brief.json`), 'utf8'), once, 'unchanged retry is a no-op');
  assert.deepEqual(f.notices(), notices, 'unchanged retry sends no duplicate brief');
});

test('family can arrive while the moment is still waiting', t => {
  const old = { ...initial(), pictures: [picture('skyline')], waiting: ['moment'] };
  const f = fixture(t, old, { ...old, pictures: [picture('skyline'), picture('family', { references: 1 })] });
  const r = f.run(); assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(f.read().pictures.map(p => p.kind), ['skyline', 'family']);
  assert.deepEqual(f.read().waiting, ['moment']);
});

test('a late moment resolves waiting without discarding a family picture already sent', t => {
  const family = picture('family', { references: 1, referenceFiles: [`from-claude/art/${date}-family-ref.jpg`] });
  const old = { ...initial(), pictures: [picture('skyline'), family], waiting: ['moment'] };
  const f = fixture(t, old, initial());
  const r = f.run(); assert.equal(r.status, 0, r.stderr);
  assert.deepEqual(f.read().pictures.map(p => p.kind), ['skyline', 'family', 'moment']);
  assert.equal(f.read().waiting, undefined);
  assert.deepEqual(f.read().pictures[1], family);
});

test('a missing family reference fails closed and leaves the previous brief retryable', t => {
  const old = initial();
  const f = fixture(t, old, { ...old, pictures: [...old.pictures, picture('family', { references: 1 })] }, true);
  const r = f.run();
  assert.notEqual(r.status, 0, 'missing reference must not be reported as successful publication');
  assert.deepEqual(f.read(), old);
  assert.equal(f.notices().length, 0);
});
