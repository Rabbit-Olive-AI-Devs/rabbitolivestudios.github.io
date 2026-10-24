const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const test = require('node:test');

// Execute unchanged copies of the real consumers, runner, compiler and page builder.
// Only native AI and Cloudflare transports are fakes. No native/visual acceptance is claimed.
const repo = path.resolve(__dirname, '..');
const evidence = process.env.EINK_CONSUMER_EVIDENCE_DIR ?? path.join(os.tmpdir(), 'eink-consumer-contract');
fs.mkdirSync(evidence, { recursive: true });
const runRoot = fs.mkdtempSync(path.join(evidence, 'run-'));
const hash = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const write = (p, body) => { fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, body); };
const json = (p, value) => write(p, JSON.stringify(value));
const readJSON = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const scripts = ['serial-mailbox.mjs', 'series-mailbox.mjs', 'agent-run.mjs', 'eink-paths.mjs', 'serial-build.mjs', 'family-selection.mjs'];

function fixture(label, mode, reviewer = 'volt') {
  const root = fs.mkdtempSync(path.join(runRoot, label + '-'));
  const stage = path.join(root, 'stage'), home = path.join(root, 'home'), tmp = path.join(root, 'tmp'), bin = path.join(root, 'bin');
  const dropbox = path.join(root, 'FAKE-DROPBOX-NOT-WATCHED');
  const mailbox = path.join(dropbox, '00_SHARED_INBOX/series');
  const inbox = path.join(mailbox, 'to-claude'), outbox = path.join(mailbox, 'from-claude');
  const work = path.join(home, 'desk-serial-work'), seriesWork = path.join(home, 'series-work');
  const proofs = path.join(root, 'FAKE-PROOFS'), events = path.join(root, 'events.jsonl');
  for (const p of [stage, home, tmp, bin, inbox, outbox, work, seriesWork, proofs]) fs.mkdirSync(p, { recursive: true });
  for (const name of scripts) write(path.join(stage, 'scripts', name), fs.readFileSync(path.join(repo, 'scripts', name)));
  for (const name of ['src', 'scripts/series-agent', 'docs/series']) fs.cpSync(path.join(repo, name), path.join(stage, name), { recursive: true });
  for (const name of ['tsconfig.json', 'tsconfig.test.json', 'package.json']) fs.copyFileSync(path.join(repo, name), path.join(stage, name));
  fs.symlinkSync(path.join(repo, 'node_modules'), path.join(stage, 'node_modules'), 'dir');
  fs.symlinkSync(process.execPath, path.join(bin, 'node'));
  const native = path.join(bin, 'fake-native');
  write(native, `#!${process.execPath}
const fs = require('node:fs'), path = require('node:path');
const args = process.argv.slice(2), mode = process.env.FAKE_MODE;
const prompt = fs.readFileSync(0, 'utf8');
fs.appendFileSync(process.env.FAKE_EVENTS, JSON.stringify({ type: 'fake-native', mode, images: args.filter(a => a === '-i').length }) + '\\n');
const answer = args[args.indexOf('-o') + 1];
if (mode.startsWith('serial-')) {
  const beatFile = /^- The beat .*: (.+)$/m.exec(prompt)[1];
  const digest = require('node:crypto').createHash('sha256').update(fs.readFileSync(beatFile)).digest('hex');
  const bound = mode === 'serial-missing-digest' ? '' : 'BEAT-SHA256: ' + digest + '\\n';
  fs.writeFileSync(answer, bound + 'VERDICT: ' + process.env.FAKE_VERDICT + (process.env.FAKE_VERDICT === 'REVISE' ? ': synthetic transport fixture only' : ''));
} else if (mode === 'chapter-generic') {
  const target = /Write exactly one file, (.+), and nothing else/.exec(prompt)[1];
  fs.writeFileSync(target, 'PASS\\nSYNTHETIC GENERIC REVIEW ONLY; not Volt.\\n');
} else if (mode !== 'message-missing') {
  const source = path.basename(/The message is in this file: (.+)/.exec(prompt)[1]);
  const outbox = process.env.FAKE_OUTBOX;
  const stamp = /named (\\d{4}-\\d{2}-\\d{2}-\\d{4})-reply-/.exec(prompt)[1];
  const name = mode === 'message-stale-update' ? '2000-01-01-0000-reply-old.md' : stamp + '-reply-fixture.md';
  const bound = mode === 'message-wrong' ? source + '.wrong' : source;
  const header = 'FYI: reply to ' + bound;
  const ending = mode === 'message-crlf' ? '\\r\\n' : '\\n';
  const body = mode === 'message-header-only' ? '' : ending + 'Synthetic reply; no production action.' + ending;
  fs.writeFileSync(path.join(outbox, name), header + ending + body);
}
if (mode.includes('nonzero')) process.exit(9);
if (mode.includes('signal')) process.kill(process.pid, 'SIGTERM');
if (mode.includes('timeout')) setInterval(() => {}, 1000);
`);
  fs.chmodSync(native, 0o700);
  // The real native spawn is exercised. This transport-only adapter shortens the
  // timeout to 150 ms so a written final answer followed by a hang is practical to test.
  const preload = path.join(root, 'native-deadline.cjs');
  write(preload, `const cp = require('node:child_process');
const original = cp.spawnSync;
cp.spawnSync = function(command, args, options) {
  if (command === process.env.FAKE_NATIVE && process.env.FAKE_MODE.includes('timeout')) {
    options = { ...options, timeout: 150 };
  }
  return original.call(this, command, args, options);
};
require('node:module').syncBuiltinESMExports();
`);
  write(path.join(bin, 'npx'), `#!${process.execPath}
const fs = require('node:fs'), cp = require('node:child_process');
const args = process.argv.slice(2);
if (args[0] === 'tsc') {
  const r = cp.spawnSync(process.execPath, [process.env.FAKE_TSC, ...args.slice(1)], { stdio: 'inherit' });
  process.exit(r.status ?? 92);
}
if (args.slice(0, 4).join(' ') === 'wrangler r2 object put') {
  const file = args[args.indexOf('--file') + 1];
  fs.appendFileSync(process.env.FAKE_EVENTS, JSON.stringify({ type: 'fake-publish', key: args[4], bytes: fs.statSync(file).size }) + '\\n');
  process.exit(0);
}
throw new Error('Unsupported external command refused: ' + args.join(' '));
`);
  fs.chmodSync(path.join(bin, 'npx'), 0o700);
  json(path.join(home, '.config/eink/paths.json'), { dropbox, photosLibrary: path.join(root, 'NO-PHOTOS'), familyTmp: path.join(tmp, 'family') });
  json(path.join(home, '.config/eink/agent.json'), { agent: 'codex', command: native, model: 'gpt-6.1-sol', seriesReviewer: reviewer });
  // Use an already-installed system toolchain inside the unchanged sandbox.
  // Do not read the unrelated external Xcode tree or loosen any deny rule.
  const env = { PATH: bin + ':/usr/bin:/bin', HOME: home, TMPDIR: tmp, LANG: 'en_US.UTF-8',
    ...(process.platform === 'darwin' && fs.existsSync('/Library/Developer/CommandLineTools/usr/bin/git')
      ? { DEVELOPER_DIR: '/Library/Developer/CommandLineTools' } : {}),
    SERIES_MAILBOX: mailbox, SERIES_PROOFS: proofs, SERIAL_WORK: work,
    FAKE_NATIVE: native, FAKE_MODE: mode, FAKE_EVENTS: events, FAKE_OUTBOX: outbox,
    FAKE_TSC: path.join(repo, 'node_modules/typescript/bin/tsc'), NODE_OPTIONS: '--require=' + preload };
  // No inherited credentials/config/provider overrides. On macOS also forbid network,
  // writes outside this fixture, and reads of production user data.
  let sandbox = [];
  if (process.platform === 'darwin') {
    const q = (p) => JSON.stringify(fs.realpathSync(p));
    const profile = path.join(root, 'confinement.sb');
    write(profile, `(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n(allow file-write* (literal "/dev/null") (subpath ${q(root)}))\n(deny file-read* (subpath "/Users") (subpath "/Volumes"))\n(allow file-read-metadata)\n(allow file-read* (subpath ${q(root)}) (subpath ${q(path.join(repo, 'node_modules'))}))\n`);
    sandbox = ['/usr/bin/sandbox-exec', '-f', profile];
  }
  let invocation = 0;
  const f = { root, stage, home, tmp, mailbox, inbox, outbox, work, seriesWork, proofs, env,
    events: () => fs.existsSync(events) ? fs.readFileSync(events, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse) : [],
    run(script, ...args) {
      const command = [...sandbox, process.execPath, path.join(stage, 'scripts', script), ...args];
      const r = spawnSync(command[0], command.slice(1), { cwd: stage, env, encoding: 'utf8', timeout: 45_000 });
      const label = String(++invocation);
      write(path.join(root, label + '.stdout'), r.stdout ?? '');
      write(path.join(root, label + '.stderr'), r.stderr ?? '');
      assert.equal(r.error, undefined, 'consumer process must finish within fixture bound');
      assert.equal(r.status, 0, 'consumer command failed: ' + r.stderr);
      return r;
    },
  };
  json(path.join(root, 'copied-source-hashes.json'), Object.fromEntries(scripts.map(n => [n, hash(path.join(stage, 'scripts', n))])));
  return f;
}

// A real nonblank 800x315 PNG made with the production PNG encoder, not a fake builder.
let chart;
async function serialFixture(mode, verdict) {
  const f = fixture(mode + '-' + verdict, mode);
  f.env.FAKE_VERDICT = verdict;
  const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Chicago', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
  f.id = date + '-1'; f.date = date;
  const beat = { id: f.id, start: '23:59', title: 'TEST ONLY', text: ['A synthetic cube waits.'], log: 'Fixture cube', scene: { focus: { fx: 0.5, fy: 0.5 } } };
  json(path.join(f.work, 's01/days', date + '.json'), { date, day: 1, beats: [beat] });
  json(path.join(f.work, 's01/settings.json'), { settings: [{ id: 'fixture-setting', focus: { fx: 0.5, fy: 0.5 } }] });
  if (!chart) {
    // Compile through the same refused-by-default npx shim as the consumer.
    const out = path.join(f.tmp, 'chart-compiler');
    const r = spawnSync(process.execPath, [path.join(repo, 'node_modules/typescript/bin/tsc'), '-p', path.join(f.stage, 'tsconfig.test.json'), '--outDir', out], { env: f.env, encoding: 'utf8', timeout: 45_000 });
    assert.equal(r.status, 0, r.stdout + r.stderr);
    const png = require(path.join(out, 'src/png.js'));
    const rgb = new Uint8Array(800 * 315 * 3);
    for (let y = 0; y < 315; y++) for (let x = 0; x < 800; x++) rgb.fill((Math.floor(x / 50) + Math.floor(y / 45)) % 2 ? 220 : 30, (y * 800 + x) * 3, (y * 800 + x) * 3 + 3);
    chart = Buffer.from(await png.encodePNGRGB(rgb, 800, 315));
  }
  write(path.join(f.inbox, 'serial', date, f.id + '.png'), chart);
  write(path.join(f.inbox, 'serial/settings/fixture-setting.png'), chart);
  return f;
}
const nativeCalls = (f) => f.events().filter(e => e.type === 'fake-native');
const publications = (f) => f.events().filter(e => e.type === 'fake-publish');

for (const mode of ['serial-nonzero', 'serial-signal', 'serial-timeout']) {
  for (const verdict of ['PASS', 'REVISE']) {
    test(`${mode}: ${verdict} from failed native run leaves beat and setting UNREVIEWED`, async () => {
      const f = await serialFixture(mode, verdict);
      f.run('serial-mailbox.mjs', 'run');
      const state = readJSON(path.join(f.work, 'state.json'));
      for (const key of [f.id, 'set:fixture-setting']) {
        assert.equal(state[key].verdict, 'UNREVIEWED');
        assert.equal(state[key].tries, 1);
      }
      assert.equal(nativeCalls(f).length, 2);
      assert.ok(nativeCalls(f).every(e => e.images === 2));
      assert.ok(publications(f).every(e => e.key.endsWith('.json')), 'failed reviews cannot publish pages');
      assert.deepEqual(fs.readdirSync(f.outbox), [], 'no receipt or REVISE fabricated from failed review');
    });
  }
}
for (const verdict of ['PASS', 'REVISE']) {
  test(`successful native ${verdict} without exact beat digest cannot publish`, async () => {
    const f = await serialFixture('serial-missing-digest', verdict);
    f.run('serial-mailbox.mjs', 'run');
    const state = readJSON(path.join(f.work, 'state.json'));
    assert.equal(state[f.id].verdict, 'UNREVIEWED');
    assert.equal(state['set:fixture-setting'].verdict, 'UNREVIEWED');
    assert.equal(nativeCalls(f).length, 2);
    assert.equal(publications(f).filter(e => e.key.endsWith('.html')).length, 0);
    assert.deepEqual(fs.readdirSync(f.outbox), []);
  });
}
for (const verdict of ['PASS', 'REVISE']) {
  test(`successful serial ${verdict} remains accepted and idempotent`, async () => {
    const f = await serialFixture('serial-success', verdict);
    f.run('serial-mailbox.mjs', 'run');
    const state = readJSON(path.join(f.work, 'state.json'));
    assert.equal(state[f.id].verdict, verdict);
    assert.equal(state['set:fixture-setting'].verdict, verdict);
    assert.equal(publications(f).filter(e => e.key.endsWith('.html')).length, verdict === 'PASS' ? 1 : 0);
    f.run('serial-mailbox.mjs', 'run');
    assert.equal(nativeCalls(f).length, 2, 'unchanged accepted reviews must not run twice');
  });
}
test('failed serial native PASS retains three attempts, escalation and five-minute retry', async () => {
  const f = await serialFixture('serial-nonzero', 'PASS');
  for (let i = 1; i <= 3; i++) {
    f.run('serial-mailbox.mjs', 'run');
    assert.equal(readJSON(path.join(f.work, 'state.json'))[f.id].tries, i);
  }
  assert.match(fs.readFileSync(path.join(f.outbox, 'NEEDS-THIAGO.md'), 'utf8'), /review could not run/);
  f.run('serial-mailbox.mjs', 'run');
  assert.equal(nativeCalls(f).length, 6, 'retry held before five minutes');
  const state = readJSON(path.join(f.work, 'state.json'));
  state[f.id].at = state['set:fixture-setting'].at = Date.now() - 301_000;
  json(path.join(f.work, 'state.json'), state);
  f.run('serial-mailbox.mjs', 'run');
  assert.equal(nativeCalls(f).length, 8, 'both pending reviews retried after five minutes');
  assert.equal(readJSON(path.join(f.work, 'state.json'))[f.id].tries, 4);
});

function messageFixture(mode, kind = 'QUESTION') {
  const f = fixture(mode, mode);
  f.source = 'fixture-question.md';
  write(path.join(f.inbox, f.source), kind + '\nSynthetic message; no production action.\n');
  const old = new Date(Date.now() - 120_000);
  fs.utimesSync(path.join(f.inbox, f.source), old, old);
  if (mode.startsWith('message-stale')) {
    // Seed both minute names so this stale-name control is stable across a clock boundary.
    for (const offset of mode === 'message-stale-current' ? [0, 60_000] : [0]) {
      const stamp = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Chicago', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).formatToParts(new Date(Date.now() + offset));
      const parts = Object.fromEntries(stamp.map(p => [p.type, p.value]));
      const name = mode === 'message-stale-current' ? `${parts.year}-${parts.month}-${parts.day}-${parts.hour}${parts.minute}-reply-fixture.md` : '2000-01-01-0000-reply-old.md';
      write(path.join(f.outbox, name), 'FYI: reply to ' + f.source + '\n\nStale fixture reply.\n');
    }
  }
  return f;
}
for (const mode of ['message-missing', 'message-wrong', 'message-stale', 'message-stale-update', 'message-stale-current', 'message-header-only', 'message-nonzero', 'message-signal', 'message-timeout']) {
  test(`${mode}: no valid successful newly created reply retains retry and escalation`, () => {
    const f = messageFixture(mode);
    // message-stale means native success without creating any new reply.
    if (mode === 'message-stale') f.env.FAKE_MODE = 'message-missing';
    for (let attempt = 1; attempt <= 2; attempt++) {
      f.run('series-mailbox.mjs', 'run');
      assert.ok(fs.existsSync(path.join(f.inbox, f.source)), 'failed message must stay pending');
      assert.equal(readJSON(path.join(f.seriesWork, '.mailbox-attempts.json'))[f.source], attempt);
      assert.ok(!fs.existsSync(path.join(f.outbox, 'NEEDS-THIAGO.md')));
    }
    f.run('series-mailbox.mjs', 'run');
    assert.ok(fs.existsSync(path.join(f.inbox, 'done', f.source)), 'existing third-failure archive semantics preserved');
    assert.deepEqual(readJSON(path.join(f.seriesWork, '.mailbox-attempts.json')), {});
    assert.match(fs.readFileSync(path.join(f.outbox, 'NEEDS-THIAGO.md'), 'utf8'), /needs a .*session/);
    assert.ok(fs.readdirSync(f.outbox).some(n => n.endsWith('-could-not-handle.md')));
    assert.equal(nativeCalls(f).length, 3);
    const log = fs.readFileSync(path.join(f.home, 'Library/Logs/eink-series/mailbox.log'), 'utf8');
    assert.match(log, /gave up/);
    assert.doesNotMatch(log, /fixture-question\.md: handled/);
    f.run('series-mailbox.mjs', 'run');
    assert.equal(nativeCalls(f).length, 3, 'archived escalation does not rerun agent');
  });
}
for (const kind of ['QUESTION', 'FYI', 'Unrecognised message']) {
  test(`${kind}: actual exact-bound newly created reply succeeds once`, () => {
    const f = messageFixture('message-good', kind);
    f.run('series-mailbox.mjs', 'run');
    assert.ok(fs.existsSync(path.join(f.inbox, 'done', f.source)));
    const replies = fs.readdirSync(f.outbox);
    assert.equal(replies.length, 1, 'consumer does not fabricate a receipt');
    assert.equal(fs.readFileSync(path.join(f.outbox, replies[0]), 'utf8').split('\n')[0], 'FYI: reply to ' + f.source);
    assert.deepEqual(readJSON(path.join(f.seriesWork, '.mailbox-attempts.json')), {});
    f.run('series-mailbox.mjs', 'run');
    assert.equal(nativeCalls(f).length, 1);
  });
}
test('CRLF exact-source reply is accepted', () => {
  const f = messageFixture('message-crlf');
  f.run('series-mailbox.mjs', 'run');
  assert.ok(fs.existsSync(path.join(f.inbox, 'done', f.source)));
  assert.ok(!fs.existsSync(path.join(f.outbox, 'NEEDS-THIAGO.md')));
});
test('a missing reply can recover on the next attempt with a genuine new reply', () => {
  const f = messageFixture('message-missing');
  f.run('series-mailbox.mjs', 'run');
  assert.ok(fs.existsSync(path.join(f.inbox, f.source)));
  f.env.FAKE_MODE = 'message-good';
  f.run('series-mailbox.mjs', 'run');
  assert.ok(fs.existsSync(path.join(f.inbox, 'done', f.source)));
  assert.deepEqual(readJSON(path.join(f.seriesWork, '.mailbox-attempts.json')), {});
  assert.ok(!fs.existsSync(path.join(f.outbox, 'NEEDS-THIAGO.md')));
});

function chapterFixture(reviewer, season = 4) {
  const f = fixture('chapter-selector', 'chapter-generic', reviewer);
  f.revision = '0123456789abcdef';
  f.base = `s${String(season).padStart(2, '0')}-c01-${f.revision}`;
  f.exportDir = path.join(f.proofs, `s${String(season).padStart(2, '0')}-c01`, 'rev-' + f.revision);
  write(path.join(path.dirname(f.exportDir), 'CURRENT.txt'), f.revision + '\n');
  json(path.join(f.exportDir, 'MANIFEST.json'), { fixture: true, revision: f.revision });
  write(path.join(f.exportDir, 'READY'), 'SYNTHETIC FIXTURE ONLY\n');
  const pkg = path.join(f.root, 'synthetic-package');
  json(path.join(pkg, 'season.json'), { fixture: true, writer: 'Jarvis' });
  f.zip = path.join(f.root, 'synthetic-package.zip');
  const r = spawnSync('/usr/bin/ditto', ['-c', '-k', pkg, f.zip], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  f.review = () => f.run('series-mailbox.mjs', 'review', String(season), '1', f.zip);
  return f;
}
for (const [i, reviewer] of [undefined, null, '', 'vlot', 'Volt', 'agent'].entries()) {
  test(`independent Jarvis review selector ${String(reviewer)} fails closed without generic review`, () => {
    const f = chapterFixture(reviewer, i % 2 ? 0 : 4);
    // Explicitly omit instead of triggering fixture's default parameter.
    if (reviewer === undefined) {
      const config = path.join(f.home, '.config/eink/agent.json');
      const value = readJSON(config); delete value.seriesReviewer; json(config, value);
    }
    const result = f.review();
    assert.equal(nativeCalls(f).length, 0, 'missing/invalid selector must never call generic/self reviewer');
    assert.match(result.stdout, /review did not finish/);
    assert.ok(!fs.existsSync(path.join(f.seriesWork, 'reviews', f.base + '.json')));
    assert.ok(!fs.existsSync(path.join(f.exportDir, 'INDEPENDENT-REVIEW.md')));
    assert.ok(!fs.existsSync(path.join(f.mailbox, 'writer-exchange')), 'no invented Volt dispatch');
  });
}
test('invalid review selection preserves existing exact-revision evidence', () => {
  const f = chapterFixture('vlot');
  const verdict = path.join(f.seriesWork, 'reviews', f.base + '.md');
  const record = path.join(f.seriesWork, 'reviews', f.base + '.json');
  write(verdict, 'REVISE\nHistorical exact-revision evidence; preserve bytes.\n');
  json(record, { fixture: true, reviewer: 'historical', revision: f.revision, verdict: 'REVISE' });
  const before = [hash(verdict), hash(record)];
  f.review();
  assert.deepEqual([hash(verdict), hash(record)], before);
  assert.equal(nativeCalls(f).length, 0);
});
test('exact volt selector dispatches real exchange request without inventing a verdict', () => {
  const f = chapterFixture('volt');
  const result = f.review();
  assert.match(result.stdout, /SENT TO VOLT/);
  assert.equal(nativeCalls(f).length, 0);
  const request = path.join(f.mailbox, 'writer-exchange/s02-volt/to-writer', 'LD-REVIEW-' + f.base);
  assert.ok(fs.existsSync(path.join(request, 'REVIEW-REQUEST.md')));
  assert.equal(hash(path.join(request, 'proof/MANIFEST.json')), hash(path.join(f.exportDir, 'MANIFEST.json')));
  assert.ok(fs.existsSync(path.join(request, 'package/season.json')));
  const pending = path.join(f.seriesWork, 'reviews', f.base + '.volt.json');
  assert.equal(readJSON(pending).revision, f.revision);
  assert.ok(!fs.existsSync(path.join(f.seriesWork, 'reviews', f.base + '.json')));
  assert.ok(!fs.existsSync(path.join(f.exportDir, 'INDEPENDENT-REVIEW.md')));
  // No synthetic from-writer answer is ever created. Collection remains pending.
  f.run('series-mailbox.mjs', 'run');
  assert.ok(fs.existsSync(pending));
  assert.ok(!fs.existsSync(path.join(f.mailbox, 'writer-exchange/s02-volt/from-writer')));
  assert.deepEqual(fs.readdirSync(f.outbox), []);
});
