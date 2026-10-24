const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const test = require('node:test');

// Real copied CLI, runner, prompt, checker and compiler; only native AI/publish
// transports are fake. These are synthetic contracts, NOT native acceptance.
const repo = path.resolve(__dirname, '..');
const evidence = process.env.EINK_WEEKLY_EVIDENCE_DIR ?? path.join(os.tmpdir(), 'eink-weekly-contract');
fs.mkdirSync(evidence, { recursive: true });
const runRoot = fs.mkdtempSync(path.join(evidence, 'run-'));
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const write = (p, body) => { fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, body); };
const json = (p, value) => write(p, JSON.stringify(value));
const shift = (date, n) => {
  const d = new Date(date + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
};
const todayChicago = () => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Chicago', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
// Independent calendar construction: first Monday at least 12 days away, then Sunday.
const horizon = today => {
  let monday = shift(today, 12);
  while (new Date(monday + 'T12:00:00Z').getUTCDay() !== 1) monday = shift(monday, 1);
  return shift(monday, 6);
};
function day(date) {
  return { date, day: Math.round((Date.parse(date + 'T12:00:00Z') - Date.parse('2026-10-17T12:00:00Z')) / 86400000) + 1,
    writer: 'Claude', // Historical credit is synthetic, and must survive unchanged.
    beats: ['07:00', '10:00', '13:00', '16:00'].map((start, i) => ({
      id: `${date}-${i + 1}`, start, title: 'TEST ONLY', text: [`Test cube ${date} step ${i + 1}.`],
      log: `Cube ${date} ${i + 1}`, scene: { setting_id: 'fixture-place', focus: { fx: 0.5, fy: 0.5 } },
    })) };
}

// Executes only as the explicitly selected fake native executable in a clean fake HOME.
function fakeNative() {
  const fs = require('node:fs'), path = require('node:path');
  const args = process.argv.slice(2), mode = process.env.FAKE_MODE;
  const prompt = fs.readFileSync(0, 'utf8');
  fs.writeFileSync(process.env.FAKE_PROMPT, prompt);
  fs.appendFileSync(process.env.FAKE_EVENTS, JSON.stringify({ type: 'fake-native', mode, args }) + '\n');
  const plan = JSON.parse(fs.readFileSync(process.env.FAKE_PLAN, 'utf8'));
  const write = (p, body) => { fs.mkdirSync(path.dirname(p), { recursive: true }); fs.writeFileSync(p, body); };
  if (mode !== 'refusal-zero') {
    for (const [i, entry] of plan.entries()) {
      if (mode === 'partial' && i === plan.length - 1) continue;
      write(entry.file, entry.body);
      if (mode !== 'no-brief') write(entry.brief, mode === 'empty-brief' ? ' \n' : 'SYNTHETIC BRIEF ONLY\n' + entry.date + '\n');
    }
    if (mode !== 'no-note') {
      const from = /Write every day from (\d{4}-\d{2}-\d{2}) through/.exec(prompt)[1];
      const through = /Write every day from \S+ through (\d{4}-\d{2}-\d{2})/.exec(prompt)[1];
      const stamp = /Today is (\d{4}-\d{2}-\d{2})/.exec(prompt)[1] + '-0000';
      const bound = mode === 'wrong-note-range' ? '1900-01-01' : from;
      const name = `${stamp}-serial-brief-${bound}-to-${through}.md`;
      const header = `SERIAL BRIEF date=${mode === 'wrong-note-header' ? '1900-01-01' : from}..${through}`;
      const prefix = mode === 'leading-note-line' ? '\n' : mode === 'leading-note-space' ? ' ' : '';
      const ending = mode === 'crlf-note' ? '\r\n' : '\n';
      write(path.join(process.env.FAKE_OUTBOX, name), prefix + header + ending + (mode === 'empty-note' ? '' : ending + 'Synthetic notice only. Briefs are in ' + process.env.FAKE_SPOILERS + '; PNGs go in ' + process.env.FAKE_INBOX + '/<date>/. Pictures by 21:00 Chicago the evening before.' + ending));
    }
  }
  const answer = args[args.indexOf('-o') + 1];
  write(answer, mode === 'refusal-zero' ? 'Synthetic refusal: sandbox_apply EPERM. Days written: 0.' : 'Synthetic completion claim only; the real consumer must check files.');
  if (mode === 'native-nonzero') process.exit(9);
  if (mode === 'native-signal') process.kill(process.pid, 'SIGTERM');
  if (mode === 'native-timeout') setInterval(() => {}, 1000);
}

function fixture(mode, selection = 'one') {
  const root = fs.mkdtempSync(path.join(runRoot, mode + '-'));
  const stage = path.join(root, 'stage'), home = path.join(root, 'home'), tmp = path.join(root, 'tmp'), bin = path.join(root, 'bin');
  const dropbox = path.join(root, 'FAKE-DROPBOX-NOT-WATCHED');
  const mailbox = path.join(dropbox, '00_SHARED_INBOX/series'), outbox = path.join(mailbox, 'from-claude');
  const inbox = path.join(mailbox, 'to-claude/serial'), spoilers = path.join(dropbox, '00_JARVIS_SPOILERS/desk-serial');
  const work = path.join(home, 'desk-serial-work'), events = path.join(root, 'events.jsonl');
  for (const p of [stage, home, tmp, bin, outbox, inbox, spoilers, work]) fs.mkdirSync(p, { recursive: true });
  const scripts = ['serial-mailbox.mjs', 'agent-run.mjs', 'eink-paths.mjs', 'serial-build.mjs'];
  for (const name of scripts) write(path.join(stage, 'scripts', name), fs.readFileSync(path.join(repo, 'scripts', name)));
  for (const name of ['src', 'scripts/series-agent', 'docs/superpowers/specs']) fs.cpSync(path.join(repo, name), path.join(stage, name), { recursive: true });
  for (const name of ['tsconfig.json', 'tsconfig.test.json', 'package.json']) fs.copyFileSync(path.join(repo, name), path.join(stage, name));
  fs.symlinkSync(path.join(repo, 'node_modules'), path.join(stage, 'node_modules'), 'dir');
  fs.symlinkSync(process.execPath, path.join(bin, 'node'));
  const native = path.join(bin, 'fake-native');
  write(native, `#!${process.execPath}\n(${fakeNative.toString()})();\n`); fs.chmodSync(native, 0o700);
  const preload = path.join(root, 'native-deadline.cjs');
  write(preload, `const cp = require('node:child_process');
const original = cp.spawnSync;
cp.spawnSync = function(command, args, options) {
  if (command === process.env.FAKE_NATIVE && process.env.FAKE_MODE === 'native-timeout') options = { ...options, timeout: 300 };
  return original.call(this, command, args, options);
};
require('node:module').syncBuiltinESMExports();\n`);
  write(path.join(bin, 'npx'), `#!${process.execPath}
const fs = require('node:fs'), cp = require('node:child_process');
const args = process.argv.slice(2);
if (args[0] === 'tsc') {
  fs.appendFileSync(process.env.FAKE_EVENTS, JSON.stringify({ type: 'real-compiler' }) + '\\n');
  const r = cp.spawnSync(process.execPath, [process.env.FAKE_TSC, ...args.slice(1)], { stdio: 'inherit' });
  process.exit(r.status ?? 92);
}
if (args.slice(0, 4).join(' ') === 'wrangler r2 object put') {
  fs.appendFileSync(process.env.FAKE_EVENTS, JSON.stringify({ type: 'fake-publish' }) + '\\n'); process.exit(0);
}
throw new Error('Unsupported external transport refused');\n`);
  fs.chmodSync(path.join(bin, 'npx'), 0o700);
  json(path.join(home, '.config/eink/paths.json'), { dropbox, photosLibrary: path.join(root, 'NO-PHOTOS'), familyTmp: path.join(tmp, 'family') });
  json(path.join(home, '.config/eink/agent.json'), { agent: 'codex', command: mode === 'native-spawn-error' ? path.join(bin, 'nonexistent-native') : native, model: 'gpt-6.1-sol', seriesReviewer: 'volt' });
  const today = todayChicago(), through = horizon(today), all = [];
  for (let date = shift(today, 1); date <= through; date = shift(date, 1)) all.push(date);
  const missing = selection === 'all' ? all : selection === 'none' ? [] : selection === 'two' ? [all[1], all.at(-1)] : [all.at(-1)];
  const seeded = [];
  for (const date of [today, ...all, shift(through, 1)]) if (!missing.includes(date)) {
    const file = path.join(work, 's01/days', date + '.json'); json(file, day(date)); seeded.push(file);
  }
  const settings = path.join(work, 's01/settings.json');
  json(settings, { writer: 'Claude', settings: [{ id: 'fixture-place', focus: { fx: 0.5, fy: 0.5 } }] }); seeded.push(settings);
  const before = Object.fromEntries(seeded.map(file => [file, hash(file)]));
  json(path.join(root, 'seeded-hashes.before.json'), before);
  const plan = missing.map(date => ({ date, file: path.join(work, 's01/days', date + '.json'), brief: path.join(spoilers, date, 'brief.md'), body: JSON.stringify(day(date)) }));
  for (const entry of plan) {
    const value = JSON.parse(entry.body);
    if (mode === 'invalid-json') { entry.body = '{ invalid synthetic JSON'; continue; }
    if (mode === 'wrong-date') value.date = today;
    if (mode === 'empty-beats') value.beats = [];
    if (mode === 'too-many-beats') value.beats.push(...value.beats);
    if (mode === 'misfit') value.beats[0].text = ['word '.repeat(200)];
    if (mode === 'bad-glyph') value.beats[0].text = ['Synthetic \u2603'];
    if (mode === 'bad-focus') value.beats[0].scene.focus.fx = 2;
    if (mode === 'unknown-setting') value.beats[0].scene.setting_id = 'not-in-library';
    if (mode === 'repeat-seeded') value.beats[0].text = day(today).beats[0].text;
    if (mode === 'repeat-generated') value.beats[0].text = ['Repeated synthetic line.'];
    entry.body = JSON.stringify(value);
  }
  const planFile = path.join(root, 'plan.json'); json(planFile, plan);
  // Use an already-installed system toolchain inside the unchanged sandbox.
  // Do not read the unrelated external Xcode tree or loosen any deny rule.
  const env = { PATH: bin + ':/usr/bin:/bin', HOME: home, TMPDIR: tmp, LANG: 'en_US.UTF-8', GIT_CONFIG_NOSYSTEM: '1',
    ...(process.platform === 'darwin' && fs.existsSync('/Library/Developer/CommandLineTools/usr/bin/git')
      ? { DEVELOPER_DIR: '/Library/Developer/CommandLineTools' } : {}),
    SERIAL_WORK: work, FAKE_NATIVE: native, FAKE_MODE: mode, FAKE_PLAN: planFile, FAKE_EVENTS: events,
    FAKE_PROMPT: path.join(root, 'resolved-prompt.md'), FAKE_OUTBOX: outbox, FAKE_SPOILERS: spoilers, FAKE_INBOX: inbox,
    FAKE_TSC: path.join(repo, 'node_modules/typescript/bin/tsc'), NODE_OPTIONS: '--require=' + preload };
  // Real disposable Git status, never a fake success or the actual checkout.
  const init = spawnSync('/usr/bin/git', ['init', '-q', stage], { env, encoding: 'utf8' });
  assert.equal(init.status, 0, init.stderr);
  let sandbox = [];
  if (process.platform === 'darwin') {
    const q = p => JSON.stringify(fs.realpathSync(p));
    const profile = path.join(root, 'confinement.sb');
    write(profile, `(version 1)\n(allow default)\n(deny network*)\n(deny file-write*)\n(allow file-write* (literal "/dev/null") (subpath ${q(root)}))\n(deny file-read* (subpath "/Users") (subpath "/Volumes"))\n(allow file-read-metadata)\n(allow file-read* (subpath ${q(root)}) (subpath ${q(path.join(repo, 'node_modules'))}) (literal ${q(process.execPath)}))\n`);
    sandbox = ['/usr/bin/sandbox-exec', '-f', profile];
  }
  json(path.join(root, 'copied-source-hashes.json'), Object.fromEntries([...scripts.map(n => 'scripts/' + n), 'scripts/series-agent/serial-writer.md'].map(n => [n, hash(path.join(stage, n))])));
  let invocation = 0;
  const f = { root, stage, work, home, outbox, spoilers, env, today, through, missing, plan,
    events: () => fs.existsSync(events) ? fs.readFileSync(events, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse) : [],
    log: () => fs.readFileSync(path.join(home, 'Library/Logs/eink-series/serial.log'), 'utf8'),
    run(command = 'weekly', file) {
      const args = [...sandbox, process.execPath, path.join(stage, 'scripts/serial-mailbox.mjs'), command, ...(file ? [file] : [])];
      const r = spawnSync(args[0], args.slice(1), { cwd: stage, env, encoding: 'utf8', timeout: 60000 });
      const label = String(++invocation);
      write(path.join(root, label + '.stdout'), r.stdout ?? ''); write(path.join(root, label + '.stderr'), r.stderr ?? '');
      json(path.join(root, label + '.result.json'), { status: r.status, signal: r.signal, error: r.error?.code });
      assert.equal(r.error, undefined, 'copied consumer must finish in time');
      assert.equal(r.signal, null, 'outer consumer must not be killed');
      assert.equal(todayChicago(), today, 'fixture must not straddle Chicago midnight');
      assert.deepEqual(Object.fromEntries(seeded.map(p => [p, hash(p)])), before, 'seeded days, settings and credits must remain byte-identical');
      assert.ok(!fs.existsSync(path.join(work, '.serial-weekly.lock')), 'weekly lock is released on every outcome');
      assert.equal(f.events().filter(e => e.type === 'fake-publish').length, 0, 'weekly must not publish');
      return r;
    } };
  return f;
}
const nativeCalls = f => f.events().filter(e => e.type === 'fake-native');

test('synthetic valid and misfit days exercise the unchanged real CLI checker', () => {
  const f = fixture('checker-control');
  write(f.plan[0].file, f.plan[0].body);
  const good = f.run('check', f.plan[0].file);
  assert.equal(good.status, 0, good.stdout + good.stderr);
  assert.match(good.stdout, /4 beats OK/);
  const bad = JSON.parse(f.plan[0].body);
  bad.beats[0].text = ['word '.repeat(200)];
  json(f.plan[0].file, bad);
  const rejected = f.run('check', f.plan[0].file);
  assert.notEqual(rejected.status, 0);
  assert.match(rejected.stdout, /story text longer/);
  assert.equal(nativeCalls(f).length, 0);
});

for (const mode of ['refusal-zero', 'partial', 'invalid-json', 'wrong-date', 'empty-beats', 'too-many-beats', 'misfit', 'bad-glyph', 'bad-focus', 'unknown-setting', 'repeat-seeded', 'repeat-generated']) {
  test(`weekly ${mode}: native exit zero cannot certify missing or invalid requested dates`, () => {
    const f = fixture(mode, ['partial', 'repeat-generated'].includes(mode) ? 'two' : 'one');
    const r = f.run();
    assert.equal(nativeCalls(f).length, 1, 'real runner reaches only the fake native transport');
    assert.notEqual(r.status, 0, 'weekly must fail closed instead of returning success');
    assert.match(f.log(), /weekly failed:/);
    assert.match(fs.readFileSync(path.join(f.outbox, 'NEEDS-THIAGO.md'), 'utf8'), /could not write|incomplete|did not finish/);
    if (mode === 'refusal-zero') assert.match(f.log(), /weekly: 0 of 1 days written/);
  });
}
for (const mode of ['native-nonzero', 'native-signal', 'native-timeout', 'native-spawn-error']) {
  test(`weekly ${mode}: surviving completion/files do not override failed native execution`, () => {
    const f = fixture(mode), r = f.run();
    if (mode !== 'native-spawn-error') assert.ok(fs.existsSync(f.plan[0].file), 'fake transport wrote the day before failing');
    assert.notEqual(r.status, 0, 'native failure must be propagated by weekly');
    assert.match(f.log(), /weekly failed:/);
  });
}
for (const mode of ['no-brief', 'empty-brief', 'no-note', 'wrong-note-range', 'wrong-note-header', 'empty-note', 'leading-note-line', 'leading-note-space']) {
  test(`weekly ${mode}: checked day alone is not a complete writer delivery`, () => {
    const f = fixture(mode), r = f.run();
    assert.notEqual(r.status, 0, 'brief and source-bound range note are required by the unchanged writer prompt');
    assert.match(f.log(), /weekly failed:/);
  });
}
test('weekly successful requested gap retains exact range, real checker, native configuration and idempotency', () => {
  const f = fixture('success', 'two');
  assert.equal(f.run().status, 0);
  assert.ok(f.events().some(e => e.type === 'real-compiler'), 'completion runs the real local checker/compiler');
  const prompt = fs.readFileSync(f.env.FAKE_PROMPT, 'utf8');
  assert.ok(prompt.includes(`Today is ${f.today}. Write every day from ${f.missing[0]} through ${f.through}.`));
  const call = nativeCalls(f)[0];
  for (const flag of ['--ignore-user-config', '--ephemeral', '--skip-git-repo-check']) assert.ok(call.args.includes(flag));
  assert.equal(call.args[call.args.indexOf('-m') + 1], 'gpt-6.1-sol');
  assert.equal(call.args[call.args.indexOf('-s') + 1], 'workspace-write');
  assert.equal(call.args[call.args.indexOf('-C') + 1], f.work);
  for (const entry of f.plan) {
    assert.equal(f.run('check', entry.file).status, 0, 'actual CLI checker accepts generated day');
    assert.ok(fs.readFileSync(entry.brief, 'utf8').trim());
  }
  const before = f.plan.map(e => hash(e.file));
  assert.equal(f.run().status, 0);
  assert.equal(nativeCalls(f).length, 1, 'no missing dates is an ordinary no-agent success');
  assert.deepEqual(f.plan.map(e => hash(e.file)), before);
  assert.match(f.log(), new RegExp(`weekly: written through ${f.through}`));
});
test('weekly all-missing horizon requires every date through the exact computed Sunday', () => {
  const f = fixture('success', 'all');
  assert.equal(f.missing[0], shift(f.today, 1));
  assert.equal(f.missing.at(-1), f.through);
  assert.equal(f.run().status, 0);
  assert.ok(f.events().some(e => e.type === 'real-compiler'));
  for (const entry of f.plan) assert.ok(fs.existsSync(entry.file) && fs.existsSync(entry.brief));
  assert.match(f.log(), new RegExp(`weekly: ${f.missing.length} of ${f.missing.length} days written`));
});
test('weekly CRLF briefing note preserves the exact first-line range convention', () => {
  const f = fixture('crlf-note');
  assert.equal(f.run().status, 0);
});
test('weekly preseeded no-missing run stays successful without native calls or compilation', () => {
  const f = fixture('success', 'none');
  assert.equal(f.run().status, 0);
  assert.deepEqual(f.events(), []);
  assert.match(f.log(), new RegExp(`weekly: written through ${f.through}`));
  assert.deepEqual(fs.readdirSync(f.outbox), []);
});
