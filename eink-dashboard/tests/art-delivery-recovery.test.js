const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { spawnSync } = require('node:child_process');
const test = require('node:test');
const script = path.resolve(__dirname, '../scripts/series-mailbox.mjs');

function fixture(t, mode, kinds = ['family']) {
  const root = fs.mkdtempSync(path.join(process.env.TMPDIR || os.tmpdir(), 'eink-delivery-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const home = path.join(root, 'home'), mailbox = path.join(root, 'mailbox'), bin = path.join(root, 'bin');
  const inbox = path.join(mailbox, 'to-claude'), art = path.join(inbox, 'art');
  for (const dir of [art, path.join(mailbox, 'from-claude'), bin]) fs.mkdirSync(dir, { recursive: true });
  const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Chicago', year:'numeric', month:'2-digit', day:'2-digit' }).format(new Date());
  const note = 'test-art-delivery.md', bytes = Buffer.from([137,80,78,71,13,10,26,10,1,2,3,4]);
  fs.writeFileSync(path.join(inbox, note), `ART DELIVERY date=${date}\n`);
  fs.utimesSync(path.join(inbox, note), new Date(0), new Date(0));
  for (const kind of kinds) fs.writeFileSync(path.join(art, `${date}-${kind}.png`), bytes);
  const calls = path.join(root, 'calls.jsonl');
  const loader = path.join(root, 'loader.cjs');
  fs.writeFileSync(loader, 'Atomics.wait = () => "timed-out";');
  fs.writeFileSync(path.join(bin, 'npx'), `#!${process.execPath}
const fs=require('node:fs'),path=require('node:path');const a=process.argv.slice(2);
if(a[0]==='tsc'){const d=a[a.indexOf('--outDir')+1];fs.mkdirSync(path.join(d,'src/series'),{recursive:true});fs.writeFileSync(path.join(d,'src/series/mailbox.js'),${JSON.stringify(`exports.parseMessage = () => ({kind:'ART DELIVERY',date:${JSON.stringify(date)}});`)});process.exit(0);}
fs.appendFileSync(${JSON.stringify(calls)},JSON.stringify(a)+'\\n');
const puts=fs.readFileSync(${JSON.stringify(calls)},'utf8').trim().split('\\n').filter(x=>JSON.parse(x)[3]==='put').length;
const mode=${JSON.stringify(mode)};
if(a[3]==='put' && (mode==='transient' || (mode==='once' && puts===1))){console.error('TypeError: fetch failed');process.exit(1);}
if(a[3]==='put' && (mode==='denied' || mode.startsWith('stop-'))){console.log(mode==='denied' ? 'Authentication error [code: 10000] HTTP 403' : mode.slice(5));process.exit(1);}
if(a[3]==='get' && mode==='get-failure'){console.error('TypeError: fetch failed');process.exit(1);}
if(a[3]==='get'){fs.writeFileSync(a[a.indexOf('--file')+1],Buffer.from(${JSON.stringify([...bytes])}));if(mode==='mismatch')fs.appendFileSync(a[a.indexOf('--file')+1],'different');}
`, { mode: 0o700 });
  const run = () => spawnSync(process.execPath, ['--require', loader, script, 'run'], {
    encoding:'utf8', timeout:15000,
    env:{ HOME:home, PATH:`${bin}:${path.dirname(process.execPath)}:/usr/bin:/bin`, SERIES_MAILBOX:mailbox, TMPDIR:root, NODE_NO_WARNINGS:'1' },
  });
  return { run, inbox, art, date, note, bytes, root, calls:() => fs.existsSync(calls) ? fs.readFileSync(calls,'utf8').trim().split('\n').map(JSON.parse) : [] };
}

test('exhausted network retry leaves the note and original pending, not falsely handled', t => {
  const f=fixture(t,'transient'); const r=f.run(); assert.equal(r.status,0,r.stderr);
  assert.ok(fs.existsSync(path.join(f.inbox,f.note)), 'failed ART DELIVERY must remain queued');
  assert.equal(f.calls().filter(a=>a[3]==='put').length,2);
  assert.deepEqual(fs.readFileSync(path.join(f.art,`${f.date}-family.png`)),f.bytes);
});

test('access denial in stdout is never retried and stops later pictures', t => {
  const f=fixture(t,'denied',['moment','family']);f.run();
  assert.equal(f.calls().length,1,'do not retry a denial or upload the next kind');
  assert.ok(fs.existsSync(path.join(f.art,`${f.date}-moment.png`)));
  assert.ok(fs.existsSync(path.join(f.art,`${f.date}-family.png`)));
  assert.ok(fs.existsSync(path.join(f.inbox,'done',f.note)), 'blocked note must not be automatically replayed');
});

test('transient retry can succeed but requires exact stored-copy verification before archiving', t => {
  const f=fixture(t,'once');const r=f.run();assert.equal(r.status,0,r.stderr);
  assert.equal(f.calls().filter(a=>a[3]==='put').length,2);
  assert.equal(f.calls().filter(a=>a[3]==='get').length,1,'successful command is not stored-copy proof');
  assert.ok(fs.existsSync(path.join(f.inbox,'done',f.note)));
  assert.deepEqual(fs.readFileSync(path.join(f.art,'done',`${f.date}-family.png`)),f.bytes);
});

test('stored-copy mismatch preserves the original and does not claim received', t => {
  const f=fixture(t,'mismatch');f.run();
  assert.ok(fs.existsSync(path.join(f.art,`${f.date}-family.png`)),'keep the original on mismatched readback');
  const out=path.join(f.inbox,'..','from-claude');
  const receipts=fs.readdirSync(out).filter(x=>x.includes('art-received')).map(x=>fs.readFileSync(path.join(out,x),'utf8')).join('\n');
  assert.doesNotMatch(receipts,/: received/);
});

for (const message of ['HTTP 401', 'HTTP 403', 'HTTP 429', 'quota exceeded', 'unknown failure']) {
  test(`terminal ${message} stops immediately without a repeated native attempt`, t => {
    const f=fixture(t,`stop-${message}`);f.run();assert.equal(f.calls().length,1);
    f.run();assert.equal(f.calls().length,1,'archived blocked note must not replay');
    assert.ok(fs.existsSync(path.join(f.art,`${f.date}-family.png`)));
  });
}

test('readback connection failure leaves original and note pending despite successful put', t => {
  const f=fixture(t,'get-failure');f.run();
  assert.equal(f.calls().filter(a=>a[3]==='put').length,1);
  assert.equal(f.calls().filter(a=>a[3]==='get').length,2);
  assert.ok(fs.existsSync(path.join(f.inbox,f.note)));
  assert.ok(fs.existsSync(path.join(f.art,`${f.date}-family.png`)));
});

test('ordinary repeat uploads nothing already archived', t => {
  const f=fixture(t,'success');const r=f.run();assert.equal(r.status,0,r.stderr);
  assert.equal(f.calls().filter(a=>a[3]==='put').length,1);
  const count=f.calls().length;const again=f.run();assert.equal(again.status,0,again.stderr);
  assert.equal(f.calls().length,count,'do not resend completed pictures');
});
