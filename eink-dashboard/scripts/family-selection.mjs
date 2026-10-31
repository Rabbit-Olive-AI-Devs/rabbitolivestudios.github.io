// Node receives only a winner JPEG and request-bound small receipt from the existing Photos app.
import crypto from 'node:crypto';
import {spawnSync} from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {fileURLToPath} from 'node:url';

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const fail = () => { throw new Error('family helper unavailable'); };
function validDate(s) {
  if (typeof s !== 'string' || !/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(s) || s.startsWith('0000')) return false;
  const d = new Date(`${s}T00:00:00Z`);
  return Number.isFinite(d.valueOf()) && d.toISOString().slice(0,10) === s;
}
function exists(file) {
  try { fs.lstatSync(file); return true; } catch (e) { if (e.code === 'ENOENT') return false; fail(); }
}
function safe(file) {
  if (typeof file !== 'string' || !path.isAbsolute(file) || path.normalize(file) !== file || file === '/') fail();
  let current = path.parse(file).root;
  for (const part of file.slice(current.length).split(path.sep)) {
    current = path.join(current,part);
    if (exists(current) && fs.lstatSync(current).isSymbolicLink()) fail();
  }
}
function smallJSON(file) {
  safe(file);
  const st = fs.lstatSync(file);
  if (!st.isFile() || st.size > 4096) fail();
  return JSON.parse(fs.readFileSync(file,'utf8'));
}
function exactKeys(o, keys) {
  return o && !Array.isArray(o) && Object.keys(o).sort().join(',') === keys.sort().join(',');
}
const SELECTION_REQUIRED = ['request','date','uuid','taken','source','round','width','height'];
// place is the approved exception (#92). latitude/longitude stay on this Mac and are optional.
const SELECTION_OPTIONAL = ['place','latitude','longitude'];
function selectionShape(o) {
  if (!o || Array.isArray(o)) return false;
  const keys = Object.keys(o);
  return SELECTION_REQUIRED.every(k => keys.includes(k)) && keys.every(k => SELECTION_REQUIRED.includes(k) || SELECTION_OPTIONAL.includes(k));
}
function validPlace(s) {
  return typeof s === 'string' && s.length >= 1 && s.length <= 80 && s === s.trim() && !/[\u0000-\u001f\u007f]/.test(s) && s !== 'Family Album';
}
function coord(v, min, max) {
  if (v == null) return null;
  if (typeof v !== 'number' || !Number.isFinite(v) || v < min || v > max) fail();
  return v;
}
function lookupPlace(lat, lon, home) {
  if (lat == null || lon == null) return 'Family';
  const script = fileURLToPath(new URL('./family-pick.py', import.meta.url));
  const cache = path.join(home, 'Library', 'Application Support', 'coelho-dashboard', 'places.json');
  const run = spawnSync('python3', ['-I', script, '--resolve-place', String(lat), String(lon), '--places-cache', cache], {encoding:'utf8', timeout:20000});
  const place = run.status === 0 ? String(run.stdout || '').trim() : '';
  return validPlace(place) ? place : 'Family';
}
// Read the dimensions encoded in the JPEG rather than trusting a stale text sidecar.
function dimensions(buf) {
  if (buf.length < 4 || buf[0] !== 255 || buf[1] !== 216) fail();
  let i = 2;
  while (i < buf.length) {
    if (buf[i++] !== 255) fail();
    while (i < buf.length && buf[i] === 255) i++;
    const marker = buf[i++];
    if (marker === 217 || marker === 218) break;
    if (marker === 1 || (marker >= 208 && marker <= 215)) continue;
    if (i+2 > buf.length) fail();
    const length = buf.readUInt16BE(i);
    if (length < 2 || i+length > buf.length) fail();
    if ([192,193,194,195,197,198,199,201,202,203,205,206,207].includes(marker)) {
      if (length < 8) fail();
      return {height:buf.readUInt16BE(i+3),width:buf.readUInt16BE(i+5)};
    }
    i += length;
  }
  fail();
}
export function readSelection(root,date,request) {
  if (!validDate(date) || !UUID.test(request)) fail();
  safe(root);
  const dir = path.join(root,request); safe(dir);
  if (!fs.lstatSync(dir).isDirectory()) fail();
  const done = smallJSON(path.join(dir,'completed.json'));
  if (!exactKeys(done,['request','date']) || done.request !== request || done.date !== date) fail();
  const c = smallJSON(path.join(dir,'selection.json'));
  if (!selectionShape(c) ||
      c.request !== request || c.date !== date || !UUID.test(c.uuid) || !validDate(c.taken) ||
      !['album','favourite'].includes(c.source) || !Number.isSafeInteger(c.round) || c.round < 1 ||
      !Number.isSafeInteger(c.width) || c.width < 1024 || !Number.isSafeInteger(c.height) || c.height < 1) fail();
  if (Object.prototype.hasOwnProperty.call(c,'place') && !validPlace(c.place)) fail();
  const lat = coord(c.latitude, -90, 90), lon = coord(c.longitude, -180, 180);
  if ((lat == null) !== (lon == null)) fail();
  const jpg = path.join(dir,'photo.jpg'); safe(jpg);
  const st = fs.lstatSync(jpg);
  if (!st.isFile() || st.size > 15*1024*1024) fail();
  const real = dimensions(fs.readFileSync(jpg));
  if (real.width !== c.width || real.height !== c.height) fail();
  return c;
}

// Persistent fixed-purpose ownership: timeout cancels, NEVER proves the LS app stopped.
// No PID signalling. An unresolved startup, crash, denial or hung producer retains this lease.
function controlFor(home) {
  const root=path.join(home,'Library/Caches/com.eink.family-fetch-control');
  safe(root); fs.mkdirSync(root,{recursive:true,mode:0o700});
  const st=fs.lstatSync(root);
  if (!st.isDirectory() || st.uid !== process.getuid()) fail();
  fs.chmodSync(root,0o700);
  return path.join(root,'active');
}
function stopped(control,owner) {
  try {
    const a=smallJSON(path.join(control,'registered.json'));
    const b=smallJSON(path.join(control,'stopped.json'));
    return a.request===owner.request && a.date===owner.date && Number.isSafeInteger(a.pid) && a.pid>1 &&
      b.request===owner.request && b.date===owner.date && b.pid===a.pid &&
      ['complete','cancelled','blocked','full-Photos-access-required'].includes(b.status);
  } catch { return false; }
}
function release(control,owner,tmp) {
  // A changed output configuration must not turn recovery into arbitrary path cleanup.
  if (owner.tmp!==tmp || !validDate(owner.date) || !UUID.test(owner.request) || !stopped(control,owner)) return false;
  const dir=path.join(tmp,owner.request);safe(dir);
  fs.rmSync(dir,{recursive:true,force:true});
  safe(control);fs.rmSync(control,{recursive:true});
  return true;
}
function transition(control,body) {
  const lock=path.join(path.dirname(control),'transition');safe(lock);
  fs.mkdirSync(lock,{mode:0o700});
  try { return body(); } finally { fs.rmdirSync(lock); }
}
function claim(home,tmp,date,request) {
  const control=controlFor(home);
  return transition(control,()=>{
  if (exists(control)) {
    const prior=smallJSON(path.join(control,'owner.json'));
    // App completion is not upload/shown-log completion. A living consumer still owns it.
    if (!exists(path.join(control,'cancelled'))) {
      if (!Number.isSafeInteger(prior.pid) || prior.pid<=1) fail();
      try { process.kill(prior.pid,0); fail(); }
      catch (error) { if (error.code!=='ESRCH') fail(); } // access denied retains ownership
    }
    if (!release(control,prior,tmp)) fail();
  }
  fs.mkdirSync(control,{mode:0o700}); // Atomic exclusion, including other mailbox processes.
  const owner={request,date,pid:process.pid,deadline:Date.now()+180000,tmp};
  fs.writeFileSync(path.join(control,'owner.json'),JSON.stringify(owner),{flag:'wx',mode:0o600});
  return {control,owner};
  });
}
function cancel(control) {
  safe(control);fs.writeFileSync(path.join(control,'cancelled'),'cancelled',{flag:'wx',mode:0o600});
}

// Dependencies are supplied by the mailbox (or explicit offline tests), never CLI/environment overrides.
export function familyPhoto(date,deps) {
  if (!validDate(date)) fail();
  const {tmp,shown,app,launch,put,log=()=>{},request:makeRequest=()=>crypto.randomUUID(),home=os.homedir(),placeOf} = deps;
  let dir, lease;
  try {
    safe(shown);
    const history = exists(shown) ? JSON.parse(fs.readFileSync(shown,'utf8')) : [];
    if (!Array.isArray(history)) fail();
    if (history.some(e=>e.date === date)) return;
    const request = makeRequest(); if (!UUID.test(request)) fail();
    safe(tmp); dir = path.join(tmp,request);
    if (exists(dir)) fail(); // A previous completion must never count for this invocation.
    lease=claim(home,tmp,date,request);
    const result = launch(['-W','-n','-g',app,'--args','--select-and-fetch',date,request]);
    if (result.status !== 0 || result.error) fail();
    if (!stopped(lease.control,lease.owner) || smallJSON(path.join(lease.control,'stopped.json')).status!=='complete') fail();
    const c = readSelection(tmp,date,request);
    const resolve = placeOf || ((lat,lon) => lookupPlace(lat,lon,home));
    let place = validPlace(c.place) ? c.place : resolve(c.latitude ?? null, c.longitude ?? null);
    if (!validPlace(place)) place = 'Family';
    const meta=path.join(dir,'photo.json');
    fs.writeFileSync(meta,JSON.stringify({taken:c.taken,place}),{flag:'wx',mode:0o600});
    if (put(path.join(dir,'photo.jpg'),`family/${date}/photo.jpg`,'image/jpeg').status !== 0 ||
        put(meta,`family/${date}/photo.json`,'application/json').status !== 0) { log(`family ${date}: upload failed`); return; }
    history.push({uuid:c.uuid,date,round:c.round});
    fs.mkdirSync(path.dirname(shown),{recursive:true});
    // Only after both uploads succeed. Same shown-log shape as the original picker.
    fs.writeFileSync(shown,JSON.stringify(history,null,1)+'\n');
    log(`family ${date}: photo from ${c.taken} (${c.source}) put up`);
  } catch {
    let diagnostic='helper blocked or incomplete';
    try {
      const s=smallJSON(path.join(lease.control,'stopped.json'));
      if (s.request===lease.owner.request && s.date===date && s.status==='full-Photos-access-required' &&
          ['notDetermined','limited','denied','restricted'].includes(s.authorization)) diagnostic=`full-Photos-access-required (${s.authorization})`;
    } catch {}
    log(`family ${date}: ${diagnostic}; skipped`);
  }
  finally {
    if (lease) {
      try {
        if (!exists(path.join(lease.control,'cancelled'))) cancel(lease.control);
        if (!transition(lease.control,()=>release(lease.control,lease.owner,tmp))) log(`family ${date}: producer ownership retained; cleanup unacknowledged`);
      } catch { log(`family ${date}: producer ownership retained; cleanup unacknowledged`); }
    }
  }
}
