// Actual Node consumer + production Swift control/snapshot/publication; only OS launch/Photos data are synthetic.
import test from 'node:test';import assert from 'node:assert/strict';
import fs from 'node:fs';import path from 'node:path';import os from 'node:os';import {spawnSync,spawn} from 'node:child_process';
import {familyPhoto} from './family-selection.mjs';
const build=fs.mkdtempSync(path.join(os.tmpdir(),'family-native-process-')),binary=path.join(build,'producer');
const compiled=spawnSync('swiftc',['-module-cache-path',process.env.SWIFT_MODULECACHE_PATH,path.join(import.meta.dirname,'family-core.swift'),path.join(import.meta.dirname,'family-lifecycle-harness.swift'),'-lsqlite3','-o',binary],{encoding:'utf8'});assert.equal(compiled.status,0,compiled.stderr);
const date='2026-10-19',request='11111111-1111-4111-8111-111111111111',second='33333333-3333-4333-8333-333333333333';
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function wait(file){for(let i=0;i<250;i++){if(fs.existsSync(file))return;await sleep(10);}assert.fail(`no synthetic ack: ${path.basename(file)}`);}
for(const mode of ['healthy','snapshot-timeout','download-timeout','late-startup'])test(`native actual producer: ${mode}`,async()=>{
 const root=fs.mkdtempSync(path.join(build,'case-')),home=path.join(root,'home'),tmp=path.join(root,'out'),shown=path.join(root,'shown');
 fs.mkdirSync(home);fs.mkdirSync(path.join(root,'library/database'),{recursive:true});fs.writeFileSync(path.join(root,'library/database/Photos.sqlite'),'synthetic only');
 const control=path.join(home,'Library/Caches/com.eink.family-fetch-control/active'),ready=path.join(root,'ready'),gate=path.join(root,'gate'),protectedFile=path.join(root,'protected');let launches=0,puts=0;
 if(mode==='healthy')fs.writeFileSync(gate,'go');
 const waiter=path.join(root,'waiter.cjs');fs.writeFileSync(waiter,`const {spawn}=require('child_process');const fs=require('fs');const [delay,binary,...args]=process.argv.slice(2);const child=()=>{const p=spawn(binary,args,{stdio:'ignore'});p.on('exit',code=>process.exit(code));};if(+delay){const {spawn}=require('child_process');spawn(process.execPath,['-e',${JSON.stringify("const {spawn}=require('child_process');setTimeout(()=>{spawn(process.argv[1],process.argv.slice(2),{detached:true,stdio:'ignore'}).unref()},500)")},binary,...args],{detached:true,stdio:'ignore'}).unref();setInterval(()=>{},1000);}else child();`);
 const launch=()=>{launches++;return spawnSync(process.execPath,[waiter,mode==='late-startup'?'500':'0',binary,home,date,request,ready,gate,protectedFile,tmp,root,mode],{timeout:mode==='healthy'?3000:250,stdio:'ignore'});};
 const deps={home,tmp,shown,request:()=>request,launch,put:()=>{puts++;return {status:0};}};
 try{
  familyPhoto(date,deps);
  if(mode==='healthy'){
   assert.equal(puts,2);assert.equal(fs.existsSync(shown),true);assert.equal(fs.existsSync(control),false);assert.equal(fs.existsSync(path.join(tmp,request)),false);
  }else{
   assert.equal(puts,0);assert.ok(fs.existsSync(control));assert.ok(fs.existsSync(path.join(control,'cancelled')));
   familyPhoto(date,{...deps,request:()=>second});assert.equal(launches,1,'no successor while producer/launch pending');
   fs.writeFileSync(gate,'go');await wait(path.join(control,'stopped.json'));
   assert.equal(fs.existsSync(protectedFile),false);assert.equal(fs.existsSync(path.join(tmp,request)),false);
   assert.deepEqual(fs.existsSync(path.join(root,'snapshots'))?fs.readdirSync(path.join(root,'snapshots')):[],[],'actual snapshot cleanup before ack');
   familyPhoto(date,{...deps,request:()=>second,launch:()=>{launches++;return {status:1};}});assert.equal(launches,2);
  }
 }finally{fs.writeFileSync(gate,'go');await sleep(100);fs.rmSync(root,{recursive:true,force:true});}
});
test('actual Node parent death retains request until native snapshot cleanup acknowledgment',async()=>{
 const root=fs.mkdtempSync(path.join(build,'parent-')),home=path.join(root,'home'),tmp=path.join(root,'out'),shown=path.join(root,'shown');
 fs.mkdirSync(home);fs.mkdirSync(path.join(root,'library/database'),{recursive:true});fs.writeFileSync(path.join(root,'library/database/Photos.sqlite'),'synthetic only');
 const control=path.join(home,'Library/Caches/com.eink.family-fetch-control/active'),gate=path.join(root,'gate'),ready=path.join(root,'ready');
 const waiter=path.join(root,'waiter.cjs');fs.writeFileSync(waiter,`const {spawn}=require('child_process');const p=spawn(process.argv[2],process.argv.slice(3),{stdio:'ignore'});p.on('exit',c=>process.exit(c));`);
 const consumer=path.join(root,'consumer.mjs');
 fs.writeFileSync(consumer,`import {familyPhoto} from ${JSON.stringify(new URL('./family-selection.mjs',import.meta.url).href)};import {spawnSync} from 'node:child_process';familyPhoto(${JSON.stringify(date)},{home:${JSON.stringify(home)},tmp:${JSON.stringify(tmp)},shown:${JSON.stringify(shown)},request:()=>${JSON.stringify(request)},launch:()=>spawnSync(process.execPath,${JSON.stringify([waiter,binary,home,date,request,ready,gate,path.join(root,'protected'),tmp,root,'snapshot-timeout'])},{timeout:10000,stdio:'ignore'}),put:()=>({status:0})});`);
 const parent=spawn(process.execPath,[consumer],{stdio:'ignore'});
 try {
  await wait(ready);assert.equal(fs.readdirSync(path.join(root,'snapshots')).length,1);
  parent.kill('SIGTERM');await new Promise(resolve=>parent.on('exit',resolve));
  assert.ok(fs.existsSync(control));assert.equal(fs.existsSync(path.join(control,'stopped.json')),false);
  fs.writeFileSync(gate,'go');await wait(path.join(control,'stopped.json'));
  assert.deepEqual(fs.readdirSync(path.join(root,'snapshots')),[]);assert.equal(fs.existsSync(path.join(tmp,request)),false);
  let launches=0;familyPhoto(date,{home,tmp,shown,request:()=>second,launch:()=>{launches++;return {status:1};},put:()=>({status:0})});assert.equal(launches,1,'dead consumer recovered only after app cleanup');
 } finally {fs.writeFileSync(gate,'go');if(parent.exitCode===null&&parent.signalCode===null)parent.kill('SIGTERM');await sleep(100);fs.rmSync(root,{recursive:true,force:true});}
});
test('helper output configuration must equal the exact owner binding before protected work',()=>{
 const root=fs.mkdtempSync(path.join(build,'config-')),home=path.join(root,'home'),tmp=path.join(root,'out'),actual=path.join(root,'other-output');
 fs.mkdirSync(home);fs.mkdirSync(path.join(root,'library/database'),{recursive:true});fs.writeFileSync(path.join(root,'library/database/Photos.sqlite'),'synthetic only');fs.writeFileSync(path.join(root,'gate'),'go');
 try {
  familyPhoto(date,{home,tmp,shown:path.join(root,'shown'),request:()=>request,put:()=>({status:0}),launch:()=>spawnSync(binary,[home,date,request,path.join(root,'ready'),path.join(root,'gate'),path.join(root,'protected'),actual,root,'healthy'],{timeout:3000})});
  assert.equal(fs.existsSync(path.join(actual,request)),false,'a config race must never leave winner output outside the exact owned path');
  assert.equal(fs.existsSync(path.join(root,'protected')),false);
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('real snapshot cleanup denial never acknowledges clean or permits successor',()=>{
 const root=fs.mkdtempSync(path.join(build,'denial-')),home=path.join(root,'home'),tmp=path.join(root,'out'),shown=path.join(root,'shown');let launches=0,puts=0;
 fs.mkdirSync(home);fs.mkdirSync(path.join(root,'library/database'),{recursive:true});fs.writeFileSync(path.join(root,'library/database/Photos.sqlite'),'synthetic only');
 const control=path.join(home,'Library/Caches/com.eink.family-fetch-control/active');
 const deps={home,tmp,shown,request:()=>request,put:()=>{puts++;return {status:0};},launch:()=>{launches++;return spawnSync(binary,[home,date,request,path.join(root,'ready'),path.join(root,'gate'),path.join(root,'protected'),tmp,root,'cleanup-denied'],{timeout:3000});}};
 try {
  familyPhoto(date,deps);assert.equal(puts,0);assert.ok(fs.existsSync(control));assert.equal(fs.existsSync(path.join(control,'stopped.json')),false,'failed removal is not cleanup success');
  familyPhoto(date,{...deps,request:()=>second});assert.equal(launches,1);
 }finally{
  const cache=path.join(root,'snapshots');if(fs.existsSync(cache))for(const name of fs.readdirSync(cache)){const sealed=path.join(cache,name,'sealed');if(fs.existsSync(sealed))fs.chmodSync(sealed,0o700);}
  fs.rmSync(root,{recursive:true,force:true});
 }
});
test.after(()=>fs.rmSync(build,{recursive:true,force:true}));
