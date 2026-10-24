// Synthetic OS launch boundary only; actual consumer, filesystem, separate waiter/producer processes.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync,spawn} from 'node:child_process';
import {pathToFileURL} from 'node:url';
const candidate=process.env.FAMILY_CANDIDATE || import.meta.dirname;
const {familyPhoto}=await import(pathToFileURL(path.join(candidate,'family-selection.mjs')));
const date='2026-10-19',r='11111111-1111-4111-8111-111111111111',r2='33333333-3333-4333-8333-333333333333';
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const pause=ms=>Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,ms);
function setup(){const root=fs.mkdtempSync(path.join(os.tmpdir(),'family-life-'));const home=path.join(root,'home');fs.mkdirSync(home);return {root,home,tmp:path.join(root,'out'),shown:path.join(root,'shown'),control:path.join(home,'Library/Caches/com.eink.family-fetch-control/active')};}
function options(f,request=r){return {tmp:f.tmp,shown:f.shown,home:f.home,request:()=>request,put:()=>assert.fail('timeout must not upload')};}
function worker(f){
 const script=path.join(f.root,'producer.cjs');
 fs.writeFileSync(script,`const fs=require('fs'),path=require('path');const [control,out,request,date,ready,gate,done]=process.argv.slice(2);fs.writeFileSync(ready,''+process.pid);fs.writeFileSync(path.join(control,'registered.json'),JSON.stringify({request,date,pid:process.pid}));setInterval(()=>{if(!fs.existsSync(gate))return;let o;try{o=JSON.parse(fs.readFileSync(path.join(control,'owner.json')))}catch{};if(o&&o.request===request&&o.date===date&&!fs.existsSync(path.join(control,'cancelled'))){fs.mkdirSync(path.join(out,request),{recursive:true});fs.writeFileSync(path.join(out,request,'late'),'bad');}if(o&&o.request===request)fs.writeFileSync(path.join(control,'stopped.json'),JSON.stringify({request,date,status:'cancelled',pid:process.pid}));fs.writeFileSync(done,'done');process.exit(0)},20);`);
 const waiter=path.join(f.root,'waiter.cjs');fs.writeFileSync(waiter,`const {spawn}=require('child_process');spawn(process.execPath,process.argv.slice(2),{detached:true,stdio:'ignore'}).unref();setInterval(()=>{},1000);`);
 return {launch:()=>spawnSync(process.execPath,[waiter,script,f.control,f.tmp,r,date,path.join(f.root,'ready'),path.join(f.root,'gate'),path.join(f.root,'done')],{timeout:300,stdio:'ignore'})};
}
test('waiter timeout retains ownership until delayed actual producer stops; successor cannot overlap',async()=>{
 const f=setup(),w=worker(f);let launches=0;
 try{
  familyPhoto(date,{...options(f),launch:w.launch});
  assert.ok(fs.existsSync(path.join(f.root,'ready')));
  assert.ok(fs.existsSync(f.control),'timed-out actual producer must remain owned');
  assert.ok(fs.existsSync(path.join(f.control,'cancelled')),'request cancelled before return');
  familyPhoto(date,{...options(f,r2),launch:()=>{launches++;return {status:1};}});
  assert.equal(launches,0,'successor must not launch while first producer exists');
  fs.writeFileSync(path.join(f.root,'gate'),'go');
  for(let i=0;i<100&&!fs.existsSync(path.join(f.root,'done'));i++)await sleep(20);
  assert.ok(fs.existsSync(path.join(f.root,'done')));assert.equal(fs.existsSync(path.join(f.tmp,r)),false,'no late output');
  familyPhoto(date,{...options(f,r2),launch:()=>{launches++;return {status:1};}});
  assert.equal(launches,1,'acknowledged cleanup permits successor');
 }finally{fs.writeFileSync(path.join(f.root,'gate'),'go');await sleep(100);fs.rmSync(f.root,{recursive:true,force:true});}
});
test('completed producer does not release a still-uploading consumer ownership',()=>{
 const f=setup();let launches=0;
 try{
  familyPhoto(date,{...options(f),launch:()=>{
   fs.writeFileSync(path.join(f.control,'registered.json'),JSON.stringify({request:r,date,pid:process.pid}));
   fs.writeFileSync(path.join(f.control,'stopped.json'),JSON.stringify({request:r,date,pid:process.pid,status:'complete'}));
   familyPhoto(date,{...options(f,r2),launch:()=>{launches++;return {status:1};}});
   return {status:1};
  }});
  assert.equal(launches,0,'app ack is not consumer/upload completion');
 }finally{fs.rmSync(f.root,{recursive:true,force:true});}
});
test('fixed typed authorization diagnostics never expose private errors or paths',()=>{
 for(const authorization of ['notDetermined','limited','denied','restricted']) {
  const f=setup(),logs=[];let puts=0;
  try {
   familyPhoto(date,{...options(f),log:s=>logs.push(s),put:()=>{puts++;return {status:0};},launch:()=>{
    fs.writeFileSync(path.join(f.control,'registered.json'),JSON.stringify({request:r,date,pid:process.pid}));
    fs.writeFileSync(path.join(f.control,'stopped.json'),JSON.stringify({request:r,date,pid:process.pid,status:'full-Photos-access-required',authorization}));return {status:3};
   }});
   assert.deepEqual(logs,[`family ${date}: full-Photos-access-required (${authorization}); skipped`]);
   assert.equal(puts,0);assert.equal(fs.existsSync(f.shown),false);assert.equal(fs.existsSync(f.control),false);
  }finally{fs.rmSync(f.root,{recursive:true,force:true});}
 }
});
test('startup timeout without registration is retained; wrong request/date ack cannot release it',()=>{
 const f=setup();let launches=0;
 try{
  familyPhoto(date,{...options(f),launch:()=>({status:null,error:{code:'ETIMEDOUT'}})});
  assert.ok(fs.existsSync(f.control));
  fs.writeFileSync(path.join(f.control,'registered.json'),JSON.stringify({request:r,date,pid:process.pid}));
  fs.writeFileSync(path.join(f.control,'stopped.json'),JSON.stringify({request:r,date:'2026-10-18',status:'cancelled',pid:process.pid}));
  familyPhoto(date,{...options(f,r2),launch:()=>{launches++;return {status:1};}});assert.equal(launches,0);
 }finally{fs.rmSync(f.root,{recursive:true,force:true});}
});
