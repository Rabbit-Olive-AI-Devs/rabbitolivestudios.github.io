// All files/receipts and image bytes here are synthetic fixtures, not live results.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const moduleURL = new URL('./family-selection.mjs', import.meta.url);
const request = '11111111-1111-4111-8111-111111111111';
const date = '2026-10-19';
const uuid = '22222222-2222-4222-8222-222222222222';
const selection = { request, date, uuid, taken:'2020-10-19', source:'album', round:2, width:1600, height:1000 };
// SOI, SOF0 (one component), EOI: tests the header reader, not image decoding.
const jpegHeader = Buffer.from([255,216,255,192,0,11,8,3,232,6,64,1,1,17,0,255,217]);
function fixture() {
  const root=fs.mkdtempSync(path.join(os.tmpdir(),'family-offline-'));
  const dir=path.join(root,request); fs.mkdirSync(dir);
  const write=(s=selection,m={request,date})=>{
    fs.writeFileSync(path.join(dir,'photo.jpg'),jpegHeader);
    fs.writeFileSync(path.join(dir,'selection.json'),JSON.stringify(s));
    fs.writeFileSync(path.join(dir,'completed.json'),JSON.stringify(m));
    const control=path.join(root,'home/Library/Caches/com.eink.family-fetch-control/active');
    if (fs.existsSync(control)) {
      fs.writeFileSync(path.join(control,'registered.json'),JSON.stringify({request,date,pid:process.pid}));
      fs.writeFileSync(path.join(control,'stopped.json'),JSON.stringify({request,date,pid:process.pid,status:'complete'}));
    }
  };
  return {root,dir,write,clean:()=>fs.rmSync(root,{recursive:true,force:true})};
}
async function module() {
  assert.ok(fs.existsSync(fileURLToPath(moduleURL)), 'missing feature: request-bound private helper consumer');
  return import(moduleURL);
}

test('request-bound receipt accepts only matching date/request and real JPEG dimensions',async()=>{
  const {readSelection}=await module(); const f=fixture();
  try {
    f.write(); assert.equal(readSelection(f.root,date,request).uuid,uuid);
    f.write({...selection, place:'Sample Shore'});
    assert.equal(readSelection(f.root,date,request).place, 'Sample Shore');
    for(const bad of [{...selection,request:'stale'}, {...selection,date:'2026-10-18'}, {...selection,width:1023}, {...selection,height:999}, {...selection,taken:'2020-02-30'}, {...selection,source:'other'}, {...selection,uuid:'../escape'}, {...selection,round:0}, {...selection,place:'Family Album'}, {...selection,place:'bad\nname'}, {...selection,place:''}, {...selection,latitude:91,longitude:0}, {...selection,latitude:1.25}]) {
      f.write(bad); assert.throws(()=>readSelection(f.root,date,request));
    }
    f.write(selection,{request:'stale',date}); assert.throws(()=>readSelection(f.root,date,request));
    f.write(selection,{request,date:'2026-10-18'}); assert.throws(()=>readSelection(f.root,date,request));
    f.write({...selection,privateExtra:'must reject'}); assert.throws(()=>readSelection(f.root,date,request));
    f.write(); fs.unlinkSync(path.join(f.dir,'completed.json')); assert.throws(()=>readSelection(f.root,date,request));
    f.write(); fs.unlinkSync(path.join(f.dir,'photo.jpg')); fs.symlinkSync(path.join(f.dir,'selection.json'),path.join(f.dir,'photo.jpg')); assert.throws(()=>readSelection(f.root,date,request));
  } finally {f.clean();}
});

test('one existing-app invocation; upload shape; shown written only after both uploads',async()=>{
  const {familyPhoto}=await module(); const f=fixture(); fs.rmSync(f.dir,{recursive:true});
  const shown=path.join(f.root,'shown.json'), calls=[], uploads=[];
  const deps={home:path.join(f.root,'home'),tmp:f.root,shown,app:'/synthetic/EinkFamilyFetch.app',request:()=>request,log:()=>{},
    launch:(args)=>{calls.push(args);fs.mkdirSync(f.dir);f.write();return {status:0};},
    put:(file,key,type)=>{uploads.push({key,type,body:key.endsWith('.json')?fs.readFileSync(file,'utf8'):null}); assert.equal(fs.existsSync(shown),false); return {status:0};}};
  try {
    familyPhoto(date,deps);
    assert.deepEqual(calls,[['-W','-n','-g',deps.app,'--args','--select-and-fetch',date,request]]);
    assert.deepEqual(uploads.map(u=>u.key),[`family/${date}/photo.jpg`,`family/${date}/photo.json`]);
    assert.deepEqual(JSON.parse(uploads[1].body), {taken:'2020-10-19', place:'Family'});
    assert.deepEqual(JSON.parse(fs.readFileSync(shown)),[{uuid,date,round:2}]);
    assert.equal(fs.existsSync(f.dir),false);
    familyPhoto(date,deps); assert.equal(calls.length,1);
  } finally {f.clean();}
});

test('failed upload or helper never writes shown and stale existing request cannot succeed',async()=>{
  const {familyPhoto}=await module();
  for(const kind of ['helper','first-upload','second-upload','stale','existing']) {
    const f=fixture(); if(kind!=='existing')fs.rmSync(f.dir,{recursive:true});
    const shown=path.join(f.root,'shown.json'); let puts=0,launches=0;
    const deps={home:path.join(f.root,'home'),tmp:f.root,shown,app:'/synthetic/EinkFamilyFetch.app',request:()=>request,log:()=>{},
      launch:()=>{launches++;fs.mkdirSync(f.dir);f.write(kind==='stale'?{...selection,date:'2026-10-18'}:selection);return {status:kind==='helper'?1:0};},
      put:()=>({status:++puts===(kind==='first-upload'?1:2)?1:0})};
    try { familyPhoto(date,deps); assert.equal(fs.existsSync(shown),false); if(kind==='existing')assert.equal(launches,0); }
    finally {f.clean();}
  }
});

test('coordinates are resolved to a place and never uploaded raw',async()=>{
  const {familyPhoto}=await module(); const f=fixture(); fs.rmSync(f.dir,{recursive:true});
  const shown=path.join(f.root,'shown.json'); const uploads=[]; let lookups=0;
  const deps={home:path.join(f.root,'home'),tmp:f.root,shown,app:'/synthetic/EinkFamilyFetch.app',request:()=>request,log:()=>{},
    placeOf:(lat,lon)=>{lookups++; assert.deepEqual([lat,lon],[1.25,2.5]); return 'Sample Shore';},
    launch:()=>{fs.mkdirSync(f.dir); f.write({...selection, latitude:1.25, longitude:2.5}); return {status:0};},
    put:(file,key)=>{uploads.push({key,body:key.endsWith('.json')?JSON.parse(fs.readFileSync(file,'utf8')):null}); return {status:0};}};
  try {
    familyPhoto(date,deps);
    assert.equal(lookups,1);
    assert.deepEqual(uploads[1].body, {taken:'2020-10-19', place:'Sample Shore'});
    assert.equal(Object.hasOwn(uploads[1].body,'latitude'), false);
  } finally {f.clean();}
});

test('invalid calendar date rejected before launch or file access',async()=>{
  const {familyPhoto}=await module();
  assert.throws(()=>familyPhoto('2026-02-30',{launch:()=>assert.fail('must not launch')}));
});
