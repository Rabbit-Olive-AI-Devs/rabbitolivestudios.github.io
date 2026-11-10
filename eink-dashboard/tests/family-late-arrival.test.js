const assert = require('node:assert/strict');
const test = require('node:test');
const path = require('node:path');
const B = process.env.EINK_TEST_BUILD_DIR;
const { storeFamily, familyStyleFor } = require(path.join(B, 'src/family.js'));
const { encodePNGRGB } = require(path.join(B, 'src/png.js'));
const D = '2026-11-10';
const K = `family:v1:${D}`;
const ART = `art/${D}/family`;
const META = `family/${D}/photo.json`;
const PHOTO = `family/${D}/photo.jpg`;
const USED = `art/used/${D}/family`;

async function env() {
  const png = await encodePNGRGB(new Uint8Array(800 * 480 * 3).fill(125), 800, 480);
  const r2 = new Map();
  const kv = new Map();
  const calls = [];
  return {
    r2, kv, calls,
    PHOTOS: {
      get: async (k) => r2.has(k) ? { json: async () => JSON.parse(r2.get(k)), arrayBuffer: async () => new TextEncoder().encode(r2.get(k)).buffer } : null,
      head: async (k) => r2.has(k) ? {} : null,
      put: async (k,v) => {calls.push(`r2-put:${k}`);r2.set(k,v);},
      delete: async (k) => {calls.push(`r2-delete:${k}`);r2.delete(k);},
    },
    CACHE: {get: async(k)=>kv.get(k)??null, put:async(k,v,opts)=>{calls.push(`kv-put:${k}`);assert.equal(opts.expirationTtl,604800);kv.set(k,v);}},
    IMAGES: { input: () => ({transform: () => ({output: async () => ({response: () => new Response(png)})})}) },
    AI: { run: async () => { throw new Error('No generation is permitted'); } },
  };
}
const caption = () => ({date:'2023-12-03',yearsAgo:3,place:'Sample Café',style:familyStyleFor(D).id});
function cached(c=caption()) {return JSON.stringify({image:'cHJldmlvdXM=',...c});}

test('late family painting replaces first fallback after its temporary metadata was deleted', async () => {
 const e=await env();e.r2.set(META,JSON.stringify({taken:'2023-12-03',place:'Sample Café'}));e.r2.set(PHOTO,'source-photo');
 assert.equal(await storeFamily(e,D),'photo');assert.ok(!e.r2.has(META));assert.ok(!e.r2.has(PHOTO));
 const first=JSON.parse(e.kv.get(K));e.r2.set(ART,'late-accepted-painting');
 assert.equal(await storeFamily(e,D),'jarvis');
 const later=JSON.parse(e.kv.get(K));assert.deepEqual({...later,image:undefined},{...first,image:undefined});
 assert.ok(e.r2.has(USED));assert.ok(!e.r2.has(ART));
 assert.ok(e.calls.indexOf(`kv-put:${K}`)<e.calls.indexOf(`r2-put:${USED}`));
});

test('missing caption never consumes a late painting or claims it was used',async()=>{
 for(const value of [undefined,'legacy-raw-base64','{broken','{"image":"only"}',JSON.stringify({image:'x',date:'bad',yearsAgo:3,style:familyStyleFor(D).id})]) {
  const e=await env();if(value!==undefined)e.kv.set(K,value);e.r2.set(ART,'painting');
  assert.equal(await storeFamily(e,D),null);assert.equal(e.kv.get(K),value);assert.ok(e.r2.has(ART));assert.ok(!e.r2.has(USED));
 }
});

test('caption must match this day, its age and its scheduled style',async()=>{
 for (const bad of [{...caption(),style:'wrong-style'},{...caption(),yearsAgo:42},{...caption(),yearsAgo:2.5},{...caption(),date:'2023-99-03'},{...caption(),date:'2023-02-31'},{...caption(),date:'2027-11-10'}]) {
  const e=await env();const old=cached(bad);e.kv.set(K,old);e.r2.set(ART,'painting');
  assert.equal(await storeFamily(e,D),null);assert.equal(e.kv.get(K),old);assert.ok(e.r2.has(ART));assert.ok(!e.r2.has(USED));
 }
});

test('a temporary image-read failure preserves fallback, painting and provenance',async()=>{
 const e=await env();const old=cached();e.kv.set(K,old);e.r2.set(ART,'painting');
 e.IMAGES.input=()=>{throw new Error('synthetic converter failure');};
 assert.equal(await storeFamily(e,D),null);assert.equal(e.kv.get(K),old);assert.ok(e.r2.has(ART));assert.ok(!e.r2.has(USED));
});

test('cache write failure never deletes the late painting or writes its used marker',async()=>{
 const e=await env();const old=cached();e.kv.set(K,old);e.r2.set(ART,'painting');
 e.CACHE.put=async()=>{throw new Error('synthetic cache write failure');};
 await assert.rejects(storeFamily(e,D),/synthetic cache write failure/);assert.equal(e.kv.get(K),old);assert.ok(e.r2.has(ART));assert.ok(!e.r2.has(USED));
});

test('no late painting leaves an existing good cache untouched',async()=>{
 const e=await env();const old=cached();e.kv.set(K,old);assert.equal(await storeFamily(e,D),null);assert.equal(e.kv.get(K),old);assert.equal(e.calls.length,0);
});

test('malformed existing metadata is not silently overridden by a cache caption',async()=>{
 const e=await env();const old=cached();e.kv.set(K,old);e.r2.set(META,'{"taken":"bad"}');e.r2.set(ART,'painting');
 assert.equal(await storeFamily(e,D),null);assert.equal(e.kv.get(K),old);assert.ok(e.r2.has(ART));assert.ok(!e.r2.has(USED));
});

test('a new raw photograph without its metadata cannot borrow the old caption',async()=>{
 const e=await env();const old=cached();e.kv.set(K,old);e.r2.set(PHOTO,'different-photo');
 assert.equal(await storeFamily(e,D),null);assert.equal(e.kv.get(K),old);assert.ok(e.r2.has(PHOTO));assert.ok(!e.r2.has(USED));
});

test('a used-marker write failure retains the accepted painting for safe recovery',async()=>{
 const e=await env();e.kv.set(K,cached());e.r2.set(ART,'painting');e.PHOTOS.put=async()=>{throw new Error('synthetic marker write failure');};
 await assert.rejects(storeFamily(e,D),/synthetic marker write failure/);assert.ok(e.r2.has(ART));assert.ok(!e.r2.has(USED));assert.equal(JSON.parse(e.kv.get(K)).date,caption().date);
});
