#!/usr/bin/env node
/**
 * 无浏览器回归：三路 ONNX 推理结果的 IndexedDB 持久化缓存（v1.26）。
 *
 *   node tools/test_inference_cache.mjs
 *
 * 用最小 IndexedDB 模拟器覆盖：
 *  1) 无 indexedDB 环境静默降级（不影响首帧推理）；
 *  2) put / get 往返后 TypedArray 零精度损失；
 *  3) ns 隔离（深度 / 语义 / 实例互不串）；
 *  4) LRU 淘汰保留最新 DB_MAX 条；
 *  5) 模拟「刷新页面」：新 sandbox 复用同一个 IDB，同图不再跑 ONNX。
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (f) => fs.readFileSync(path.join(ROOT, f), 'utf8');
const SIZE = 518;
const tick = (ms = 30) => new Promise((r) => setTimeout(r, ms));

/* 只实现 js/depth.js 缓存层用到的 API：
   open / createObjectStore / createIndex / transaction / get / put / clear /
   index.count / index.openCursor（按 ts 升序）/ cursor.delete / cursor.continue。
   请求用 microtask 投递，事务 oncomplete 用 macrotask 投递，保证回调顺序。 */
function fakeIDB() {
  const records = new Map();
  let upgraded = false;
  const makeOS = () => {
    const req = (compute) => {
      const r = { onsuccess: null, onerror: null, result: undefined };
      queueMicrotask(() => {
        try { r.result = compute(); } catch (e) { if (r.onerror) r.onerror(); return; }
        if (r.onsuccess) r.onsuccess();
      });
      return r;
    };
    return {
      get: (k) => req(() => records.get(k)),
      put: (v) => req(() => { records.set(v.key, v); return v.key; }),
      clear: () => req(() => { records.clear(); }),
      createIndex: () => ({}),
      index: () => ({
        count: () => req(() => records.size),
        openCursor: () => {
          const sorted = Array.from(records.values()).sort((a, b) => a.ts - b.ts);
          let pos = 0;
          const cr = { onsuccess: null, result: null };
          const emit = () => {
            if (pos >= sorted.length) cr.result = null;
            else {
              const rec = sorted[pos];
              cr.result = {
                delete: () => { records.delete(rec.key); },
                continue: () => { pos++; queueMicrotask(emit); }
              };
            }
            if (cr.onsuccess) cr.onsuccess();
          };
          queueMicrotask(emit);
          return cr;
        }
      })
    };
  };
  const makeDB = (fresh) => ({
    objectStoreNames: { contains: (n) => !fresh && n === 'results' },
    createObjectStore: () => makeOS(),
    transaction: () => {
      const os = makeOS();
      const tx = { oncomplete: null, onerror: null, onabort: null, objectStore: () => os };
      setTimeout(() => { if (tx.oncomplete) tx.oncomplete(); }, 0);
      return tx;
    }
  });
  return {
    open: () => {
      const r = { onsuccess: null, onerror: null, onupgradeneeded: null, result: null };
      queueMicrotask(() => {
        if (!upgraded) {
          r.result = makeDB(true);     // 首次建库：objectStoreNames 为空，触发建索引
          if (r.onupgradeneeded) r.onupgradeneeded();
          upgraded = true;
        }
        r.result = makeDB(false);
        if (r.onsuccess) r.onsuccess();
      });
      return r;
    },
    _records: records
  };
}

function sandbox(idb) {
  const ctx2d = {
    fillRect() {}, drawImage() {}, putImageData() {}, fillStyle: '',
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    getImageData: (x, y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
  };
  const s = {
    console, URL, setTimeout, clearTimeout, queueMicrotask,
    fetch: () => Promise.resolve({ ok: true, arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)) }),
    document: {
      currentScript: { src: 'http://localhost/js/depth.js' },
      createElement: () => ({ width: 0, height: 0, getContext: () => ctx2d }),
    },
    location: { href: 'http://localhost/' },
    performance: { now: () => 0 },
  };
  if (idb) s.indexedDB = idb;
  s.window = s; s.global = s;
  vm.createContext(s);
  return s;
}

function withOrt(s) {
  s.PW = {};
  let runs = 0;
  s.ort = {
    env: { wasm: {} },
    Tensor: function (type, data, dims) { this.type = type; this.data = data; this.dims = dims; },
    InferenceSession: {
      create: async () => ({
        inputNames: ['input'], outputNames: ['output'],
        run: async () => {
          runs++;
          return { output: { data: new Float32Array(SIZE * SIZE), dims: [1, 1, SIZE, SIZE] } };
        }
      })
    }
  };
  vm.runInContext(read('js/depth.js'), s, { filename: 'depth.js' });
  return () => runs;
}

// ---- 1) 无 indexedDB：静默降级，不影响推理 ----
{
  // 同名函数静默覆盖是这个项目出过的事故，先锁住源码里只有一个 cacheStats
  const src = read('js/depth.js');
  for (const name of ['cacheStats', 'cacheClear', 'cacheGet', 'cachePut', 'idbStats']) {
    const n = (src.match(new RegExp('function ' + name + '\\(', 'g')) || []).length;
    assert.equal(n, 1, 'js/depth.js 里 ' + name + ' 定义了 ' + n + ' 次');
  }
  const s = sandbox(null);
  const runs = withOrt(s);
  const img = { naturalWidth: 1536, naturalHeight: 968, currentSrc: 'http://localhost/a.jpg' };
  const a = await s.PW.depth.estimate(img);
  const b = await s.PW.depth.estimate(img);
  assert.equal(runs(), 1, '无 IDB 时不应重复推理');
  assert.equal(b.cached, true, 'L1 内存缓存仍应命中');
  assert.equal(a.persisted, false);
  console.log('PASS  1  无 indexedDB 静默降级（L1 仍生效）');
}

// ---- 2) put / get 往返 + TypedArray 零精度损失 + ns 隔离 ----
const idb = fakeIDB();
{
  const s = sandbox(idb);
  withOrt(s);
  const C = s.PW.inferenceCache;
  const data = new Float32Array([0.123456789, 0.9999, 0.0000123, 0.5]);
  await C.put('depth-v2', 'photo-A', { data, width: 2, height: 2 });
  const got = await C.get('depth-v2', 'photo-A');
  assert.ok(got, 'put 之后 get 必须命中');
  assert.deepEqual(Array.from(got.data), Array.from(data), 'Float32 不得有精度损失');
  assert.equal(await C.get('sem-v1', 'photo-A'), null, '不同 ns 不得串数据');
  await C.put('sem-v1', 'photo-A', { groups: new Uint8Array([1, 2, 3]) });
  assert.equal((await C.get('sem-v1', 'photo-A')).groups[2], 3);
  console.log('PASS  2  put/get 往返零精度损失 + ns 隔离');
}

// ---- 3) 模拟刷新：新 sandbox 同图不再跑 ONNX ----
{
  const s1 = sandbox(idb);
  const runs1 = withOrt(s1);
  const img = { naturalWidth: 1536, naturalHeight: 968, currentSrc: 'http://localhost/refresh.jpg' };
  const first = await s1.PW.depth.estimate(img);
  assert.equal(first.cached, false);
  assert.equal(runs1(), 1);
  await tick();                       // 等 L2 异步写入落盘

  const s2 = sandbox(idb);            // 新页面：内存 LRU 为空，只有 IndexedDB
  const runs2 = withOrt(s2);
  const hit = await s2.PW.depth.estimate(img);
  assert.equal(runs2(), 0, '刷新后不应再跑 ONNX 推理');
  assert.equal(hit.cached, true);
  assert.equal(hit.persisted, true, '必须标记为持久化命中');
  assert.equal(hit.width, first.width);
  assert.equal(hit.height, first.height);
  assert.equal(hit.data.length, first.data.length);
  console.log('PASS  3  模拟刷新：同图 0 次推理，命中 L2（' + hit.data.length + ' floats）');
}

// ---- 4) LRU 淘汰保留最新 DB_MAX 条 ----
{
  const s = sandbox(idb);
  withOrt(s);
  const C = s.PW.inferenceCache;
  const MAX = C.DB_MAX;
  for (let i = 0; i < MAX + 8; i++) {
    await C.put('lru-test', 'k' + i, { i, payload: new Uint8Array([i & 255]) });
  }
  await tick(60);
  const keys = Array.from(idb._records.keys()).filter((k) => k.startsWith('lru-test:'));
  assert.ok(keys.length <= MAX, '淘汰后条数 ' + keys.length + ' 应 <= ' + MAX);
  assert.ok(await C.get('lru-test', 'k' + (MAX + 7)), '最新一条必须保留');
  assert.equal(await C.get('lru-test', 'k0'), null, '最旧一条必须被淘汰');
  console.log('PASS  4  LRU 淘汰：保留最新 ' + keys.length + '/' + MAX + ' 条');
}

// ---- 5) clear 清空后不再命中 ----
{
  const s = sandbox(idb);
  withOrt(s);
  await s.PW.inferenceCache.clear();
  await tick();
  assert.equal(await s.PW.inferenceCache.get('depth-v2', 'photo-A'), null);
  console.log('PASS  5  clear 清空持久化缓存');
}

console.log('\n推理缓存回归全部通过');
