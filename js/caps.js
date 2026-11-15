/* 运行能力探测：WebGPU / 本地重建服务 / 推理后端
 *
 * 存在的理由：这套东西有两条完全不同的重建链路（浏览器内在线链路 vs GPU 服务上
 * 的 v2 链路），观感差一个代际。使用者打开页面必须先知道**自己现在跑的是哪条**，
 * 否则"我这边效果不好"根本无法归因 —— 可能只是服务没起，也可能浏览器把活儿
 * 交给了核显。
 *
 * 所有探测都是**只读且带超时**的：不允许因为探测失败而阻塞进世界。
 */
(function (global) {
  'use strict';

  function withTimeout(promise, ms) {
    return new Promise((resolve) => {
      let done = false;
      const t = setTimeout(() => { if (!done) { done = true; resolve(null); } }, ms);
      promise.then((v) => {
        if (done) return;
        done = true; clearTimeout(t); resolve(v);
      }, () => {
        if (done) return;
        done = true; clearTimeout(t); resolve(null);
      });
    });
  }

  /* WebGPU：拿到 adapter 才算真的可用。只看 navigator.gpu 不够 ——
     Chromium 在只有软件光栅/被策略禁用时也会暴露 navigator.gpu。 */
  async function probeWebGPU() {
    const out = { present: !!navigator.gpu, ok: false, vendor: '', architecture: '',
      device: '', features: [], maxTexture: 0, reason: '' };
    if (!navigator.gpu) { out.reason = '浏览器没有 navigator.gpu'; return out; }
    try {
      const adapter = await withTimeout(
        navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }), 3000);
      if (!adapter) { out.reason = 'requestAdapter 返回 null（驱动/策略不支持）'; return out; }
      const info = adapter.info || {};
      out.ok = true;
      out.vendor = info.vendor || '';
      out.architecture = info.architecture || '';
      out.device = info.device || info.description || '';
      out.features = [...(adapter.features || [])];
      out.maxTexture = (adapter.limits && adapter.limits.maxTextureDimension2D) || 0;
      return out;
    } catch (e) {
      out.reason = 'requestAdapter 抛错：' + (e && e.message ? e.message : e);
      return out;
    }
  }

  /* 本地重建服务：/health 里有 gpu 与 reconstruct 两段，直接决定 v2 能不能用。 */
  async function probeService(api) {
    const out = { url: api || '', ok: false, gpu: '', reconstruct: null, reason: '' };
    if (!api) { out.reason = '未指定 ?api=，走浏览器内在线链路'; return out; }
    const base = String(api).replace(/\/+$/, '');
    try {
      const r = await withTimeout(fetch(base + '/health', { cache: 'no-store' }), 2500);
      if (!r || !r.ok) { out.reason = '服务无响应（' + base + '）'; return out; }
      const j = await r.json();
      out.ok = !!j.ok;
      out.gpu = j.gpu || '';
      out.reconstruct = j.reconstruct || null;
      if (!out.ok) out.reason = '服务返回 ok=false';
      else if (j.reconstruct && j.reconstruct.ok === false) {
        out.reason = j.reconstruct.reason || '服务缺少 v2 依赖';
      }
      return out;
    } catch (e) {
      out.reason = '连接失败：' + (e && e.message ? e.message : e);
      return out;
    }
  }

  /* 自动发现本机重建服务。
     为什么要这个：v2 的画质优势必须靠"起一个本地服务"，而让使用者手敲
     ?api=http://127.0.0.1:8100 是把人挡在门外的最大摩擦点。
     这里按"显式指定 → 上次成功的 → 常见端口"的顺序找，找到就记住。
     全部带超时，找不到就是找不到，不阻塞任何东西。 */
  const CANDIDATES = ['http://127.0.0.1:8100', 'http://localhost:8100', 'http://127.0.0.1:8101'];
  const LS_KEY = 'pw.api';

  function savedApi() {
    try { return localStorage.getItem(LS_KEY) || ''; } catch (e) { return ''; }
  }
  function saveApi(url) {
    try { url ? localStorage.setItem(LS_KEY, url) : localStorage.removeItem(LS_KEY); } catch (e) {}
  }

  async function discoverService(explicit) {
    const tried = [];
    const order = [];
    if (explicit) order.push(explicit);
    const s = savedApi();
    if (s && !order.includes(s)) order.push(s);
    for (const c of CANDIDATES) if (!order.includes(c)) order.push(c);
    for (const url of order) {
      const r = await probeService(url);
      tried.push({ url, ok: r.ok, reason: r.reason, reconstruct: r.reconstruct });
      if (r.ok) { saveApi(url); return { found: r, tried, url }; }
    }
    saveApi('');
    return { found: null, tried, url: '' };
  }

  /* 一句话结论，给顶栏徽标与自检页共用。 */
  function verdict(caps) {
    const s = (caps && caps.service) || {};
    const g = (caps && caps.webgpu) || {};
    if (s.ok && s.reconstruct && s.reconstruct.ok) {
      return { level: 'v2', label: 'v2 就绪', detail: 'GPU 服务 ' + (s.gpu || '已连接') };
    }
    if (s.ok) {
      return { level: 'v2-partial', label: 'v2 部分可用', detail: s.reason || '服务缺少 v2 依赖' };
    }
    if (g.ok && g.vendor && !/intel/i.test(g.vendor)) {
      return { level: 'gpu', label: '在线 · 独显', detail: 'WebGPU：' + (g.vendor + ' ' + g.architecture).trim() };
    }
    if (g.ok) {
      return { level: 'igpu', label: '在线 · 核显', detail: 'WebGPU 落在 ' + ((g.vendor || '?') + ' ' + (g.architecture || '')).trim() };
    }
    return { level: 'v1', label: '在线链路', detail: g.reason || '无 WebGPU' };
  }

  global.PW = global.PW || {};
  global.PW.caps = { probeWebGPU, probeService, verdict, withTimeout,
    discoverService, savedApi, saveApi, CANDIDATES };
})(window);
