/* 浏览器内图像补全：MI-GAN (Picsart Research, ICCV 2023) —— 「第三种方案」的网页版。
 *
 * 为什么不是原版 3d-photo-inpainting：那套是 PyTorch 三模型（color/depth/edge）
 * + MiDaS，单图 2~3 分钟、权重数百 MB，浏览器进不来。
 * MI-GAN 29MB、512×512、CPU 单次 0.57s（本机 onnxruntime 实测），
 * 这正是 inpaint-web / rullino / WatermarkOut 在浏览器里用的模型。
 *
 * 模型：https://hf-mirror.com/andraniksargsyan/migan/resolve/main/migan.onnx
 * 输入 image [1,3,512,512] uint8（RGB）、mask [1,1,512,512] uint8
 *   —— **mask 255 = 保留，0 = 生成**（实测：中心 255 时 centerDiff = 0.0）
 * 输出 result [1,3,512,512] uint8。
 *
 * 用法（简化 LDI）：把照片放在中心、四周留白，让模型补出画面外的内容，
 * 作为世界背景幕（worldBackdrop），转身/看远处时不再是一片雾。
 */
(function (global) {
  'use strict';

  const SELF = (document.currentScript && document.currentScript.src) || location.href;
  const BASE = new URL('../', SELF).href;
  const MODEL = BASE + 'models/migan.onnx';
  const S = 512;

  const S_ = { session: null, loading: null, failed: null, ort: null };

  function loadScript(src) {
    if (global.ort) return Promise.resolve();
    return new Promise((res, rej) => {
      const el = document.createElement('script');
      el.src = src;
      el.onload = () => res();
      el.onerror = () => rej(new Error('ort-web 加载失败'));
      document.head.appendChild(el);
    });
  }

  async function init(onProgress) {
    if (S_.session) return S_.session;
    if (S_.failed) throw S_.failed;
    if (S_.loading) return S_.loading;
    S_.loading = (async () => {
      await loadScript(BASE + 'vendor/ort/ort.min.js');
      const ort = global.ort;
      if (!ort) throw new Error('onnxruntime-web 未就绪');
      ort.env.wasm.wasmPaths = BASE + 'vendor/ort/';
      ort.env.wasm.numThreads = 1;
      ort.env.logLevel = 'error';
      S_.ort = ort;
      const r = await fetch(MODEL);
      if (!r.ok) throw new Error('MI-GAN 模型下载失败 ' + r.status);
      const total = Number(r.headers.get('content-length')) || 29 * 1024 * 1024;
      const reader = r.body && r.body.getReader ? r.body.getReader() : null;
      let buf;
      if (reader) {
        const chunks = [];
        let got = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          got += value.length;
          if (onProgress) onProgress(Math.min(0.95, got / total));
        }
        buf = new Uint8Array(got);
        let off = 0;
        for (const c of chunks) { buf.set(c, off); off += c.length; }
      } else {
        buf = new Uint8Array(await r.arrayBuffer());
      }
      const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
      if (onProgress) onProgress(1);
      S_.session = session;
      return session;
    })();
    try { return await S_.loading; }
    catch (e) { S_.failed = e; S_.loading = null; throw e; }
  }

  /* 把照片缩放到中心区域，四周留 pad（默认 18%），让模型补出画面外 */
  async function expand(img, opt, onProgress) {
    opt = opt || {};
    const pad = opt.pad === undefined ? 0.18 : opt.pad;
    const session = await init(onProgress);
    const ort = S_.ort;
    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;

    const inner = Math.round(S * (1 - 2 * pad));
    const sc = inner / Math.max(iw, ih);
    const nw = Math.max(1, Math.round(iw * sc));
    const nh = Math.max(1, Math.round(ih * sc));
    const ox = Math.floor((S - nw) / 2), oy = Math.floor((S - nh) / 2);

    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    // 先把整张画布用照片边缘色填满，避免黑边被当成内容
    ctx.drawImage(img, 0, 0, iw, ih, 0, 0, S, S);
    const edge = ctx.getImageData(0, 0, S, S).data;
    const avg = [0, 0, 0];
    let cnt = 0;
    for (let i = 0; i < edge.length; i += 40) { avg[0] += edge[i]; avg[1] += edge[i + 1]; avg[2] += edge[i + 2]; cnt++; }
    ctx.fillStyle = 'rgb(' + Math.round(avg[0] / cnt) + ',' + Math.round(avg[1] / cnt) + ',' + Math.round(avg[2] / cnt) + ')';
    ctx.fillRect(0, 0, S, S);
    ctx.drawImage(img, 0, 0, iw, ih, ox, oy, nw, nh);

    const rgba = ctx.getImageData(0, 0, S, S).data;
    const image = new Uint8Array(3 * S * S);
    const mask = new Uint8Array(S * S);
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const p = (y * S + x) * 4;
        const q = y * S + x;
        image[q] = rgba[p];
        image[S * S + q] = rgba[p + 1];
        image[2 * S * S + q] = rgba[p + 2];
        // 255 = 保留；中心区域保留，四周交给模型
        mask[q] = (x >= ox && x < ox + nw && y >= oy && y < oy + nh) ? 255 : 0;
      }
    }

    const out = await session.run({
      image: new ort.Tensor('uint8', image, [1, 3, S, S]),
      mask: new ort.Tensor('uint8', mask, [1, 1, S, S])
    });
    const data = out[Object.keys(out)[0]].data;

    const res = document.createElement('canvas');
    res.width = res.height = S;
    const rctx = res.getContext('2d');
    const dst = rctx.createImageData(S, S);
    for (let q = 0; q < S * S; q++) {
      dst.data[q * 4] = data[q];
      dst.data[q * 4 + 1] = data[S * S + q];
      dst.data[q * 4 + 2] = data[2 * S * S + q];
      dst.data[q * 4 + 3] = 255;
    }
    rctx.putImageData(dst, 0, 0);
    return res;
  }

  /* 按给定 mask 补全：maskCanvas 里 **白色 = 保留、黑色 = 补**（与模型极性一致）。
     用于 LDI：把前景物体擦掉，让模型补出「它背后」的背景。 */
  async function fill(img, maskCanvas, onProgress) {
    const session = await init(onProgress);
    const ort = S_.ort;
    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;

    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, iw, ih, 0, 0, S, S);
    const rgba = ctx.getImageData(0, 0, S, S).data;

    const mc = document.createElement('canvas');
    mc.width = mc.height = S;
    const mctx = mc.getContext('2d', { willReadFrequently: true });
    mctx.drawImage(maskCanvas, 0, 0, maskCanvas.width, maskCanvas.height, 0, 0, S, S);
    const md = mctx.getImageData(0, 0, S, S).data;

    const image = new Uint8Array(3 * S * S);
    const mask = new Uint8Array(S * S);
    for (let q = 0; q < S * S; q++) {
      const p = q * 4;
      image[q] = rgba[p];
      image[S * S + q] = rgba[p + 1];
      image[2 * S * S + q] = rgba[p + 2];
      // 灰度 > 127 视为「保留」（255），否则「补」（0）
      mask[q] = md[p] > 127 ? 255 : 0;
    }

    const out = await session.run({
      image: new ort.Tensor('uint8', image, [1, 3, S, S]),
      mask: new ort.Tensor('uint8', mask, [1, 1, S, S])
    });
    const data = out[Object.keys(out)[0]].data;

    const res = document.createElement('canvas');
    res.width = res.height = S;
    const rctx = res.getContext('2d');
    const dst = rctx.createImageData(S, S);
    for (let q = 0; q < S * S; q++) {
      dst.data[q * 4] = data[q];
      dst.data[q * 4 + 1] = data[S * S + q];
      dst.data[q * 4 + 2] = data[2 * S * S + q];
      dst.data[q * 4 + 3] = 255;
    }
    rctx.putImageData(dst, 0, 0);
    return res;
  }

  global.PW = global.PW || {};
  global.PW.inpaint = {
    init, expand, fill,
    get ready() { return !!S_.session; },
    get failed() { return S_.failed; }
  };
})(window);
