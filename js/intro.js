/* 开屏字标：扁平的 PHOTOWORLD（照片填进字里）→ 按照片地形立起 → 溶解进三维场景
 *
 * 用 createPattern 把照片直接填进字形，不依赖离屏合成，稳定且快。
 * 立起的幅度由"照片的暗部"决定：越暗（越近的地面）立得越深，于是字标的立体感
 * 就是这张照片的地形 —— 这正是产品要做的事。
 */
(function (global) {
  'use strict';

  const WORD = 'PHOTOWORLD';
  const SIDE_COLOR = '#0d1216';

  function fontStack() {
    return '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif';
  }

  function createIntro(canvas) {
    const ctx = canvas.getContext('2d');
    let photo = null;
    let faceCv = null;
    let sideCv = null;
    const depthCols = new Float32Array(44).fill(0.5);
    let W = 2, H = 2, dpr = 1;
    let progress = 0;
    let dissolve = 0;
    const reduced = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function fs() { return Math.round(H * 0.5); }

    function wordMetrics(g) {
      const size = fs();
      g.font = '900 ' + size + 'px ' + fontStack();
      const sp = size * 0.055;
      let w = 0;
      for (const ch of WORD) w += g.measureText(ch).width + sp;
      return { size: size, sp: sp, width: w - sp, y: H * 0.56 };
    }

    function fillWord(g, style, dy, m) {
      g.fillStyle = style;
      g.textBaseline = 'middle';
      g.textAlign = 'left';
      let x = (W - m.width) / 2;
      for (const ch of WORD) {
        g.fillText(ch, x, m.y + dy);
        x += g.measureText(ch).width + m.sp;
      }
    }

    function build() {
      sideCv = document.createElement('canvas');
      sideCv.width = W; sideCv.height = H;
      const sg = sideCv.getContext('2d');
      fillWord(sg, SIDE_COLOR, 0, wordMetrics(sg));

      faceCv = document.createElement('canvas');
      faceCv.width = W; faceCv.height = H;
      const fg = faceCv.getContext('2d');
      const m = wordMetrics(fg);
      let ok = false;
      if (photo) {
        try {
          const s = Math.max(W / photo.width, H / photo.height);
          const dw = photo.width * s, dh = photo.height * s;
          try { fg.filter = 'brightness(1.22) contrast(1.08) saturate(1.15)'; } catch (e) {}
          const pat = fg.createPattern(photo, 'no-repeat');
          if (pat && pat.setTransform && global.DOMMatrix) {
            const mat = new global.DOMMatrix();
            mat.a = s; mat.d = s;
            mat.e = (W - dw) / 2;
            mat.f = (H - dh) / 2;
            pat.setTransform(mat);
            fillWord(fg, pat, 0, m);
            try { fg.filter = 'none'; } catch (e) {}
            ok = true;
          }
        } catch (e) { ok = false; }
      }
      if (!ok) fillWord(fg, '#79c8b8', 0, m);
      measureDepth();
    }

    /* 逐列"深度代理"：照片暗部 = 近 = 立得高 */
    function measureDepth() {
      if (!photo) return;
      const S = 96;
      const h = Math.max(8, Math.round(S * H / Math.max(1, W)));
      const tmp = document.createElement('canvas');
      tmp.width = S; tmp.height = h;
      const tg = tmp.getContext('2d', { willReadFrequently: true });
      const s = Math.max(S / photo.width, h / photo.height);
      tg.drawImage(photo, (S - photo.width * s) / 2, (h - photo.height * s) / 2, photo.width * s, photo.height * s);
      let data = null;
      try { data = tg.getImageData(0, 0, S, h).data; } catch (e) { data = null; }
      if (!data) return;
      const n = depthCols.length;
      for (let i = 0; i < n; i++) {
        const x0 = Math.floor(i / n * S), x1 = Math.max(x0 + 1, Math.floor((i + 1) / n * S));
        let sum = 0, cnt = 0;
        for (let x = x0; x < x1; x++) {
          for (let y = Math.floor(h * 0.5); y < h; y += 2) {
            const o = (y * S + x) * 4;
            sum += 0.2126 * data[o] + 0.7152 * data[o + 1] + 0.0722 * data[o + 2];
            cnt++;
          }
        }
        depthCols[i] = 1 - (sum / Math.max(1, cnt)) / 255;
      }
      for (let pass = 0; pass < 2; pass++) {
        const src = depthCols.slice();
        for (let i = 0; i < n; i++) {
          depthCols[i] = (src[Math.max(0, i - 1)] + src[i] * 2 + src[Math.min(n - 1, i + 1)]) / 4;
        }
      }
    }

    function render() {
      ctx.clearRect(0, 0, W, H);
      if (!faceCv) return;

      const lift = (reduced ? 0.55 : progress) * fs() * 0.30;
      const step = Math.max(1.5, 2 * dpr);
      const layers = Math.max(1, Math.round(1 + lift / step));

      ctx.save();
      ctx.globalAlpha = (0.5 + 0.5 * (reduced ? 1 : progress)) * (1 - dissolve);
      for (let k = layers; k >= 1; k--) {
        ctx.drawImage(sideCv, 0, (k / layers) * lift);
      }
      ctx.restore();

      ctx.save();
      ctx.globalAlpha = 1 - dissolve;
      const cols = depthCols.length;
      const sw = W / cols;
      for (let i = 0; i < cols; i++) {
        const sx = Math.floor(i * sw);
        const w = Math.max(1, Math.min(W - sx, Math.ceil(sw) + 1));
        const off = lift * (0.15 + 0.85 * (depthCols[i] || 0.5)) * 0.5;
        ctx.drawImage(faceCv, sx, 0, w, H, sx, off, w, H);
      }
      ctx.restore();
    }

    function measure() {
      const rect = canvas.getBoundingClientRect();
      dpr = Math.min(global.devicePixelRatio || 1, 2);
      const w = Math.max(2, Math.round(rect.width * dpr));
      const h = Math.max(2, Math.round(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
      }
      W = w; H = h;
      build();
      render();
    }

    function setPhoto(img) {
      if (!img || !img.width || photo === img) return;
      photo = img;
      measure();
    }

    function easeOut(x) { return 1 - Math.pow(1 - x, 3); }

    function play(onMid) {
      if (reduced) {
        progress = 1; dissolve = 0; render();
        if (onMid) onMid();
        return new Promise(function (res) { setTimeout(function () { dissolve = 1; render(); res(); }, 320); });
      }
      const t0 = performance.now();
      const DUR = 1500;
      let mid = false;
      return new Promise(function (res) {
        function tick(now) {
          const p = Math.min(1, (now - t0) / DUR);
          progress = p < 0.5 ? easeOut(Math.min(1, p / 0.5)) : 1;
          if (p >= 0.45 && !mid) { mid = true; if (onMid) onMid(); }
          dissolve = p < 0.72 ? 0 : Math.pow((p - 0.72) / 0.28, 1.5);
          render();
          if (p < 1) requestAnimationFrame(tick);
          else res();
        }
        requestAnimationFrame(tick);
      });
    }

    function reset() { progress = 0; dissolve = 0; render(); }

    measure();
    if (global.addEventListener) global.addEventListener('resize', measure);

    return { setPhoto: setPhoto, play: play, reset: reset, render: render, measure: measure };
  }

  global.PW = global.PW || {};
  global.PW.createIntro = createIntro;
})(window);
