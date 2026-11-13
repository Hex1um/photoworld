/* 开屏：一张照片 → 一个有纵深的世界（产品隐喻）
 *
 * 两层画布：
 *   1) #introBg   —— 全屏底噪：照片被"解析"成竖向切片，按深度上下错动 + 一条扫描线扫过，
 *                    看起来像机器正在测量这张照片的纵深（弱化、模糊、压暗，只做氛围）
 *   2) #introCanvas —— 字标：字母内部填充当前照片，厚度是一叠沿视线错开的切片，
 *                    自动循环演示"扁平的方块立成有厚度的实体"
 *
 * 两者都用同一张照片（画廊悬停会换图），所以用户在开屏就能看出"照片决定世界"。
 */
(function (global) {
  'use strict';

  const WORD = 'PHOTOWORLD';
  const SIDE_DARK = '#0a0e11';
  const BG_COLS = 16;          // 底噪切片列数
  const BG_DEPTH = 0.55;       // 底噪错动幅度（相对画布高度）

  function fontStack() {
    return '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif';
  }

  function createIntro(canvas, bgCanvas) {
    const ctx = canvas.getContext('2d');
    const bg = bgCanvas ? bgCanvas.getContext('2d') : null;
    let photo = null;
    let faceCv = null;
    let sideCv = null;
    let srcCv = null;                 // 底噪用的压暗模糊源
    let sharpCv = null;               // 锐利源：切片与等高线在上面画，才看得出"结构"
    let colDepth = null;              // 每列的相对深度 0..1（1=近）
    let W = 2, H = 2, dpr = 1;
    let BW = 2, BH = 2;
    let dissolve = 0;
    let raf = 0, t0 = 0, looping = false;
    let bgRaf = 0, bgT0 = 0;
    const reduced = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ---------------- 字标 ---------------- */
    function fs() {
      return Math.max(10, Math.round(Math.min(H * 0.52, W / 10.5)));
    }

    function metrics(g) {
      const size = fs();
      g.font = '900 ' + size + 'px ' + fontStack();
      const sp = size * 0.05;
      let w = 0;
      for (const ch of WORD) w += g.measureText(ch).width + sp;
      return { size: size, sp: sp, width: w - sp, y: H * 0.56 };
    }

    function fillWord(g, style, m) {
      g.fillStyle = style;
      g.textBaseline = 'middle';
      g.textAlign = 'left';
      let x = (W - m.width) / 2;
      for (const ch of WORD) {
        g.fillText(ch, x, m.y);
        x += g.measureText(ch).width + m.sp;
      }
    }

    function build() {
      sideCv = document.createElement('canvas');
      sideCv.width = W; sideCv.height = H;
      const sg = sideCv.getContext('2d');
      fillWord(sg, SIDE_DARK, metrics(sg));

      faceCv = document.createElement('canvas');
      faceCv.width = W; faceCv.height = H;
      const fg = faceCv.getContext('2d');
      const m = metrics(fg);
      let ok = false;
      if (photo) {
        try {
          const s = Math.max(W / photo.width, H / photo.height);
          const dw = photo.width * s, dh = photo.height * s;
          const pat = fg.createPattern(photo, 'no-repeat');
          if (pat && pat.setTransform && global.DOMMatrix) {
            const mat = new global.DOMMatrix();
            mat.a = s; mat.d = s;
            mat.e = (W - dw) / 2;
            mat.f = (H - dh) / 2;
            pat.setTransform(mat);
            if ('filter' in fg) fg.filter = 'brightness(1.18) contrast(1.06) saturate(1.12)';
            fillWord(fg, pat, m);
            if ('filter' in fg) fg.filter = 'none';
            ok = true;
          }
        } catch (e) { ok = false; }
      }
      if (!ok) fillWord(fg, '#79c8b8', m);
    }

    function drawFrame(thickness, yaw) {
      ctx.clearRect(0, 0, W, H);
      if (!faceCv) return;

      const m = metrics(ctx);
      const fit = Math.min(1, (W * 0.9) / Math.max(1, m.width));
      ctx.save();
      if (fit < 1) {
        ctx.translate(W / 2, H / 2);
        ctx.scale(fit, fit);
        ctx.translate(-W / 2, -H / 2);
      }

      const size = fs();
      const depth = size * 0.62 * thickness;
      const steps = Math.max(1, Math.round(depth / Math.max(1, 1.6 * dpr)));
      const sinY = Math.sin(yaw);

      if (thickness > 0.02 && dissolve < 0.9) {
        const cx = W / 2, cy = H * 0.92;
        const rx = Math.min(W * 0.34, size * 3.0), ry = size * 0.13;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rx);
        g.addColorStop(0, 'rgba(0,0,0,' + (0.32 * thickness * (1 - dissolve)).toFixed(3) + ')');
        g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.save();
        ctx.translate(cx, cy);
        ctx.scale(1, ry / rx);
        ctx.translate(-cx, -cy);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(cx, cy, rx, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      ctx.save();
      for (let k = steps; k >= 1; k--) {
        const z = k / steps;
        ctx.globalAlpha = (1 - 0.42 * z) * (1 - dissolve);
        ctx.drawImage(sideCv, -sinY * depth * z, 0);
      }
      ctx.globalAlpha = 1 - dissolve;
      ctx.drawImage(faceCv, 0, 0);
      ctx.restore();
      ctx.restore();
    }

    /* ---------------- 全屏底噪：照片切片 + 扫描线 ---------------- */
    function buildBg() {
      if (!bg || !photo) { srcCv = null; colDepth = null; return; }
      const sw = 480;
      const sh = Math.max(2, Math.round(sw * (BH / Math.max(1, BW))));
      srcCv = document.createElement('canvas');
      srcCv.width = sw; srcCv.height = sh;
      const sc = srcCv.getContext('2d');
      sharpCv = document.createElement('canvas');
      sharpCv.width = sw; sharpCv.height = sh;
      const hc = sharpCv.getContext('2d');
      try {
        const s = Math.max(sw / photo.width, sh / photo.height);
        const dw = photo.width * s, dh = photo.height * s;
        if ('filter' in sc) sc.filter = 'blur(8px) brightness(0.56) saturate(0.9)';
        sc.drawImage(photo, (sw - dw) / 2, (sh - dh) / 2, dw, dh);
        if ('filter' in sc) sc.filter = 'none';
        if ('filter' in hc) hc.filter = 'brightness(0.62) saturate(0.95)';
        hc.drawImage(photo, (sw - dw) / 2, (sh - dh) / 2, dw, dh);
        if ('filter' in hc) hc.filter = 'none';
      } catch (e) { srcCv = null; sharpCv = null; return; }

      // 每列深度：越暗 + 越靠上 → 越远（风景照的通用先验，够用且便宜）
      const probe = document.createElement('canvas');
      probe.width = BG_COLS; probe.height = 24;
      const pc = probe.getContext('2d', { willReadFrequently: true });
      let data = null;
      try {
        pc.drawImage(sharpCv, 0, 0, BG_COLS, 24);
        data = pc.getImageData(0, 0, BG_COLS, 24).data;
      } catch (e) { data = null; }
      colDepth = new Float32Array(BG_COLS);
      for (let i = 0; i < BG_COLS; i++) {
        if (!data) { colDepth[i] = 0.5; continue; }
        let lum = 0, n = 0;
        for (let y = 0; y < 24; y++) {
          const o = (y * BG_COLS + i) * 4;
          lum += (0.2126 * data[o] + 0.7152 * data[o + 1] + 0.0722 * data[o + 2]) / 255;
          n++;
        }
        lum /= Math.max(1, n);
        colDepth[i] = Math.min(1, Math.max(0, 0.85 * (1 - lum)));
      }
    }

    let scanGrad = null, lastScanH = 0;

    function drawBg(t) {
      if (!bg) return;
      bg.clearRect(0, 0, BW, BH);
      if (!srcCv || !colDepth) return;

      const ph = reduced ? 0.4 : Math.sin(t * 0.55) * 0.5 + 0.5;
      const amp = BH * 0.03 * (0.35 + 0.65 * ph);

      // 底图：轻微过扫，避免错动后露边
      bg.save();
      bg.globalAlpha = 1.0;
      const ox = -BW * 0.02, oy = -BH * 0.02;
      bg.drawImage(srcCv, ox, oy, BW * 1.04, BH * 1.04);
      bg.restore();

      // 切片错动：越近的列上下位移越大 → 读起来像"照片被解析出厚度"
      const colW = BW / BG_COLS;
      const m = BH * 0.02;
      const srcW = sharpCv.width / BG_COLS;
      for (let i = 0; i < BG_COLS; i++) {
        const d = colDepth[i];
        const dy = Math.sin(t * 0.9 + i * 0.24) * amp * d;
        bg.globalAlpha = 0.26;
        bg.drawImage(sharpCv, i * srcW, 0, srcW, sharpCv.height,
          i * colW - 0.6, dy - m, colW + 1.2, BH + m * 2);
      }
      bg.globalAlpha = 1;

      // 等高线：把深度场"画出来"，一眼看出这是被测量过的照片
      bg.strokeStyle = 'rgba(224,207,169,0.11)';
      bg.lineWidth = Math.max(1, BH / 900);
      for (let k = 0; k < 9; k++) {
        const base = BH * (0.16 + 0.72 * (k / 8));
        bg.beginPath();
        for (let i = 0; i <= BG_COLS; i++) {
          const ci = Math.min(BG_COLS - 1, i);
          const yy = base + Math.sin(t * 0.6 + ci * 0.7) * amp * 1.6 * colDepth[ci];
          if (i === 0) bg.moveTo(0, yy);
          else bg.lineTo((i / BG_COLS) * BW, yy);
        }
        bg.stroke();
      }

      // 扫描线：一条暖色细线 + 前方渐亮的拖尾，像机器正逐行测量
      const sy = ((t * 0.11) % 1.2 - 0.1) * BH;
      if (!scanGrad || lastScanH !== BH) {
        scanGrad = bg.createLinearGradient(0, -BH * 0.16, 0, 0);
        scanGrad.addColorStop(0, 'rgba(224,207,169,0)');
        scanGrad.addColorStop(0.86, 'rgba(224,207,169,0.07)');
        scanGrad.addColorStop(1, 'rgba(240,226,192,0.20)');
        lastScanH = BH;
      }
      bg.save();
      bg.translate(0, sy);
      bg.fillStyle = scanGrad;
      bg.fillRect(0, -BH * 0.16, BW, BH * 0.16);
      bg.fillStyle = 'rgba(240,226,192,0.26)';
      bg.fillRect(0, -1, BW, 1.6);
      bg.restore();

      // 暗角：把注意力推回中央字标
      const vg = bg.createRadialGradient(BW * 0.5, BH * 0.46, Math.min(BW, BH) * 0.12,
        BW * 0.5, BH * 0.46, Math.max(BW, BH) * 0.78);
      vg.addColorStop(0, 'rgba(8,9,11,0)');
      vg.addColorStop(0.62, 'rgba(8,9,11,0.34)');
      vg.addColorStop(1, 'rgba(8,9,11,0.84)');
      bg.fillStyle = vg;
      bg.fillRect(0, 0, BW, BH);
    }

    function bgLoop(now) {
      if (!looping) return;
      drawBg((now - bgT0) / 1000);
      bgRaf = requestAnimationFrame(bgLoop);
    }

    /* ---------------- 字标循环 ---------------- */
    function loop(now) {
      if (!looping) return;
      const p = (((now - t0) / 1000) % 7.0) / 7.0;
      let thickness = 0, yaw = 0;
      if (p < 0.14) {
        thickness = 0; yaw = 0;
      } else if (p < 0.38) {
        const k = (p - 0.14) / 0.24;
        const e = 1 - Math.pow(1 - k, 3);
        thickness = e; yaw = e * 0.40;
      } else if (p < 0.70) {
        const k = (p - 0.38) / 0.32;
        thickness = 1;
        yaw = 0.40 - Math.sin(k * Math.PI * 2) * 0.30;
      } else if (p < 0.88) {
        const k = (p - 0.70) / 0.18;
        const e = k * k * (3 - 2 * k);
        thickness = 1 - e; yaw = 0.40 * (1 - e);
      } else {
        thickness = 0; yaw = 0;
      }
      if (reduced) { thickness = 0.85; yaw = 0.34; }
      drawFrame(thickness, yaw);
      raf = requestAnimationFrame(loop);
    }

    function autoPlay() {
      if (looping) return;
      looping = true;
      t0 = performance.now();
      bgT0 = performance.now();
      if (reduced) {
        drawFrame(0.85, 0.34);
        drawBg(0.4);
        return;
      }
      raf = requestAnimationFrame(loop);
      bgRaf = requestAnimationFrame(bgLoop);
    }

    function stopLoop() {
      looping = false;
      if (raf) cancelAnimationFrame(raf);
      if (bgRaf) cancelAnimationFrame(bgRaf);
      raf = 0; bgRaf = 0;
    }

    function play(onMid) {
      stopLoop();
      return new Promise(function (res) {
        if (reduced) {
          drawFrame(0.85, 0.34);
          if (onMid) onMid();
          dissolve = 1; drawFrame(0.85, 0.34); res();
          return;
        }
        const start = performance.now();
        const DUR = 1100;
        let mid = false;
        function step(now) {
          const p = Math.min(1, (now - start) / DUR);
          const e = 1 - Math.pow(1 - Math.min(1, p / 0.55), 3);
          const thickness = 1 - 0.15 * Math.max(0, p - 0.55);
          if (p >= 0.5 && !mid) { mid = true; if (onMid) onMid(); }
          dissolve = p < 0.6 ? 0 : Math.pow((p - 0.6) / 0.4, 1.4);
          drawFrame(thickness, 0.5 * e);
          if (p < 1) requestAnimationFrame(step);
          else res();
        }
        requestAnimationFrame(step);
      });
    }

    /* ---------------- 尺寸 ---------------- */
    function measureBg() {
      if (!bgCanvas) return;
      const w = Math.max(2, Math.round(bgCanvas.clientWidth * dpr));
      const h = Math.max(2, Math.round(bgCanvas.clientHeight * dpr));
      if (bgCanvas.width !== w || bgCanvas.height !== h) {
        bgCanvas.width = w; bgCanvas.height = h;
      }
      BW = w; BH = h;
      scanGrad = null;
      buildBg();
      if (!looping) drawBg(0.4);
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
      measureBg();
      if (!looping) drawFrame(0, 0);
    }

    function setPhoto(img) {
      if (!img || !img.width || photo === img) return;
      photo = img;
      build();
      buildBg();
      if (!looping) { drawFrame(0, 0); drawBg(0.4); }
    }

    measure();
    if (global.addEventListener) global.addEventListener('resize', measure);

    return { setPhoto: setPhoto, play: play, autoPlay: autoPlay, stopLoop: stopLoop, measure: measure };
  }

  global.PW = global.PW || {};
  global.PW.createIntro = createIntro;
})(window);
