/* 开屏字标：PHOTOWORLD 从"扁平"立成"立体"，自动循环演示
 *
 * 字母内部填充当前照片；厚度是一叠沿视线方向错开的切片，配合偏航角摆动，
 * 看起来就是一块有厚度的实体被转过来。这正是产品的隐喻：
 * 一张平面的照片 → 一个有纵深的世界。
 */
(function (global) {
  'use strict';

  const WORD = 'PHOTOWORLD';
  const SIDE_DARK = '#0a0e11';

  function fontStack() {
    return '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif';
  }

  function createIntro(canvas) {
    const ctx = canvas.getContext('2d');
    let photo = null;
    let faceCv = null;
    let sideCv = null;
    let W = 2, H = 2, dpr = 1;
    let dissolve = 0;
    let raf = 0, t0 = 0, looping = false;
    const reduced = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function fs() {
      // 双约束：高度上限 + 宽度上限（10 个字母，粗体宽度约 1.05em）
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

      // 兜底缩放：无论字号怎么算，都保证字标完整落在画布内（窄屏尤其重要）
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
      if (reduced) { drawFrame(0.85, 0.34); return; }
      raf = requestAnimationFrame(loop);
    }

    function stopLoop() {
      looping = false;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
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
      if (!looping) drawFrame(0, 0);
    }

    function setPhoto(img) {
      if (!img || !img.width || photo === img) return;
      photo = img;
      build();
      if (!looping) drawFrame(0, 0);
    }

    measure();
    if (global.addEventListener) global.addEventListener('resize', measure);

    return { setPhoto: setPhoto, play: play, autoPlay: autoPlay, stopLoop: stopLoop, measure: measure };
  }

  global.PW = global.PW || {};
  global.PW.createIntro = createIntro;
})(window);
