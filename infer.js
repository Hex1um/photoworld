/* 浏览器端推理（与 train.py 的 forward 逐层对应）
   96x96x3 -> 24x24 高度 + 24x24x3 反照率
   结构：c1..c4 编码，24x24 处一条跳连，d1/d2 解码后与跳连拼接，双头输出 */
var PHOTO_INFER = (function () {
  'use strict';

  function conv(inp, IH, IW, IC, wk, bias, OC, stride) {
    var OH = ((IH + 2 - 3) / stride | 0) + 1;
    var OW = ((IW + 2 - 3) / stride | 0) + 1;
    var out = new Float32Array(OH * OW * OC);
    var K = IC * 9;
    for (var oy = 0; oy < OH; oy++) {
      for (var ox = 0; ox < OW; ox++) {
        var o = (oy * OW + ox) * OC;
        for (var ky = 0; ky < 3; ky++) {
          var iy = oy * stride + ky - 1;
          if (iy < 0 || iy >= IH) continue;
          for (var kx = 0; kx < 3; kx++) {
            var ix = ox * stride + kx - 1;
            if (ix < 0 || ix >= IW) continue;
            var ib = (iy * IW + ix) * IC;
            var wb = (ky * 3 + kx) * IC;
            for (var ci = 0; ci < IC; ci++) {
              var v = inp[ib + ci];
              if (v === 0) continue;
              var wrow = wb + ci;
              for (var co = 0; co < OC; co++) out[o + co] += v * wk[co * K + wrow];
            }
          }
        }
        for (var c2 = 0; c2 < OC; c2++) out[o + c2] += bias[c2];
      }
    }
    return { d: out, h: OH, w: OW, c: OC };
  }

  function relu(t) {
    for (var i = 0; i < t.d.length; i++) if (t.d[i] < 0) t.d[i] = 0;
    return t;
  }

  function up2(t) {
    var out = new Float32Array(t.h * 2 * t.w * 2 * t.c);
    for (var y = 0; y < t.h; y++) {
      for (var x = 0; x < t.w; x++) {
        var s = (y * t.w + x) * t.c;
        for (var dy = 0; dy < 2; dy++) {
          for (var dx = 0; dx < 2; dx++) {
            var o = ((y * 2 + dy) * t.w * 2 + (x * 2 + dx)) * t.c;
            for (var c = 0; c < t.c; c++) out[o + c] = t.d[s + c];
          }
        }
      }
    }
    return { d: out, h: t.h * 2, w: t.w * 2, c: t.c };
  }

  function sig(t) {
    for (var i = 0; i < t.d.length; i++) t.d[i] = 1 / (1 + Math.exp(-t.d[i]));
    return t;
  }

  function infer(pix, W) {
    var x = { d: pix, h: 96, w: 96, c: 3 };
    x = relu(conv(x.d, x.h, x.w, x.c, W.c1w, W.c1b, 24, 2));
    x = relu(conv(x.d, x.h, x.w, x.c, W.c2w, W.c2b, 48, 2));
    var skip = { d: x.d, h: x.h, w: x.w, c: x.c };
    x = relu(conv(x.d, x.h, x.w, x.c, W.c3w, W.c3b, 64, 2));
    x = relu(conv(x.d, x.h, x.w, x.c, W.c4w, W.c4b, 96, 2));
    x = relu(conv(up2(x).d, x.h * 2, x.w * 2, x.c, W.d1w, W.d1b, 64, 1));
    x = relu(conv(up2(x).d, x.h * 2, x.w * 2, x.c, W.d2w, W.d2b, 48, 1));

    var HW = x.h * x.w;
    var cat = new Float32Array(HW * 96);
    for (var i = 0; i < HW; i++) {
      for (var a = 0; a < 48; a++) cat[i * 96 + a] = x.d[i * 48 + a];
      for (var b = 0; b < 48; b++) cat[i * 96 + 48 + b] = skip.d[i * 48 + b];
    }

    var hh = sig(conv(cat, x.h, x.w, 96, W.hw, W.hb, 1, 1));
    var cc = sig(conv(cat, x.h, x.w, 96, W.cw, W.cb, 3, 1));
    return { h: hh, c: cc };
  }

  return { infer: infer };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = PHOTO_INFER;
