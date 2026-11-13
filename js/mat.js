/* 极简 4x4 矩阵 / 向量工具（列主序，与 WebGL 一致） */
(function (global) {
  'use strict';

  const M4 = {
    create() {
      const m = new Float32Array(16);
      m[0] = m[5] = m[10] = m[15] = 1;
      return m;
    },

    identity(out) {
      out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
      out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0;
      out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0;
      out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
      return out;
    },

    perspective(out, fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2);
      const nf = 1 / (near - far);
      out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0;
      out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0;
      out[8] = 0; out[9] = 0; out[10] = (far + near) * nf; out[11] = -1;
      out[12] = 0; out[13] = 0; out[14] = 2 * far * near * nf; out[15] = 0;
      return out;
    },

    multiply(out, a, b) {
      const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
      const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
      const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
      const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
      for (let i = 0; i < 4; i++) {
        const b0 = b[i * 4], b1 = b[i * 4 + 1], b2 = b[i * 4 + 2], b3 = b[i * 4 + 3];
        out[i * 4] = b0 * a00 + b1 * a10 + b2 * a20 + b3 * a30;
        out[i * 4 + 1] = b0 * a01 + b1 * a11 + b2 * a21 + b3 * a31;
        out[i * 4 + 2] = b0 * a02 + b1 * a12 + b2 * a22 + b3 * a32;
        out[i * 4 + 3] = b0 * a03 + b1 * a13 + b2 * a23 + b3 * a33;
      }
      return out;
    },

    perspectiveFromCamera(out, eye, yaw, pitch, fovy, aspect, near, far) {
      // view = R^T * T(-eye)，直接展开得到 VP
      const proj = M4.perspective(M4.create(), fovy, aspect, near, far);
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      // 相机朝向向量
      const fx = sy * cp, fy = sp, fz = -cy * cp;
      // right / up
      const rx = cy, ry = 0, rz = sy;
      const ux = -sy * sp, uy = cp, uz = cy * sp;
      const view = new Float32Array(16);
      view[0] = rx; view[4] = ry; view[8] = rz;
      view[1] = ux; view[5] = uy; view[9] = uz;
      view[2] = -fx; view[6] = -fy; view[10] = -fz;
      view[12] = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);
      view[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
      view[14] = (fx * eye[0] + fy * eye[1] + fz * eye[2]);
      view[15] = 1;
      return M4.multiply(out, proj, view);
    },

    invert(out, a) {
      const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
      const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
      const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
      const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

      const b00 = a00 * a11 - a01 * a10;
      const b01 = a00 * a12 - a02 * a10;
      const b02 = a00 * a13 - a03 * a10;
      const b03 = a01 * a12 - a02 * a11;
      const b04 = a01 * a13 - a03 * a11;
      const b05 = a02 * a13 - a03 * a12;
      const b06 = a20 * a31 - a21 * a30;
      const b07 = a20 * a32 - a22 * a30;
      const b08 = a20 * a33 - a23 * a30;
      const b09 = a21 * a32 - a22 * a31;
      const b10 = a21 * a33 - a23 * a31;
      const b11 = a22 * a33 - a23 * a32;

      let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
      if (!det) return null;
      det = 1.0 / det;

      out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
      out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
      out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
      out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
      out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
      out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
      out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
      out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
      out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
      out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
      out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
      out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
      out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
      out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
      out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
      out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
      return out;
    }
  };

  /* 确定性随机数（同一个世界每次生成一致） */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hashString(str) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function smoothstep(e0, e1, x) {
    const t = clamp((x - e0) / (e1 - e0 || 1e-6), 0, 1);
    return t * t * (3 - 2 * t);
  }

  global.PW = global.PW || {};
  global.PW.M4 = M4;
  global.PW.util = { mulberry32, hashString, clamp, lerp, smoothstep };
})(window);
