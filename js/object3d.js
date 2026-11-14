/* 单体真网格升级（可选，混合方案的「高质量」那一半）
 *
 * 默认路径：world.js 的十字账单牌（本地、零依赖、绝不拉伸）。
 * 本模块：把近景显著单体（树/房屋/人/车）的裁片发给本地 GPU 服务，
 *        用 InstantMesh / TripoSR 生成真网格，替换掉对应的账单牌，
 *        这样绕到侧面/背面也是合理的三维形状。
 *
 * 触发条件：?obj3d=1 且 ?api=http://127.0.0.1:8100（服务端 /health 里 object3d 不为 none）
 * 任何一步失败都保留账单牌，不影响主流程。
 */
(function (global) {
  'use strict';

  function makeTexture(gl, img) {
    if (!img) return null;
    return global.PW.GL.createTexture(gl, img, { wrap: 'clamp', clampV: true, aniso: 4, flipY: false });
  }

  function transformLayer(layer, it, targetH, targetW) {
    const p = layer.pos;
    let minY = 1e9, maxY = -1e9, minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9;
    for (let i = 0; i < p.length; i += 3) {
      const x = p[i], y = p[i + 1], z = p[i + 2];
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const mh = Math.max(1e-3, maxY - minY);
    const mw = Math.max(1e-3, maxX - minX);
    const s = Math.min(targetH / mh, targetW / mw);
    const cx = (minX + maxX) * 0.5, cz = (minZ + maxZ) * 0.5;
    const out = new Float32Array(p.length);
    for (let i = 0; i < p.length; i += 3) {
      out[i] = it.cx + (p[i] - cx) * s;
      out[i + 1] = it.ymin + (p[i + 1] - minY) * s;
      out[i + 2] = it.cz + (p[i + 2] - cz) * s;
    }
    return out;
  }

  async function upgrade(world, img, opts) {
    opts = opts || {};
    const server = (opts.server || '').replace(/\/$/, '');
    const gl = opts.gl;
    if (!server || !gl || !world || !world.objectInstances || !world.objectInstances.length) {
      return { upgraded: 0, reason: 'no-server-or-instances' };
    }
    const max = opts.max || 4;
    const minT = opts.minT || 2, maxT = opts.maxT || 70;
    const cands = world.objectInstances
      .filter((it) => it.t >= minT && it.t <= maxT)
      .sort((a, b) => b.area - a.area)
      .slice(0, max);
    if (!cands.length) return { upgraded: 0, reason: 'no-candidate' };

    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;
    const cv = document.createElement('canvas');
    const ctx = cv.getContext('2d');
    const vfov = (world.reliefMeta && world.reliefMeta.vfov) || 0.62;
    const tanV = Math.tan(vfov / 2), tanH = Math.tan(31 * Math.PI / 180);

    const upgraded = [];
    for (const it of cands) {
      try {
        const u0 = it.uv[0], v0 = it.uv[1], u1 = it.uv[2], v1 = it.uv[3];
        const sx = Math.max(0, Math.floor(u0 * iw)), sy = Math.max(0, Math.floor(v0 * ih));
        const sw = Math.max(8, Math.min(iw - sx, Math.ceil((u1 - u0) * iw)));
        const sh = Math.max(8, Math.min(ih - sy, Math.ceil((v1 - v0) * ih)));
        cv.width = sw; cv.height = sh;
        ctx.clearRect(0, 0, sw, sh);
        ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
        const blob = await new Promise((res) => cv.toBlob(res, 'image/png'));
        if (!blob) continue;

        const fd = new FormData();
        fd.append('image', blob, 'object.png');
        fd.append('bbox', '0,0,1,1');
        const r = await fetch(server + '/api/object3d', { method: 'POST', body: fd });
        if (r.status === 501) break;             // 服务端没有模型，保留账单牌
        if (!r.ok) continue;

        const buf = await r.arrayBuffer();
        const url = URL.createObjectURL(new Blob([buf], { type: 'model/gltf-binary' }));
        let asset;
        try { asset = await global.PW.loadGLB(url); }
        finally { URL.revokeObjectURL(url); }
        if (!asset || !asset.layers.length) continue;

        const targetH = Math.max(1.0, (v1 - v0) * 2 * tanV * it.t);
        const targetW = Math.max(1.0, (u1 - u0) * 2 * tanH * it.t);
        const tex = makeTexture(gl, asset.img);

        for (let li = 0; li < asset.layers.length; li++) {
          const l = asset.layers[li];
          const pos = transformLayer(l, it, targetH, targetW);
          const data = {
            pos, normal: l.nrm, uv: l.uv,
            color: l.color, index: l.index
          };
          let mesh;
          try { mesh = global.PW.GL.createMesh(gl, data); }
          catch (e) { continue; }
          if (!world.extraMeshes) world.extraMeshes = [];
          world.extraMeshes.push({
            key: 'obj' + it.id + '_' + li, mesh,
            tex: tex || world.tex.ground,
            fog: 0.18
          });
        }
        upgraded.push(it.id);
      } catch (e) {
        // 单个实例失败不影响其他实例
        console.warn('单体补全失败 id=' + it.id, e && e.message);
      }
    }

    if (upgraded.length && world.hideObjectCards) world.hideObjectCards(upgraded);
    if (document.body) document.body.dataset.obj3d = String(upgraded.length);
    return { upgraded: upgraded.length, ids: upgraded };
  }

  global.PW = global.PW || {};
  global.PW.object3d = { upgrade };
})(window);
