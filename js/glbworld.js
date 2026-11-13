/* 资产加载模式：读取离线管线产出的 world.glb
 *
 * 只支持本仓库管线写出的那一种子集（单 mesh、POSITION/NORMAL/TEXCOORD_0、
 * 内嵌 JPEG、PBR 材质），足够把 glb 直接喂给现有渲染器，且不引入 three.js。
 * 坐标系与浏览器端一致：原点在拍摄点，x 右、y 上、-Z 为视线方向，单位米。
 */
(function (global) {
  'use strict';

  function parseGLB(buf) {
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('不是 GLB 文件');
    const version = dv.getUint32(4, true);
    const total = dv.getUint32(8, true);
    let off = 12, json = null, bin = null;
    while (off < total) {
      const len = dv.getUint32(off, true);
      const type = dv.getUint32(off + 4, true);
      const start = off + 8;
      if (type === 0x4E4F534A) {
        json = JSON.parse(new TextDecoder('utf-8').decode(new Uint8Array(buf, start, len)));
      } else if (type === 0x004E4942) {
        bin = new Uint8Array(buf, start, len);
      }
      off = start + len;
    }
    if (!json) throw new Error('GLB 缺少 JSON 块');
    return { json, bin, version };
  }

  function readAccessor(json, bin, idx) {
    const acc = json.accessors[idx];
    const view = json.bufferViews[acc.bufferView];
    const compSize = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 }[acc.componentType];
    const typeSize = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 }[acc.type];
    const start = (view.byteOffset || 0) + (acc.byteOffset || 0);
    const count = acc.count * typeSize;
    let arr;
    if (acc.componentType === 5126) {
      arr = new Float32Array(bin.buffer, bin.byteOffset + start, count);
    } else if (acc.componentType === 5125) {
      arr = new Uint32Array(bin.buffer, bin.byteOffset + start, count);
    } else if (acc.componentType === 5123) {
      arr = new Uint16Array(bin.buffer, bin.byteOffset + start, count);
    } else {
      throw new Error('不支持的 accessor 类型 ' + acc.componentType);
    }
    // 必须拷贝：ArrayBuffer 可能是共享/对齐限制
    const out = arr.slice(0, count);
    return { data: out, size: typeSize, count: acc.count };
  }

  function toImage(bytes) {
    return new Promise((res, rej) => {
      const blob = new Blob([bytes], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => { URL.revokeObjectURL(url); res(img); };
      img.onerror = () => { URL.revokeObjectURL(url); rej(new Error('贴图解码失败')); };
      img.src = url;
    });
  }

  async function load(url, onProgress) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('加载失败 ' + resp.status);
    const total = Number(resp.headers.get('content-length') || 0);
    let buf;
    if (resp.body && total) {
      const reader = resp.body.getReader();
      const chunks = [];
      let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        got += value.length;
        if (onProgress) onProgress(got / total);
      }
      buf = new Uint8Array(got);
      let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
      buf = buf.buffer;
    } else {
      buf = await resp.arrayBuffer();
    }

    const { json, bin } = parseGLB(buf);
    const layers = [];
    for (let mi = 0; mi < json.meshes.length; mi++) {
      const m = json.meshes[mi];
      if (!m.primitives || !m.primitives.length) continue;
      const prim = m.primitives[0];
      const pos = readAccessor(json, bin, prim.attributes.POSITION);
      const nrm = prim.attributes.NORMAL !== undefined ? readAccessor(json, bin, prim.attributes.NORMAL) : null;
      const uv = prim.attributes.TEXCOORD_0 !== undefined ? readAccessor(json, bin, prim.attributes.TEXCOORD_0) : null;
      // 顶点色（HunyuanWorld / ply2glb 产出的无贴图资产走这条）；VEC3 补齐成 VEC4
      const vcol = prim.attributes.COLOR_0 !== undefined ? readAccessor(json, bin, prim.attributes.COLOR_0) : null;
      let colData = null;
      if (vcol) {
        if (vcol.size === 3) {
          const n = vcol.count;
          colData = new Float32Array(n * 4);
          for (let i = 0; i < n; i++) {
            colData[i * 4] = vcol.data[i * 3];
            colData[i * 4 + 1] = vcol.data[i * 3 + 1];
            colData[i * 4 + 2] = vcol.data[i * 3 + 2];
            colData[i * 4 + 3] = 1;
          }
        } else {
          colData = vcol.data;
        }
      }
      const idx = readAccessor(json, bin, prim.indices);
      layers.push({
        name: m.name || ('layer' + mi),
        pos: pos.data,
        nrm: nrm ? nrm.data : null,
        uv: uv ? uv.data : null,
        color: colData,
        index: idx.data,
        vertices: pos.count,
        triangles: idx.count / 3
      });
    }

    let img = null;
    if (json.images && json.images.length) {
      const bv = json.bufferViews[json.images[0].bufferView];
      const start = bv.byteOffset || 0;
      img = await toImage(bin.slice(start, start + bv.byteLength));
    }

    let verts = 0, tris = 0;
    for (const l of layers) { verts += l.vertices; tris += l.triangles; }
    return { layers, img, vertices: verts, triangles: tris, bytes: buf.byteLength };
  }

  global.PW = global.PW || {};
  global.PW.loadGLB = load;
})(window);
