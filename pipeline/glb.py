"""极简 glTF 2.0 / GLB 写出器（不依赖 trimesh）。

只支持：一个 mesh、一个 PBR 材质、内嵌 JPEG 贴图、可选 Draco 不做（体积靠降采样控制）。
"""
from __future__ import annotations

import json
import struct

import numpy as np


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(path: str, layers: list, texture_jpeg: bytes, name: str = "world",
              extras: dict | None = None) -> dict:
    """layers: [{"pos","nrm","uv","index","name"}, ...] —— 同一张贴图，多个 mesh。"""
    blobs: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    offset = 0
    total_verts = 0
    total_tris = 0

    def add_view(data: bytes, target: int | None = None) -> int:
        nonlocal offset
        padded = _pad4(data)
        blobs.append(padded)
        v = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        offset += len(padded)
        return len(views) - 1

    for li, layer in enumerate(layers):
        pos = np.asarray(layer["pos"], dtype=np.float32).reshape(-1, 3)
        is_points = bool(layer.get("points"))
        index = (None if is_points
                 else np.asarray(layer.get("index", []), dtype=np.uint32).reshape(-1))
        if pos.shape[0] == 0 or (index is not None and index.size == 0):
            continue

        a_pos = len(accessors)
        v = add_view(pos.tobytes(), 34962)
        accessors.append({"bufferView": v, "componentType": 5126, "count": int(pos.shape[0]),
                          "type": "VEC3", "min": pos.min(axis=0).tolist(),
                          "max": pos.max(axis=0).tolist()})
        attrs = {"POSITION": a_pos}
        if is_points:
            # 点云层：颜色直接来自照片像素（COLOR_0，VEC4 带 alpha），不需要法线与 UV。
            # 图像采样点本来就是"每个像素一个点"，平色点已等价于照片本身。
            col = np.asarray(layer["color"], dtype=np.float32).reshape(-1, 4)
            a_col = len(accessors)
            v = add_view(col.tobytes(), 34962)
            accessors.append({"bufferView": v, "componentType": 5126,
                              "count": int(col.shape[0]), "type": "VEC4"})
            attrs["COLOR_0"] = a_col
            # 自定义属性（glTF 允许 `_` 前缀）：每点的原始距离，渲染时用来算点尺寸
            sz = np.asarray(layer.get("size", np.ones(col.shape[0])), dtype=np.float32).reshape(-1, 1)
            a_sz = len(accessors)
            v = add_view(sz.tobytes(), 34962)
            accessors.append({"bufferView": v, "componentType": 5126,
                              "count": int(sz.shape[0]), "type": "SCALAR"})
            attrs["_SIZE"] = a_sz
        else:
            nrm = np.asarray(layer["nrm"], dtype=np.float32).reshape(-1, 3)
            uv = np.asarray(layer["uv"], dtype=np.float32).reshape(-1, 2)
            a_nrm = len(accessors)
            v = add_view(nrm.tobytes(), 34962)
            accessors.append({"bufferView": v, "componentType": 5126,
                              "count": int(nrm.shape[0]), "type": "VEC3"})
            a_uv = len(accessors)
            v = add_view(uv.tobytes(), 34962)
            accessors.append({"bufferView": v, "componentType": 5126,
                              "count": int(uv.shape[0]), "type": "VEC2"})
            attrs["NORMAL"] = a_nrm
            attrs["TEXCOORD_0"] = a_uv
            a_idx = len(accessors)
            v = add_view(index.tobytes(), 34963)
            accessors.append({"bufferView": v, "componentType": 5125,
                              "count": int(index.shape[0]), "type": "SCALAR"})

        prim = {"attributes": attrs, "material": 0, "mode": 0 if is_points else 4}
        if index is not None:
            prim["indices"] = a_idx
        meshes.append({"name": layer.get("name", "layer%d" % li), "primitives": [prim]})
        nodes.append({"mesh": len(meshes) - 1, "name": layer.get("name", "layer%d" % li)})
        total_verts += int(pos.shape[0])
        total_tris += 0 if index is None else int(index.size // 3)

    # T8：只要存在"远景/背景"层（名字里带 backdrop/far/bg），就把材质标成双面。
    layers_double_sided = any(
        any(k in str(lay.get("name", "")).lower() for k in ("backdrop", "far", "bg"))
        for lay in layers)

    v_img = add_view(texture_jpeg)

    gltf = {
        "asset": {"version": "2.0", "generator": "photoworld-pipeline"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes))), "name": name}],
        "nodes": nodes,
        "meshes": meshes,
        # T8：远景/背景层必须双面，否则从侧面或背面看会整片消失
        #   （视锥外延拓 + 自由视角一转身就露洞）。主体层保持单面。
        "materials": [{
            "name": "photo",
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                     "metallicFactor": 0.0, "roughnessFactor": 0.95},
            "doubleSided": bool(layers_double_sided),
        }],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": v_img, "mimeType": "image/jpeg"}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }
    # 相机内参必须随资产走：前端 GLB 路径原本是按固定 66° 视场渲染的，
    # 而几何是按照片真实视场建的 —— 两者不一致时整幅重建被放大并错位，
    # 画面底部露出被拉长的边缘几何（实测的"底部尖刺带"就是这么来的）。
    if extras:
        gltf["asset"]["extras"] = extras

    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = b"".join(blobs)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)

    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
        f.write(json_chunk)
        f.write(struct.pack("<II", len(bin_chunk), 0x004E4942))
        f.write(bin_chunk)

    return {"bytes": total, "vertices": total_verts, "triangles": total_tris,
            "layers": len(nodes), "texture_bytes": len(texture_jpeg)}
