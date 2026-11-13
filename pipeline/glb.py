"""极简 glTF 2.0 / GLB 写出器（不依赖 trimesh）。

只支持：一个 mesh、一个 PBR 材质、内嵌 JPEG 贴图、可选 Draco 不做（体积靠降采样控制）。
"""
from __future__ import annotations

import json
import struct

import numpy as np


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(path: str, layers: list, texture_jpeg: bytes, name: str = "world") -> dict:
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
        nrm = np.asarray(layer["nrm"], dtype=np.float32).reshape(-1, 3)
        uv = np.asarray(layer["uv"], dtype=np.float32).reshape(-1, 2)
        index = np.asarray(layer["index"], dtype=np.uint32).reshape(-1)
        if pos.shape[0] == 0 or index.size == 0:
            continue

        a_pos = len(accessors)
        v = add_view(pos.tobytes(), 34962)
        accessors.append({"bufferView": v, "componentType": 5126, "count": int(pos.shape[0]),
                          "type": "VEC3", "min": pos.min(axis=0).tolist(),
                          "max": pos.max(axis=0).tolist()})
        a_nrm = len(accessors)
        v = add_view(nrm.tobytes(), 34962)
        accessors.append({"bufferView": v, "componentType": 5126,
                          "count": int(nrm.shape[0]), "type": "VEC3"})
        a_uv = len(accessors)
        v = add_view(uv.tobytes(), 34962)
        accessors.append({"bufferView": v, "componentType": 5126,
                          "count": int(uv.shape[0]), "type": "VEC2"})
        a_idx = len(accessors)
        v = add_view(index.tobytes(), 34963)
        accessors.append({"bufferView": v, "componentType": 5125,
                          "count": int(index.shape[0]), "type": "SCALAR"})

        meshes.append({"name": layer.get("name", "layer%d" % li),
                       "primitives": [{"attributes": {"POSITION": a_pos, "NORMAL": a_nrm,
                                                      "TEXCOORD_0": a_uv},
                                       "indices": a_idx, "material": 0, "mode": 4}]})
        nodes.append({"mesh": len(meshes) - 1, "name": layer.get("name", "layer%d" % li)})
        total_verts += int(pos.shape[0])
        total_tris += int(index.size // 3)

    v_img = add_view(texture_jpeg)

    gltf = {
        "asset": {"version": "2.0", "generator": "photoworld-pipeline"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes))), "name": name}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [{
            "name": "photo",
            "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                     "metallicFactor": 0.0, "roughnessFactor": 0.95},
            "doubleSided": False,
        }],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": v_img, "mimeType": "image/jpeg"}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": offset}],
    }

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
