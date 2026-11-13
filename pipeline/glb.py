"""极简 glTF 2.0 / GLB 写出器（不依赖 trimesh）。

只支持：一个 mesh、一个 PBR 材质、内嵌 JPEG 贴图、可选 Draco 不做（体积靠降采样控制）。
"""
from __future__ import annotations

import json
import struct

import numpy as np


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(path: str, pos: np.ndarray, nrm: np.ndarray, uv: np.ndarray,
              index: np.ndarray, texture_jpeg: bytes, name: str = "world") -> dict:
    pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
    nrm = np.asarray(nrm, dtype=np.float32).reshape(-1, 3)
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    index = np.asarray(index, dtype=np.uint32).reshape(-1)

    blobs: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []
    offset = 0

    def add_view(data: bytes, target: int | None = None) -> int:
        nonlocal offset
        blobs.append(_pad4(data))
        v = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        offset += len(_pad4(data))
        return len(views) - 1

    # POSITION
    v_pos = add_view(pos.tobytes(), 34962)
    accessors.append({
        "bufferView": v_pos, "componentType": 5126, "count": int(pos.shape[0]),
        "type": "VEC3",
        "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist(),
    })
    # NORMAL
    v_nrm = add_view(nrm.tobytes(), 34962)
    accessors.append({"bufferView": v_nrm, "componentType": 5126,
                      "count": int(nrm.shape[0]), "type": "VEC3"})
    # TEXCOORD_0
    v_uv = add_view(uv.tobytes(), 34962)
    accessors.append({"bufferView": v_uv, "componentType": 5126,
                      "count": int(uv.shape[0]), "type": "VEC2"})
    # indices
    v_idx = add_view(index.tobytes(), 34963)
    accessors.append({"bufferView": v_idx, "componentType": 5125,
                      "count": int(index.shape[0]), "type": "SCALAR"})
    # texture
    v_img = add_view(texture_jpeg)

    gltf = {
        "asset": {"version": "2.0", "generator": "photoworld-pipeline"},
        "scene": 0,
        "scenes": [{"nodes": [0], "name": name}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                "indices": 3,
                "material": 0,
                "mode": 4,
            }],
        }],
        "materials": [{
            "name": "photo",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.95,
            },
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

    return {"bytes": total, "vertices": int(pos.shape[0]), "triangles": int(index.size // 3),
            "texture_bytes": len(texture_jpeg)}
