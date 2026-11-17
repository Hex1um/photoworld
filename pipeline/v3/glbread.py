"""GLB 读取：把打包好的 world.glb 读回内存，用于"验证交付物"而不是"验证内存"。

v2 的教训（ISSUES §25）：指标跑在管线内部的对象上，一旦打包环节出错
（UV 缩放、白条、层名拼错），指标全绿但前端是坏的。这里刻意走文件。
"""
from __future__ import annotations

import json
import struct

import numpy as np


def load(path: str) -> dict:
    with open(path, "rb") as f:
        buf = f.read()
    magic, ver, total = struct.unpack_from("<III", buf, 0)
    assert magic == 0x46546C67, "不是 GLB"
    off, js, bin_ = 12, None, None
    while off < total:
        ln, typ = struct.unpack_from("<II", buf, off)
        chunk = buf[off + 8: off + 8 + ln]
        if typ == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif typ == 0x004E4942:
            bin_ = chunk
        off += 8 + ln
    assert js is not None and bin_ is not None

    def acc(i):
        a = js["accessors"][i]
        v = js["bufferViews"][a["bufferView"]]
        start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
        ct = {5126: np.float32, 5125: np.uint32, 5123: np.uint16,
              5122: np.int16, 5121: np.uint8}[a["componentType"]]
        arr = np.frombuffer(bin_, dtype=ct, count=a["count"] * n, offset=start)
        if a.get("normalized"):
            # glTF 规范化整数：POSITION/NORMAL=INT16→[-1,1]，UV=UINT16、COLOR=UINT8→[0,1]
            arr = arr.astype(np.float64) / {
                5122: 32767.0, 5123: 65535.0, 5121: 255.0}[a["componentType"]]
        return arr.reshape(-1, n) if n > 1 else arr

    layers = []
    for m in js["meshes"]:
        p = m["primitives"][0]
        layers.append({
            "name": m.get("name", ""),
            "pos": acc(p["attributes"]["POSITION"]).astype(np.float64).reshape(-1, 3),
            "uv": acc(p["attributes"]["TEXCOORD_0"]).astype(np.float64).reshape(-1, 2)
            if "TEXCOORD_0" in p["attributes"] else None,
            "color": acc(p["attributes"]["COLOR_0"]).astype(np.float64).reshape(-1, 4)
            if "COLOR_0" in p["attributes"] else None,
            "index": acc(p["indices"]).astype(np.int64).reshape(-1, 3)
            if "indices" in p else None,
        })
    img = None
    if js.get("images"):
        bv = js["bufferViews"][js["images"][0]["bufferView"]]
        img = bin_[bv.get("byteOffset", 0): bv.get("byteOffset", 0) + bv["byteLength"]]
    return {"layers": layers, "extras": js.get("asset", {}).get("extras", {}),
            "jpeg": img, "json": js}
