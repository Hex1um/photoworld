"""单体物体的三维补全（单图 → 多面网格）。

前端默认用「十字账单牌」保证不拉伸；本模块是质量升级路径：
对近景显著单体（树/房屋/人/车）调用单图转 3D 模型，返回 GLB，
前端用真网格替换账单牌，从而侧面/背面也合理。

优先顺序（均为高质量开源）：
  1. InstantMesh  https://github.com/TencentARC/InstantMesh  （腾讯 ARC，质量最好）
  2. TripoSR      https://github.com/VAST-AI-Research/TripoSR （快速，单卡可跑）
  3. SF3D         https://github.com/Stability-AI/generative-models (Stable Fast 3D)
均为可选依赖：未安装时 available() 返回 False，/api/object3d 返回 501，
前端保持账单牌，不影响主流程。
"""
from __future__ import annotations

import io
import os
from typing import Optional

_BACKEND = None


def available() -> bool:
    return _detect() is not None


def backend_name() -> str:
    b = _detect()
    return b[0] if b else "none"


def _detect():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    # InstantMesh
    try:
        import importlib
        importlib.import_module("instantmesh")
        _BACKEND = ("instantmesh", None)
        return _BACKEND
    except Exception:
        pass
    # TripoSR
    try:
        import importlib
        importlib.import_module("tsr")
        _BACKEND = ("triposr", None)
        return _BACKEND
    except Exception:
        pass
    _BACKEND = None
    return None


def generate(image_bytes: bytes, bbox: Optional[list] = None) -> tuple[Optional[bytes], str]:
    """单图 → GLB 字节。失败返回 (None, reason)。

    真实接入时需要：
      1) 安装对应仓库与权重（见 README）；
      2) 用 PIL 解码 image_bytes，按 bbox 裁出单体；
      3) 调模型前向，导出 GLB（InstantMesh: export_mesh / TripoSR: model.extract_mesh）。
    这里保留最小骨架，避免把未验证的推理代码伪装成可用。
    """
    b = _detect()
    if not b:
        return None, "no_backend"
    name = b[0]
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        if bbox and len(bbox) == 4:
            w, h = img.size
            x, y, bw, bh = bbox
            img = img.crop((int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)))
        # 预留：按 name 调对应模型，输出 GLB bytes
        _ = (name, img)
        raise RuntimeError("backend installed but inference not wired")
    except Exception as e:
        return None, str(e)
