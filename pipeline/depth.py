"""深度后端：可插拔，CPU 也能跑。

后端优先级（由 --backend 控制）：
  onnx    Depth-Anything-V2-Small（ONNX, 相对深度）—— 现在就能跑，CPU 数秒
  depthpro  Depth Pro（度量深度，需 torch + 权重）—— 装了才可用
  unik3d    UniK3D（度量深度，需 torch + 权重）
  vggt      VGGT（多视图，直接出位姿+点云）—— 双图首选

相对深度后端必须经过 geom.py 的**地面平面标定**换算成米，
不再使用任何"魔法曲线"。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image


class BackendUnavailable(RuntimeError):
    pass


@dataclass
class DepthResult:
    depth: np.ndarray       # (H, W) float32，值越大越**近**
    metric: bool            # True 表示单位是米
    backend: str
    note: str = ""


def _pad_to_square(img: Image.Image, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """等比缩放 + padding 到正方形，避免拉伸（旧实现直接 resize 是错的）。"""
    w, h = img.size
    s = size / float(max(w, h))
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    ox = (size - nw) // 2
    oy = (size - nh) // 2
    canvas.paste(resized, (ox, oy))
    return np.asarray(canvas, dtype=np.float32) / 255.0, (ox, oy, nw, nh)


class OnnxDepthAnything:
    name = "onnx-depth-anything-v2-small"
    metric = False

    def __init__(self, model_path: str, size: int = 518):
        import onnxruntime as ort  # 延迟导入

        if not os.path.exists(model_path):
            raise BackendUnavailable("找不到模型：" + model_path)
        self.size = size
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def __call__(self, img: Image.Image) -> DepthResult:
        size = self.size
        padded, (ox, oy, nw, nh) = _pad_to_square(img, size)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (padded - mean) / std
        x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
        out = self.sess.run([self.out_name], {self.in_name: x})[0]
        d = np.squeeze(out).astype(np.float32)

        # 裁掉 padding（等比，不拉伸）
        d = d[oy:oy + nh, ox:ox + nw]
        d = np.asarray(Image.fromarray(d, mode="F").resize(img.size, Image.BICUBIC), dtype=np.float32)

        # 极性：天空应在远端。用顶部/底部条带的均值判断并翻转
        top = d[: max(1, d.shape[0] // 8)].mean()
        bot = d[-max(1, d.shape[0] // 8):].mean()
        inverted = top > bot
        if inverted:
            d = -d
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        d = np.clip((d - lo) / max(1e-6, hi - lo), 0.0, 1.0)
        # 统一成"越大越近"：风景照里近景在下 ⇒ 行均值应随行号递增
        corr = np.corrcoef(d.mean(axis=1), np.arange(d.shape[0]))[0, 1]
        if corr < 0:
            d = 1.0 - d
        return DepthResult(depth=d, metric=False, backend=self.name,
                           note="relative-depth; scaled by ground-plane fit")


class _TorchBackend:
    """Depth Pro / UniK3D / VGGT 的公共占位：缺依赖时给出明确指引而不是崩。"""

    def _need(self, pkg: str, hint: str):
        try:
            __import__(pkg)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(
                f"{self.name} 需要 {pkg}（{hint}）。\n"
                f"安装后重试；否则用 --backend onnx 走 CPU 路径。"
            ) from exc


class DepthProBackend(_TorchBackend):
    name = "depth-pro"

    def __init__(self, weights: str | None = None):
        self._need("torch", "pip install torch --index-url https://download.pytorch.org/whl/cu124")
        self._need("depth_pro", "参考 pipeline/environment.yml")
        self.weights = weights

    def __call__(self, img: Image.Image) -> DepthResult:
        raise BackendUnavailable("Depth Pro 权重未安装：请把 ml-depth-pro 权重放到 pipeline/weights/")


class VGGTBackend(_TorchBackend):
    name = "vggt"

    def __init__(self):
        self._need("torch", "pip install torch")
        self._need("vggt", "参考 pipeline/environment.yml（facebookresearch/vggt）")

    def __call__(self, images):
        raise BackendUnavailable("VGGT 权重未安装：请把 facebook/VGGT-1B 放到 pipeline/weights/")


def build_backend(name: str, model_path: str, size: int = 518):
    if name == "onnx":
        return OnnxDepthAnything(model_path, size=size)
    if name == "depthpro":
        return DepthProBackend()
    if name == "vggt":
        return VGGTBackend()
    raise BackendUnavailable("未知后端：" + name)
