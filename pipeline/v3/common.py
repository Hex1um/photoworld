"""v3 公共设施：相机模型、色彩空间、图像 IO、GLB 打包。

与 v1/v2 的关键差异（这次重写的第一条硬约定）
------------------------------------------------
世界坐标与**前端渲染器逐字一致**：y 轴向上、地面 y=0、相机在 (0, eye, 0)、
光轴朝 **-Z**。v1 的 `geom.Camera.rays()` 用 +Z 朝前，而 `js/glbworld.js` 与
WebGL 相机用 -Z 朝前，两套符号在跨模块拼接时反复出错（CHANGELOG 里
"底部尖刺带""整幅重建被放大并错位"都属于这一类）。v3 只保留一种。

第二条硬约定：**任何几何位移只允许沿相机射线**。
沿射线缩放不改变顶点在"原点相机"下的投影，所以把地面压到 y=0、把远山推远，
都不会破坏"从原点看与原图相同"。v3 的所有正则化都建立在这个不变量上，
它把"好看"和"像原图"从互相拉扯变成互不干扰。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageOps

# pipeline/ 与仓库根都进 sys.path：v2 的标定与 GLB 写出器已经过大量实测，
# 直接复用比抄一份更安全（ISSUES 附录 A 的教训就是"同一类修复要 grep 全仓库"）。
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (_ROOT, os.path.join(_ROOT, "pipeline"), os.path.join(_ROOT, "pipeline", "v2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EYE_DEFAULT = 1.7          # 相机眼高（米），与前端 player.eyeHeight 同源
WHITE_STRIP_PX = 8         # 贴图底部预留的纯白条：给"顶点色层"取 1.0 的乘子
DEFAULT_HFOV_DEG = 60.0


# --------------------------------------------------------------------------- 相机

class Camera3:
    """针孔相机 + 俯仰。pitch > 0 = 光轴抬离水平面。

    世界系：地面 y=0，相机在 (0, eye, 0)，未俯仰时光轴指向 -Z。
    """

    def __init__(self, w: int, h: int, hfov: float, vfov: float | None = None,
                 pitch: float = 0.0, eye: float = EYE_DEFAULT):
        self.w, self.h = int(w), int(h)
        self.hfov = float(hfov)
        self.vfov = float(vfov) if vfov else 2.0 * math.atan(
            math.tan(hfov * 0.5) * self.h / self.w)
        self.pitch = float(pitch)
        self.eye = float(eye)

    # --- 派生量 ---
    @property
    def tan_h(self) -> float:
        return math.tan(self.hfov * 0.5)

    @property
    def tan_v(self) -> float:
        return math.tan(self.vfov * 0.5)

    @property
    def eye_pos(self) -> np.ndarray:
        return np.array([0.0, self.eye, 0.0], dtype=np.float32)

    def with_pitch(self, pitch: float) -> "Camera3":
        return Camera3(self.w, self.h, self.hfov, self.vfov, pitch, self.eye)

    # --- 采样 ---
    def rays(self, dtype=np.float32) -> np.ndarray:
        """(H,W,3) 单位视线（世界系，含俯仰）。像素中心采样。"""
        u = (np.arange(self.w) + 0.5) / self.w
        v = (np.arange(self.h) + 0.5) / self.h
        dx = (u - 0.5) * 2.0 * self.tan_h
        dy = (0.5 - v) * 2.0 * self.tan_v
        X, Y = np.meshgrid(dx.astype(np.float64), dy.astype(np.float64))
        d = np.stack([X, Y, -np.ones_like(X)], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        if abs(self.pitch) > 1e-9:
            c, s = math.cos(self.pitch), math.sin(self.pitch)
            x, y, z = d[..., 0], d[..., 1], d[..., 2]
            d = np.stack([x, y * c - z * s, y * s + z * c], axis=-1)
        return d.astype(dtype)

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """世界点 → (像素坐标 (N,2), 深度 (N,))。深度 <=0 表示在相机背后。"""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        d = P - self.eye_pos.astype(np.float64)
        if abs(self.pitch) > 1e-9:
            c, s = math.cos(self.pitch), math.sin(self.pitch)
            x, y, z = d[:, 0], d[:, 1], d[:, 2]
            d = np.stack([x, y * c + z * s, -y * s + z * c], axis=1)
        depth = -d[:, 2]
        safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        u = (d[:, 0] / (safe * self.tan_h)) * 0.5 + 0.5
        v = 0.5 - (d[:, 1] / (safe * self.tan_v)) * 0.5
        uv = np.stack([u * self.w - 0.5, v * self.h - 0.5], axis=1)
        return uv, depth

    def fov_meta(self) -> dict:
        return {"hfov": self.hfov, "vfov": self.vfov, "w": self.w, "h": self.h,
                "pitch": self.pitch, "eye_height": self.eye}


# --------------------------------------------------------------------------- 色彩

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


# --------------------------------------------------------------------------- 图像 IO

def load_photo(path: str) -> tuple[np.ndarray, dict]:
    """读图 → (H,W,3) float32 sRGB 0..1，EXIF 方向已纠正，附元信息。"""
    img = Image.open(path)
    exif = {}
    try:
        exif = {k: v for k, v in (img.getexif() or {}).items()}
    except Exception:
        pass
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    meta = {"path": path, "w": img.width, "h": img.height, "exif": exif}
    meta["hfov_deg"], meta["fov_source"] = hfov_from_exif(exif, img.width, img.height)
    return arr, meta


def hfov_from_exif(exif: dict, w: int, h: int) -> tuple[float, str]:
    """EXIF 焦距 → 水平视场。取不到就用 60° 的经验值（与前端默认一致）。"""
    try:
        f35 = exif.get(0xA405)          # FocalLengthIn35mmFilm
        if f35:
            # 35mm 画幅宽 36mm → hfov = 2·atan(18/f)
            return math.degrees(2 * math.atan(18.0 / float(f35))), "exif-35mm"
        f = exif.get(0x920A)            # FocalLength（mm，实际焦距）
        unit = exif.get(0xA210)         # FocalPlaneResolutionUnit
        resx = exif.get(0xA20E)
        if f and resx and unit:
            # 焦距像素数 = f(mm) / 25.4 * 每毫米像素数
            per_mm = float(resx) / (25.4 if unit == 3 else 10.0)
            fpx = float(f) * per_mm
            return math.degrees(2 * math.atan(w / (2.0 * fpx))), "exif-focal-plane"
    except Exception:
        pass
    return DEFAULT_HFOV_DEG, "default"


def save_rgb(path: str, rgb: np.ndarray, quality: int = 92) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    u8 = (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8).save(path, quality=quality)


def save_gray(path: str, g: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    g = np.asarray(g, dtype=np.float32)
    lo, hi = float(np.nanmin(g)), float(np.nanmax(g))
    n = (g - lo) / max(hi - lo, 1e-6)
    Image.fromarray((n * 255).astype(np.uint8)).save(path)


# --------------------------------------------------------------------------- 贴图打包

def pack_texture(rgb: np.ndarray, quality: int = 90) -> tuple[bytes, float]:
    """照片 + 底部纯白条 → JPEG 字节流，返回 (bytes, uv_v_scale)。

    **为什么要有白条**：GLB 里只有一张贴图，而 v3 的视锥外地形/地物是顶点色的。
    渲染器 mode 4 的公式是 `贴图 × 顶点色`，顶点色层只要把 UV 指到白条上，
    乘积就等于顶点色本身 —— 这样不需要改前端一行代码就能让两类图层共存。
    uv_v_scale 是照片区的 v 上限（照片 UV 必须乘它，否则会采到白条）。
    """
    import io as _io
    h, w = rgb.shape[:2]
    canvas = np.ones((h + WHITE_STRIP_PX, w, 3), dtype=np.float32)
    canvas[:h] = rgb
    buf = _io.BytesIO()
    Image.fromarray((np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8)).save(
        buf, format="JPEG", quality=quality, subsampling=1)
    return buf.getvalue(), h / float(h + WHITE_STRIP_PX)


def white_uv(uv_v_scale: float) -> tuple[float, float]:
    """白条中心的 UV（给顶点色层用）。"""
    return 0.5, (uv_v_scale + 1.0) * 0.5


# --------------------------------------------------------------------------- GLB

def write_world_glb(path: str, layers: list, tex_jpeg: bytes, extras: dict,
                    name: str = "world", compact: bool = True) -> dict:
    """把图层列表写成 GLB。复用 pipeline/glb.py 的写出器（已验证过对齐/双面等坑）。

    每个 layer: {"name","pos","nrm","uv","color"(可选),"index"}
    compact=True（默认）把 NORMAL/UV/COLOR 写成规范化整数、索引尽量用 uint16，
    只省字节不改几何；--no-compact 可退回全 float32 以便对比。
    """
    import glb as glbmod
    return glbmod.write_glb(path, layers, tex_jpeg, name=name, extras=extras,
                            compact=compact)


# --------------------------------------------------------------------------- 小工具

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def smoothstep(a: float, b: float, x):
    t = np.clip((np.asarray(x, dtype=np.float32) - a) / max(b - a, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _jsonable(o):
    """numpy 标量/数组 → Python 原生（标定结果里混着这两种，直接 dump 会炸）。"""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<ndarray {o.shape}>"
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def json_dump(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_jsonable)
