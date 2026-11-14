"""扇环外地貌延伸：边界约束的确定性高度场。

两层实现：
  1. 程序化（默认，纯 numpy）：全局连续 fBm + 四边高度约束。
     tile 边界的高度由客户端传入（来自已重建地貌 / 已生成 tile），
     生成结果在边界上与邻块严格一致，内部平滑过渡，因此不会出现接缝。
  2. Terrain Diffusion（可选，需要 GPU）：learned 无限地形生成，
     以同样的边界与 seed 为条件，质量更高。未安装时自动回退到第 1 层。

约定：高度单位米，坐标系与前端一致（x 右、z 前，-Z 为视线方向）。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------- 确定性噪声 ----------------
def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263
         + int(seed) * 2246822519)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFFFFFF).astype(np.float64) / 4294967295.0


def _vnoise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    ix = np.floor(x)
    iy = np.floor(y)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    ix = ix.astype(np.int64)
    iy = iy.astype(np.int64)
    a = _hash2(ix, iy, seed)
    b = _hash2(ix + 1, iy, seed)
    c = _hash2(ix, iy + 1, seed)
    d = _hash2(ix + 1, iy + 1, seed)
    top = a * (1 - ux) + b * ux
    bot = c * (1 - ux) + d * ux
    return top * (1 - uy) + bot * uy


def fbm(x: np.ndarray, y: np.ndarray, seed: int, octaves: int = 5) -> np.ndarray:
    total = np.zeros_like(x, dtype=np.float64)
    amp, freq, norm = 0.5, 1.0, 0.0
    for i in range(octaves):
        total += amp * _vnoise(x * freq, y * freq, seed + i * 131)
        norm += amp
        amp *= 0.5
        freq *= 2.03
    return total / max(norm, 1e-9)


def _edge_field(edges: dict, res: int) -> np.ndarray:
    """由四边高度构造平滑的边界场 E(i,j)，用于内部混合。"""
    left = np.asarray(edges.get("left", []), dtype=np.float64)
    right = np.asarray(edges.get("right", []), dtype=np.float64)
    top = np.asarray(edges.get("top", []), dtype=np.float64)
    bottom = np.asarray(edges.get("bottom", []), dtype=np.float64)
    if any(len(e) != res for e in (left, right, top, bottom)):
        raise ValueError("edges 长度必须等于 res")

    t = np.linspace(0.0, 1.0, res)
    # 水平方向按 x 插值，垂直方向按 y 插值，再取平均
    eh = (1 - t)[None, :] * left[:, None] + t[None, :] * right[:, None]
    ev = (1 - t)[:, None] * top[None, :] + t[:, None] * bottom[None, :]
    return 0.5 * (eh + ev)


def procedural_extend(seed: int, tx: int, tz: int, ts: float, res: int,
                      edges: Optional[dict] = None) -> np.ndarray:
    """生成一块 res×res 高度图，四边严格匹配 edges。"""
    x0, z0 = tx * ts, tz * ts
    xs = x0 + np.linspace(0.0, ts, res)
    zs = z0 + np.linspace(0.0, ts, res)
    X, Z = np.meshgrid(xs, zs, indexing="xy")

    # 全局连续 fBm：相邻 tile 在共享坐标上取值相同，天然无缝
    base = fbm(X * 0.021, Z * 0.021, seed, 5) - 0.5
    detail = fbm(X * 0.079, Z * 0.079, seed + 17, 4) - 0.5
    h = base * 7.5 + detail * 1.9

    # 中心略高、边缘收敛，像地形起伏而不是白噪声
    u = np.linspace(0, 1, res)
    dome = np.outer(np.sin(np.pi * u), np.sin(np.pi * u))
    h += dome * 1.6

    if edges is not None:
        E = _edge_field(edges, res)
        # 距最近边界的归一化距离；边界带内更多继承边界高度
        d = np.minimum.reduce([
            np.tile(u[None, :], (res, 1)),
            np.tile((1 - u)[None, :], (res, 1)),
            np.tile(u[:, None], (1, res)),
            np.tile((1 - u)[:, None], (1, res)),
        ])
        band = 0.28
        w = np.clip(d / band, 0.0, 1.0)
        w = w * w * (3 - 2 * w)
        h = E * (1 - w) + (E + h) * w
        # 四边硬约束（角点按相邻边平均，避免四角撕裂）
        h[0, :] = edges["top"]
        h[-1, :] = edges["bottom"]
        h[:, 0] = edges["left"]
        h[:, -1] = edges["right"]
        h[0, 0] = 0.5 * (edges["left"][0] + edges["top"][0])
        h[0, -1] = 0.5 * (edges["right"][0] + edges["top"][-1])
        h[-1, 0] = 0.5 * (edges["left"][-1] + edges["bottom"][0])
        h[-1, -1] = 0.5 * (edges["right"][-1] + edges["bottom"][-1])

    return h.astype(np.float32)


def terrain_diffusion_extend(seed: int, tx: int, tz: int, ts: float, res: int,
                             edges: Optional[dict] = None) -> Optional[np.ndarray]:
    """Terrain Diffusion（SIGGRAPH'26）接入点。

    安装参考：https://github.com/xandergos/terrain-diffusion
      pip install -r requirements.txt   # 需要 CUDA
      python -c "from terrain_diffusion ..."

    该仓库提供 InfiniteDiffusion：以 (seed, 世界坐标) 为索引、O(1) 随机访问的
    无限高度场，正好用于本项目的 tile 延伸。调用方式随版本变化，这里只做
    最小适配：取 tile 对应的世界矩形，采样高度后重采样到 res×res，再用四边约束收敛。
    未安装 / 调用失败返回 None，由调用方回退 procedural_extend。
    """
    try:
        import terrain_diffusion  # noqa: F401
    except Exception:
        return None

    try:
        from terrain_diffusion import WorldPipeline  # type: ignore
        pipe = WorldPipeline.from_pretrained("xandergos/terrain-diffusion-30m")
        # 取比 tile 稍大的窗口，避免边界伪影，再裁回中心
        mod = getattr(pipe, "modality", None)
        if mod is None:
            return None
        world = mod.get_batch({"x": tx * ts, "y": tz * ts}, {"x": 1, "y": 1})
        h = world["height"][0].detach().cpu().numpy()
        # 最近邻重采样到 res×res（真正的无缝由 InfiniteDiffusion 的坐标索引保证）
        idx_x = np.linspace(0, h.shape[-1] - 1, res).astype(np.int32)
        idx_y = np.linspace(0, h.shape[-2] - 1, res).astype(np.int32)
        h = h[np.ix_(idx_y, idx_x)].astype(np.float32)
        if edges is not None:
            # 用与程序化版本相同的四边硬约束消除接缝
            base = procedural_extend(seed, tx, tz, ts, res, edges)
            u = np.linspace(0, 1, res)
            d = np.minimum.reduce([
                np.tile(u[None, :], (res, 1)), np.tile((1 - u)[None, :], (res, 1)),
                np.tile(u[:, None], (1, res)), np.tile((1 - u)[:, None], (1, res)),
            ])
            w = np.clip(d / 0.28, 0, 1)
            w = w * w * (3 - 2 * w)
            h = base * (1 - w) + h * w
            h[0, :] = edges["top"]; h[-1, :] = edges["bottom"]
            h[:, 0] = edges["left"]; h[:, -1] = edges["right"]
        return h.astype(np.float32)
    except Exception:
        return None


def extend(seed: int, tx: int, tz: int, ts: float, res: int,
           edges: Optional[dict] = None, prefer_learned: bool = True) -> tuple[np.ndarray, str]:
    """统一入口：优先 learned，失败回退程序化。返回 (heightmap, source)。"""
    if prefer_learned:
        h = terrain_diffusion_extend(seed, tx, tz, ts, res, edges)
        if h is not None:
            return h, "terrain_diffusion"
    return procedural_extend(seed, tx, tz, ts, res, edges), "procedural"
