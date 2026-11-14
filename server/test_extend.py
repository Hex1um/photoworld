"""延伸高度场的自检：四边硬约束 + 邻块接缝 + 确定性。

    python server/test_extend.py

不依赖 fastapi / torch，只用 numpy —— 这样在没有 GPU 依赖的环境里也能验证
server/terrain.py 的核心算法（HTTP 那层是薄包装）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import terrain  # noqa: E402

RES, TS, SEED = 17, 32.0, 7


def H(x, z):
    """假想的「客户端已知道的高度场」：连续、可解析，用来当四边条件。"""
    return 0.9 * np.sin(x * 0.045) + 0.6 * np.cos(z * 0.06) + 0.02 * z


def edge_samples(tx, tz):
    """与 js/extend.js 的 fetchServerHeights 完全一致的采样方式。"""
    x0, z0 = tx * TS, tz * TS
    t = np.linspace(0.0, 1.0, RES)
    return {
        "left": H(x0, z0 + t * TS).tolist(),
        "right": H(x0 + TS, z0 + t * TS).tolist(),
        "top": H(x0 + t * TS, z0).tolist(),
        "bottom": H(x0 + t * TS, z0 + TS).tolist(),
    }


def main() -> int:
    tx, tz = 3, -4
    a = terrain.procedural_extend(SEED, tx, tz, TS, RES, edge_samples(tx, tz))
    b = terrain.procedural_extend(SEED, tx + 1, tz, TS, RES, edge_samples(tx + 1, tz))
    c = terrain.procedural_extend(SEED, tx, tz + 1, TS, RES, edge_samples(tx, tz + 1))
    e = edge_samples(tx, tz)
    ok = True

    def check(name, value, limit=1e-5):
        nonlocal ok
        good = value <= limit
        ok = ok and good
        print(("PASS  " if good else "FAIL  ") + f"{name}（{value:.6f} ≤ {limit}）")

    print(f"形状 {a.shape} dtype {a.dtype} 值域 {a.min():.2f}..{a.max():.2f}")
    # 角点按「相邻两边平均」处理，是刻意的（否则四角会撕裂），所以不含角点比较
    check("左边硬约束", float(np.abs(a[1:-1, 0] - np.array(e["left"])[1:-1]).max()))
    check("右边硬约束", float(np.abs(a[1:-1, -1] - np.array(e["right"])[1:-1]).max()))
    check("上边硬约束", float(np.abs(a[0, 1:-1] - np.array(e["top"])[1:-1]).max()))
    check("下边硬约束", float(np.abs(a[-1, 1:-1] - np.array(e["bottom"])[1:-1]).max()))
    check("左右邻块接缝", float(np.abs(a[:, -1] - b[:, 0]).max()))
    check("上下邻块接缝", float(np.abs(a[-1, :] - c[0, :]).max()))
    check("确定性（同参数两次）",
          float(np.abs(a - terrain.procedural_extend(SEED, tx, tz, TS, RES, edge_samples(tx, tz))).max()))
    print("全部通过" if ok else "存在失败项")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
