#!/usr/bin/env python3
"""P2 回归：世界模型动作语义 / RoPE 等变性 / 初始化与 CFG / 预编码 latent。

全部纯 numpy + 源码结构断言，不需要 torch 或 GPU：

  python tools/test_regression_p2.py

覆盖 5.1~5.5 里可以脱离训练机验证的部分。训练全链路仍需在有 CUDA torch 的
机器上实跑；本脚本保证"语义定义"与"代码结构"不会再悄悄漂移回去。
"""
from __future__ import annotations

import ast
import importlib.util
import io
import json
import math
import os
import sys
import tarfile
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from worldmodel.action_codec import (  # noqa: E402
    ACT_DIM, DSL_NAMES, cam_to_world, dsl_action, parse_spec, rel_action,
)
from worldmodel.data.dataset import TarWindows, _stack  # noqa: E402
from worldmodel.model.rope import apply_rope_np, rope_angles  # noqa: E402


def _src(rel: str) -> str:
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


def test_action_codec() -> None:
    """5.1：DSL 与训练标签共用 rel_action，维度/符号/量纲一致。"""
    assert ACT_DIM == 7
    for name in DSL_NAMES:
        a = dsl_action(name)
        assert a.shape == (7,) and np.isfinite(a).all(), name
    fwd = dsl_action("forward")
    assert abs(fwd[5] + 1.0) < 1e-6 and np.allclose(fwd[:3], 0), \
        "forward 应是相机系 -z 平移（归一化后 -1），不是 a[3] 的 x 平移：%s" % fwd
    assert abs(dsl_action("back")[5] - 1.0) < 1e-6
    assert dsl_action("turn_left")[1] < -0.04, "turn_left 的轴角 y 应为负（旧实现符号是反的）"
    assert dsl_action("turn_right")[1] > 0.04
    assert dsl_action("look_up")[0] < -0.01, "look_up 应落在轴角 x 维，不是 z 平移"
    assert np.allclose(dsl_action("idle"), 0.0)
    assert parse_spec(["forward 20", ("turn_left", 3)]) == [("forward", 20), ("turn_left", 3)]

    # 与 render_traj.mjs::makePoses 同规则生成一段轨迹，比较标签均值
    s = 9301 + 49297

    def rnd():
        nonlocal s
        s = (s * 9301 + 49297) % 233280
        return s / 233280

    x = z = 0.0
    yaw, pitch = (rnd() - 0.5) * 0.5, (rnd() - 0.5) * 0.12
    labels = {}
    for _ in range(300):
        T0 = cam_to_world(x, 1.7, z, yaw, pitch)
        r, act = rnd(), "idle"
        if r < 0.45:
            act = "forward"
            d = 0.06 + rnd() * 0.06
            x -= math.sin(yaw) * d
            z -= math.cos(yaw) * d
        elif r < 0.65:
            act = "turn_left"
            yaw += 0.035 + rnd() * 0.03
        elif r < 0.85:
            act = "turn_right"
            yaw -= 0.035 + rnd() * 0.03
        elif r < 0.93:
            act = "look_up"
            pitch = min(0.45, pitch + 0.02)
        T1 = cam_to_world(x, 1.7, z, yaw, pitch)
        labels.setdefault(act, []).append(rel_action(T0, T1, 1.2, 1.2, 0.09))
    for name in ("turn_left", "turn_right", "look_up", "idle"):
        mean = np.mean(labels[name], axis=0)
        err = float(np.max(np.abs(mean - dsl_action(name))))
        assert err < 0.02, "%s 与合成轨迹标签偏离 %.4f（阈值 0.02）" % (name, err)
    # forward 的基准取 yaw=0（纯 -z）；训练轨迹里 yaw 会累积漂移，
    # 所以均值方向会分散到 x/z 上，但归一化后的**模长**必须仍为 1。
    mean_fwd = np.mean(labels["forward"], axis=0)
    assert abs(np.linalg.norm(fwd[3:6]) - 1.0) < 1e-6, "DSL 平移未按中位速度归一化"
    assert abs(np.linalg.norm(mean_fwd[3:6]) - 1.0) < 0.15, \
        "训练 forward 的平移模长 %.3f 与 DSL 不一致" % np.linalg.norm(mean_fwd[3:6])
    assert mean_fwd[5] < -0.5, "forward 的主导方向应仍是相机系 -z"

    # prepare_data.py 必须 import 同一个函数，不能各写一份
    spec = importlib.util.spec_from_file_location(
        "_pw_prepare_data", os.path.join(ROOT, "scripts", "prepare_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.rel_action is rel_action, "训练侧没有复用共享 action_codec"
    print("PASS  P2-5.1 动作同构：7 维 = 轴角3 + 平移3/speed + logFOV1；"
          "turn_left/look_up/idle 与合成轨迹标签最大偏差 < 0.02")


def _old_rope_angles(dim, n_t, gh, gw):
    """修复前的角度表（用于证明等变性单测确实能抓到这个 bug）。"""
    per = dim // 3
    per -= per % 2
    half = per // 2
    inv = 1.0 / (10000.0 ** (np.arange(half, dtype=np.float64) / half))

    def freqs(n):
        ang = np.outer(np.arange(n, dtype=np.float64), inv)
        return np.concatenate([ang, ang], axis=-1)

    ft, fy, fx = freqs(n_t), freqs(gh), freqs(gw)
    out = np.concatenate([
        np.broadcast_to(ft[:, None, None, :], (n_t, gh, gw, per)),
        np.broadcast_to(fy[None, :, None, :], (n_t, gh, gw, per)),
        np.broadcast_to(fx[None, None, :, :], (n_t, gh, gw, per)),
    ], axis=-1)
    pad = dim - out.shape[-1]
    if pad > 0:
        out = np.concatenate([out, np.zeros((n_t, gh, gw, pad))], axis=-1)
    return out


def _invariance_error(ang3) -> float:
    """整体平移 (Δt,Δy,Δx) 后，注意力分数应只依赖相对位移。"""
    rng = np.random.default_rng(7)
    q = rng.standard_normal(64)
    k = rng.standard_normal(64)
    a = ang3[1, 3, 5]
    b = ang3[2, 4, 6]
    c = ang3[3, 3, 5]
    d = ang3[4, 4, 6]
    assert np.allclose(a - b, c - d), "两组位置的角度差不相等，测试构造有误"
    s1 = float(np.dot(apply_rope_np(q, a), apply_rope_np(k, b)))
    s2 = float(np.dot(apply_rope_np(q, c), apply_rope_np(k, d)))
    return abs(s1 - s2)


def test_rope() -> None:
    """5.2：half-split 前后半频率配对，旋转等变性成立。"""
    dim, n_t, gh, gw = 64, 6, 16, 24
    ang = rope_angles(dim, n_t, gh, gw, extra=0).reshape(n_t, gh, gw, dim)
    assert np.allclose(ang[..., :dim // 2], ang[..., dim // 2:]), "前后半频率仍不配对"
    assert _invariance_error(ang) < 1e-9

    old = _old_rope_angles(dim, n_t, gh, gw)
    assert not np.allclose(old[..., :dim // 2], old[..., dim // 2:]), \
        "旧实现居然是配对的，单测基准失效"
    old_err = _invariance_error(old)
    assert old_err > 1e-3, "旧实现等变性误差不够大，单测抓不到回归：%.3g" % old_err

    with_extra = rope_angles(dim, n_t, gh, gw, extra=1)
    n = gh * gw
    assert with_extra.shape == (n_t * n + 1, dim)
    assert np.allclose(with_extra[n_t * n - n], with_extra[n_t * n - n + 1]), \
        "动作 token 应沿用目标帧起始位置"
    print("PASS  P2-5.2 RoPE：前后半频率严格配对，平移等变性误差 %.2g；"
          "旧实现误差 %.3f（单测可抓到回归）" % (_invariance_error(ang), old_err))


def test_structure() -> None:
    """5.3/5.4：初始化、CFG 路径、采样中点、EMA 验证的结构断言。"""
    dit = _src("worldmodel/model/dit.py")
    assert "nn.init.zeros_(self.head.weight)" not in dit, "输出头又回到零初始化"
    assert "self.mod.weight[lo:lo + cfg.dim].zero_()" in dit, "adaLN gate 没有零初始化"
    assert "nn.init.normal_(self.head.weight" in dit
    assert "rope_angles" in dit, "dit.py 没有使用配对后的角度表"

    tree = ast.parse(dit)
    sigs = [[a.arg for a in n.args.args]
            for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "forward"]
    assert any("uncond" in s and "drop_act" not in s and "drop_init" not in s for s in sigs), \
        "CausalWorldDiT.forward 没有统一的条件开关：%s" % sigs

    roll = _src("worldmodel/infer/rollout.py")
    assert "uncond=True" in roll, "CFG 无条件分支没走 uncond=True"
    assert "ab * 0" not in roll, "仍在用 action 置零冒充无条件"
    assert "a[1] = yaw" not in roll and "dsl_action" in roll, "推理没有使用共享动作编码器"
    assert "(s + 0.5) * dt" in roll, "Euler 仍是左端点"
    assert "parse_actions" in roll

    tr = _src("worldmodel/train/trainer.py")
    assert "uncond=uncond" in tr and "drop_act=" not in tr, "训练仍是两次独立 dropout"
    assert "ema.copy_to(model)" in tr, "验证循环仍用非 EMA 权重"
    assert 'if "z_init" in batch' in tr, "训练没有预编码 latent 分支"

    for f in ("configs/s1_base.yaml", "configs/smoke.yaml"):
        cfg = _src(f)
        assert "drop_cond:" in cfg and "drop_init:" not in cfg, "%s 未同步条件 dropout" % f
    print("PASS  P2-5.3/5.4 结构：head 常规初始化 + gate 零初始化；"
          "训练/推理共用 uncond；中点采样；val loss 用 EMA")


def test_eval_direction_sign() -> None:
    """5.1 的验收链：相位相关必须返回「b 相对 a」的位移，否则动作可控性会反判。

    eval.py 顶层 import torch，本机没有；这里用 AST 把 phase_shift 函数体单独
    编译出来执行，测的是仓库里的真实实现而非复刻版。
    """
    tree = ast.parse(_src("scripts/eval.py"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "phase_shift")
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"np": np}
    exec(compile(mod, "eval.py:phase_shift", "exec"), ns)
    phase_shift = ns["phase_shift"]

    rng = np.random.default_rng(0)
    a = (rng.random((64, 96, 3)) * 255).astype("uint8")
    for shift in (6, -6, 12):
        dx, dy = phase_shift(a, np.roll(a, shift, axis=1))
        assert abs(dx - shift) < 0.6, \
            "内容右移 %+d 像素却报 dx=%+.1f，方向判据会反判" % (shift, dx)
        assert abs(dy) < 0.6
    print("PASS  P2-5.1 验收链：相位相关方向符号正确（右移 +6 → dx=+6.0），"
          "turn_left/right 不会再被反判")


def test_preencoded_latents() -> None:
    """5.5：.latent 分片直读，形状与内容与离线写入一致。"""
    rng = np.random.default_rng(3)
    w, h = 48, 32                       # latent 4×6
    with tempfile.TemporaryDirectory() as td:
        saved = {}
        with tarfile.open(os.path.join(td, "shard-00000.tar"), "w") as tf:
            for s in range(3):
                key = "traj_0002_%06d" % s
                lat = rng.standard_normal((6, 4, h // 8, w // 8)).astype("<f4")
                saved[key] = lat
                meta = {"action": [0, 0, 0, 0, 0, -1, 0],
                        "latent_shape": list(lat.shape)}
                for name, data in ((key + ".latent", lat.tobytes()),
                                   (key + ".json", json.dumps(meta).encode("utf-8"))):
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
        ds = TarWindows(td, w, h, n_hist=4)
        assert len(ds) == 3
        one = ds[0]
        assert one["z_init"].shape == (4, h // 8, w // 8)
        assert one["z_hist"].shape == (4, 4, h // 8, w // 8)
        assert one["z_target"].shape == (4, h // 8, w // 8)
        assert np.allclose(one["z_target"], saved["traj_0002_000000"][5])
        assert one["action"].shape == (7,)
        assert "init" not in one, "预编码样本不应再解码 JPEG"
        batch = [ds[i] for i in range(3)]
        assert _stack(batch, "z_init").shape == (3, 4, h // 8, w // 8)
        assert _stack(batch, "z_hist").shape == (3, 4, 4, h // 8, w // 8)
        assert _stack(batch, "z_target").shape == (3, 4, h // 8, w // 8)
        ds.close()
    print("PASS  P2-5.5 预编码 latent：.latent 直读形状/数值正确，"
          "单样本 6×4×4×6 float32，训练侧可跳过 VAE encode")


def main() -> None:
    test_action_codec()
    test_rope()
    test_structure()
    test_eval_direction_sign()
    test_preencoded_latents()
    print("\nP2 回归全部通过")


if __name__ == "__main__":
    main()
