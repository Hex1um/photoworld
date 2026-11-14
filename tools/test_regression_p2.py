#!/usr/bin/env python3
"""P2 回归：世界模型动作语义 / RoPE 等变性 / 初始化与 CFG / 预编码 latent。

全部纯 numpy + 源码结构断言，不需要 torch 或 GPU：

  python tools/test_regression_p2.py

覆盖 5.1~5.5 里可以脱离训练机验证的部分。训练全链路仍需在有 CUDA torch 的
机器上实跑；本脚本保证"语义定义"与"代码结构"不会再悄悄漂移回去。
"""
from __future__ import annotations

import ast
import glob
import importlib.util
import io
import json
import math
import os
import re
import sys
import tarfile
import tempfile

import numpy as np
import yaml

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


# ---------------- v1.17 训练侧省时开关：无 torch 静态回归 ----------------
# 与 tools/test_trainer_opts.py 互补：后者 import torch，只能在训练机跑。
# 本组断言纯 ast + numpy + yaml，在无 torch / 无 GPU 的开发机也能锁住
# 「默认值、配置语义、权重键、alpha 分布、MACs 解析估算」不随重构漂移。


def _trainer_fn(src: str, name: str, ns: dict):
    """把 trainer.py 的顶层纯函数单独编译执行（trainer 顶层 import torch，本机没有）。"""
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, "trainer.py:%s" % name, "exec"), ns)
    return ns[name]


def _get_defaults(src: str, key: str) -> list:
    """收集 cfg.*.get("key", <default>) 里的字面默认值。"""
    out = []
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and len(n.args) >= 2
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == key):
            try:
                out.append(ast.literal_eval(n.args[1]))
            except ValueError:
                out.append("<expr>")
    return out


def _module_attrs(src: str, cls_name: str) -> set:
    """__init__ 里 self.X = nn.*(...) / nn.Parameter(...) 的属性名 = state_dict 顶层键。"""
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    out = set()
    for n in ast.walk(init):
        if not isinstance(n, ast.Assign):
            continue
        seg = ast.get_source_segment(src, n.value) or ""
        custom = (isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                  and n.value.func.id in ("Attention", "Block"))
        if "nn." not in seg and "Parameter" not in seg and not custom:
            continue
        for t in n.targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                out.add(t.attr)
    return out


def test_train_opt_defaults() -> None:
    """① 四个 opt-in 开关的默认值走旧路径；配置档位语义与上游 v1.17 一致。"""
    tr = _src("worldmodel/train/trainer.py")
    assert _get_defaults(tr, "ss_mode") == ["forward"], "ss_mode 默认值不是旧 forward 路径"
    assert _get_defaults(tr, "val_max_samples") == [0], "val_max_samples 默认值不是全量"
    assert _get_defaults(tr, "ema_every") == [1], "ema_every 默认值不是逐步更新"
    assert 'if not cfg["data"].get("prefer_latent")' in tr, "prefer_latent 默认分支丢失"
    for frag in ("resolve_shard_dir", "coerce_train_cfg", "get_tok().encode(init)",
                 "v_self = model(", "for vb in dl_va:", "ema.copy_to(model)",
                 'ck = {"model"', "uncond=uncond"):
        assert frag in tr, "默认执行路径被破坏：%s" % frag

    paths = sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml")))
    assert len(paths) == 10, "配置档位数量变了：%d" % len(paths)
    for p in paths:
        name = os.path.basename(p)
        cfg = yaml.safe_load(open(p, encoding="utf-8"))
        t, d = cfg["train"], cfg["data"]
        assert bool(d.get("prefer_latent", False)) == (name == "s1_fast.yaml"), \
            "%s 的 prefer_latent 改变了基线语义" % name
        assert t.get("ss_mode", "forward") == \
            ("noise_proxy" if name == "s3.yaml" else "forward"), \
            "%s 的 ss_mode 改变了默认执行路径" % name
        assert (t.get("val_max_samples") or 0) == \
            (64 if name in ("s1_fast.yaml", "s3.yaml") else 0), \
            "%s 的 val_max_samples 不合约定" % name
        assert (t.get("ema_every") or 1) == (4 if name == "s1_fast.yaml" else 1), \
            "%s 的 ema_every 不合约定" % name

    # 纯函数：prefer_latent 自动切换 / 缺失回退
    resolve = _trainer_fn(tr, "resolve_shard_dir", {"os": os})
    with tempfile.TemporaryDirectory() as td:
        wm, lat = os.path.join(td, "wm"), os.path.join(td, "wm_latent")
        os.makedirs(wm)
        os.makedirs(lat)
        shard = os.path.join(lat, "shard-00000.tar")
        open(shard, "wb").write(b"")
        assert os.path.normpath(
            resolve({"data": {"shard_dir": wm, "prefer_latent": False}})) == os.path.normpath(wm), \
            "prefer_latent=false 时不得切换目录"
        assert os.path.normpath(
            resolve({"data": {"shard_dir": wm, "prefer_latent": True}})) == os.path.normpath(lat), \
            "prefer_latent=true 时未切到同名 _latent 目录"
        os.remove(shard)
        assert os.path.normpath(
            resolve({"data": {"shard_dir": wm, "prefer_latent": True}})) == os.path.normpath(wm), \
            "找不到 latent 分片时应回退而非报错"

    # 纯函数：PyYAML 把 lr: 1e-3 读成字符串，coerce_train_cfg 必须统一转数值
    coerce = _trainer_fn(tr, "coerce_train_cfg", {})
    out = coerce({"train": {"lr": "1.5e-4", "steps": "2000",
                            "val_max_samples": "64", "ema_every": "4", "clip": 1.0}})
    assert isinstance(out["train"]["lr"], float) and abs(out["train"]["lr"] - 1.5e-4) < 1e-12
    assert out["train"]["steps"] == 2000 and isinstance(out["train"]["steps"], int)
    assert out["train"]["val_max_samples"] == 64 and isinstance(out["train"]["val_max_samples"], int)
    assert out["train"]["ema_every"] == 4 and isinstance(out["train"]["ema_every"], int)
    print("PASS  v1.17-① 默认路径：ss_mode/val_max_samples/ema_every = forward/0/1，"
          "10 个配置语义正确；latent 自动切换与超参数值化可复现")


def test_dit_structure_and_state_dict_keys() -> None:
    """② CausalWorldDiT 结构、权重键名、损失 / EMA / CFG 路径逐项锁定。"""
    dit = _src("worldmodel/model/dit.py")
    expected = {
        "CausalWorldDiT": {"patch", "t_embed", "act_embed", "null_act", "null_dom",
                           "dom_embed", "blocks", "norm", "head"},
        "Block": {"n1", "attn", "n2", "mlp", "mod"},
        "Attention": {"qkv", "out"},
    }
    for cls_name, keys in expected.items():
        got = _module_attrs(dit, cls_name)
        assert got == keys, "%s 的权重键变了：新增 %s / 丢失 %s" % (
            cls_name, sorted(got - keys), sorted(keys - got))

    tree = ast.parse(dit)
    cfg_cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "DiTConfig")
    fields = {n.target.id: ast.literal_eval(n.value)
              for n in cfg_cls.body if isinstance(n, ast.AnnAssign)}
    assert fields == {"in_ch": 4, "latent_h": 32, "latent_w": 48, "patch": 2,
                      "dim": 768, "depth": 12, "heads": 12, "n_hist": 4,
                      "act_dim": 7, "n_domains": 4, "mlp_ratio": 4.0}, \
        "DiTConfig 字段/默认值变了，旧 last.pt 会加载失败：%s" % fields
    cw = next(n for n in tree.body
              if isinstance(n, ast.ClassDef) and n.name == "CausalWorldDiT")
    fwd = next(n for n in cw.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assert [a.arg for a in fwd.args.args] == \
        ["self", "init_lat", "hist_lat", "x_t", "t", "action", "domain", "uncond"], \
        "forward 签名变了，CFG 等价性无法保证"
    for frag in ("torch.clamp(idx // per_frame, max=n_frames - 1)",
                 "fid.view(-1, 1) >= fid.view(1, -1)",
                 "self.head(x[:, -cfg.per_frame:])",
                 "rope_cache(", "nn.init.normal_(self.head.weight"):
        assert frag in dit, "DiT 关键结构被改动：%s" % frag

    tr = _src("worldmodel/train/trainer.py")
    for frag in ("F.mse_loss(v.float(), v_target.float()) / accum",
                 "ema.update(model)", "ema.copy_to(model)", "uncond=uncond",
                 "v_target = z1 - z0", 'model.load_state_dict(ck["model"])'):
        assert frag in tr, "训练语义被改动：%s" % frag
    assert 'if "z_init" in batch' in tr, "预编码 latent 分支丢失"
    print("PASS  v1.17-② 结构：DiT 三个类的 state_dict 顶层键、DiTConfig、"
          "forward 签名、损失/EMA/CFG 全部未变（可加载既有 last.pt）")


def test_no_deprecated_drop_keys() -> None:
    """③ configs/*.yaml 清掉 drop_action / drop_init，统一 drop_cond。"""
    for p in sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml"))):
        s = open(p, encoding="utf-8").read()
        assert "drop_action" not in s and "drop_init" not in s, \
            "%s 仍有废弃的条件 dropout 键" % os.path.basename(p)
        assert yaml.safe_load(s)["train"]["drop_cond"] == 0.1, \
            "%s 的 drop_cond 不是 0.1" % os.path.basename(p)
    print("PASS  v1.17-③ 配置键名：10 个 yaml 无 drop_action/drop_init，统一 drop_cond: 0.1")


def test_noise_proxy_distribution() -> None:
    """④ noise_proxy 的 alpha 分布 / 混合分布，以及因果掩码有效对 3099265/5313025。"""
    tr = _src("worldmodel/train/trainer.py")
    m = re.search(r"alpha\s*=\s*([0-9.]+)\s*\+\s*torch\.rand\([^)]*\)\s*\*\s*([0-9.]+)", tr)
    assert m, "找不到 noise_proxy 的 alpha 采样式"
    amin, span = float(m.group(1)), float(m.group(2))
    assert amin == 0.05 and span == 0.20, "alpha 采样区间不是 U(0.05, 0.25)"
    for frag in ('if ss_mode == "noise_proxy":',
                 'z_self = (1 - alpha) * z1 + alpha * torch.randn_like(z1)',
                 'v_self = model('):
        assert frag in tr, "scheduled sampling 双路径被改动：%s" % frag

    rng = np.random.default_rng(20261114)
    n = 400000
    a = rng.random(n) * span + amin
    assert a.min() >= amin and a.max() < amin + span, "alpha 越出 [0.05, 0.25)"
    assert abs(a.mean() - 0.15) < 5e-4, "alpha 均值不是 0.15"
    assert abs(a.var() - span * span / 12) < 5e-5, "alpha 方差不是 0.2^2/12"
    z1 = rng.standard_normal((3, 4, 32, 48))
    eps = rng.standard_normal(z1.shape)
    alpha = rng.random((3, 1, 1, 1)) * span + amin
    out = (1 - alpha) * z1 + alpha * eps
    assert out.shape == z1.shape, "alpha 未按 (B,1,1,1) 广播"
    assert np.allclose(out, (1 - alpha) * z1 + alpha * eps, rtol=0, atol=1e-12)
    e = rng.standard_normal(n)
    aa = rng.random(n) * span + amin
    y = (1 - aa) + aa * e               # z1 恒为 1 时的解析分布
    assert abs(y.mean() - 0.85) < 0.004, "输出均值应趋于 1-E[alpha]=0.85"
    assert abs(y.var() - 7 / 240) < 0.001, \
        "输出方差应为 Var(c)+E[(1-c)^2] = 7/240 ≈ 0.029167"

    L, per, nf = 2305, 384, 6
    fid = np.minimum(np.arange(L) // per, nf - 1)   # 动作 token 归入最后一帧
    allow = fid[:, None] >= fid[None, :]
    assert L * L == 5313025, "序列长度不是 2305"
    assert int(allow.sum()) == 3099265, "因果掩码有效对数与基线不一致：%d" % allow.sum()
    assert allow.sum() / (L * L) == 3099265 / 5313025, "有效占比不再精确一致"
    for frag in ("n_frames = 2 + cfg.n_hist",
                 "torch.clamp(idx // per_frame, max=n_frames - 1)",
                 "fid.view(-1, 1) >= fid.view(1, -1)"):
        assert frag in _src("worldmodel/model/dit.py"), "掩码构造漂移：%s" % frag
    print("PASS  v1.17-④ 数值：alpha~U(0.05,0.25) 均值 0.15 / 方差 0.003333；"
          "noise_proxy 均值 0.85 / 方差 7/240；因果掩码 3099265/5313025 = 58.33%")


def test_macs_estimate() -> None:
    """⑤ 解析估算（非实测）：把 CHANGELOG/README 里的收益数字用公式复算一遍。"""
    d, L, depth, bs, accum = 768, 2305, 12, 2, 16
    per_block = (3 * L * d * d          # QKV
                 + 2 * L * L * d        # QK^T + AV
                 + L * d * d           # out proj
                 + 8 * L * d * d       # MLP
                 + 6 * d * d)          # adaLN mod
    fwd = depth * per_block
    step_base = bs * accum * 3 * fwd   # 反向按 2x 前向估算 -> fwd+bwd = 3x fwd
    extra_ss = 0.5 * bs * accum * fwd  # p_ss=0.5 时每 micro-batch 期望 0.5 次额外前向
    assert abs(fwd / 1e9 - 293.75) < 0.1, "单样本前向基线漂移"
    assert abs(step_base / 1e12 - 28.20) < 0.02, "每 step 基线漂移"
    assert abs(extra_ss / step_base - 1 / 6) < 1e-12, "相对基线不是 -16.7%"
    assert abs(extra_ss / (step_base + extra_ss) - 1 / 7) < 1e-12, "相对含 SS 总量不是 -14.3%"
    print("PASS  v1.17-⑤ 解析估算（非实测）：前向 %.2f G / 步基线 %.2f T MACs；"
          "取消 SS 额外前向 -%.2f T/step（相对基线 -16.7%%，相对含 SS 总量 -14.3%%）"
          % (fwd / 1e9, step_base / 1e12, extra_ss / 1e12))


def main() -> None:
    test_action_codec()
    test_rope()
    test_structure()
    test_eval_direction_sign()
    test_preencoded_latents()
    test_train_opt_defaults()
    test_dit_structure_and_state_dict_keys()
    test_no_deprecated_drop_keys()
    test_noise_proxy_distribution()
    test_macs_estimate()
    print("\nP2 回归全部通过")


if __name__ == "__main__":
    main()
