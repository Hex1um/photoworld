#!/usr/bin/env python3
"""轨迹目录 → 训练分片（webdataset tar）

输入：
    <traj_dir>/traj_XXXX/frames/000000.jpg ...
    <traj_dir>/traj_XXXX/poses.json      # {fovY, fovX, width, height, frames:[{file, camToWorld[16], action}]}

输出：
    <out>/shard-00000.tar
      每个样本包含：
        {key}.init.jpg        初始帧
        {key}.h0..h3.jpg      最近 N=4 帧历史（旧→新）
        {key}.target.jpg      目标帧
        {key}.json            {action:[7], actions_dsl:[...], meta:{source,scene,seed,t}, ...}
    <out>/index.json        统计信息（样本数、分片数、动作分布）

动作 a_t = [axis-angle(3), Δt(3) 相机系且按中位速度归一化, log(FOV) 1]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tarfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldmodel.action_codec import (  # noqa: E402
    mat4_from_rowmajor, rel_action, rot_to_axis_angle,
)

N_HIST = 4

# 三个函数改由 worldmodel/action_codec.py 提供：推理侧的 DSL 也调用同一个
# rel_action，避免训练标签与 rollout 动作再次漂移（见 action_codec 模块注释）。


def load_traj(tdir: str) -> dict | None:
    pj = os.path.join(tdir, "poses.json")
    if not os.path.exists(pj):
        return None
    meta = json.load(open(pj, encoding="utf-8"))
    frames = meta.get("frames", [])
    if len(frames) < N_HIST + 2:
        return None
    Ts = [mat4_from_rowmajor(f["camToWorld"]) for f in frames]
    fov = float(meta.get("fovX") or meta.get("fovY") or 1.2)
    speeds = []
    for i in range(len(Ts) - 1):
        speeds.append(float(np.linalg.norm(Ts[i + 1][:3, 3] - Ts[i][:3, 3])))
    moving = [s for s in speeds if s > 1e-6]          # 只用真正在动的帧做尺度
    med = float(np.median(moving)) if moving else 0.08
    return {"meta": meta, "Ts": Ts, "fov": fov, "speed": max(med, 1e-3),
            "dir": tdir, "frames": frames}


def jpeg_bytes(path: str, size: tuple[int, int], quality: int) -> bytes:
    # 已是目标尺寸就直接复用原字节（省掉 3 万次重编码）
    if quality >= 92:
        with Image.open(path) as probe:
            if probe.size == size and probe.format == "JPEG":
                return open(path, "rb").read()
    im = Image.open(path).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="轨迹根目录（内含 traj_XXXX）")
    ap.add_argument("--out", default="data/wm", help="输出目录")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--shard", type=int, default=200, help="每个 tar 的样本数")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--append", action="store_true", help="追加到已有分片之后")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.json")
    index = {"samples": 0, "shards": 0, "trajs": 0, "actions": {}}
    start_shard = 0
    if args.append and os.path.exists(index_path):
        index = json.load(open(index_path, encoding="utf-8"))
        start_shard = index["shards"]

    traj_dirs = sorted(
        os.path.join(args.traj, d) for d in os.listdir(args.traj)
        if d.startswith("traj_") and os.path.isdir(os.path.join(args.traj, d))
    )
    if not traj_dirs:
        print("没有找到 traj_XXXX 目录：" + args.traj)
        return 2

    size = (args.width, args.height)
    shard_idx = start_shard
    sample_in_shard = 0
    total = 0
    tf = None

    def open_shard():
        nonlocal tf
        path = os.path.join(args.out, "shard-%05d.tar" % shard_idx)
        tf = tarfile.open(path, "w")
        return tf

    def add(name: str, data: bytes):
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        ti.mtime = int(time.time())
        tf.addfile(ti, io.BytesIO(data))

    open_shard()
    t0 = time.time()
    _ACT_SAMPLES = []          # 4.7：动作标准化的统计样本

    for tdir in traj_dirs:
        tr = load_traj(tdir)
        if not tr:
            continue
        index["trajs"] += 1
        scene = os.path.basename(tdir)
        frames = tr["frames"]
        n = len(frames)
        # P1（v2.15.8 修）：窗口从 i=0 开始，历史不足的部分用 frames[0] 补齐。
        # 原因：rollout 的冷启动历史是 `[z_init] * hist_frames`（同一帧复制 N 份），
        # 而旧数据只产 i ≥ N_HIST 的窗口 —— 训练集里**从来没有**这种"历史全是首帧"
        # 的样本，模型第一帧就处于分布外，动作响应最先失真（rollout 塌黑的第一帧来源）。
        for i in range(0, n - 1):
            a = rel_action(tr["Ts"][i], tr["Ts"][i + 1], tr["fov"], tr["fov"], tr["speed"])
            key = "%s_%06d" % (scene, i)
            try:
                add(key + ".init.jpg", jpeg_bytes(os.path.join(tdir, "frames", frames[0]["file"]), size, args.quality))
                for k in range(N_HIST):
                    # 早于 0 的历史帧用首帧补齐（与 rollout 的 init×N 同构）
                    src = frames[max(0, i - N_HIST + 1 + k)]["file"]
                    add("%s.h%d.jpg" % (key, k), jpeg_bytes(os.path.join(tdir, "frames", src), size, args.quality))
                add(key + ".target.jpg", jpeg_bytes(os.path.join(tdir, "frames", frames[i + 1]["file"]), size, args.quality))
            except Exception as exc:  # noqa: BLE001
                print("跳过 %s：%s" % (key, exc))
                continue

            # P2（v2.15.8 修）：历史动作只收到**目标动作之前**那一帧（j ≤ i-1），
            # 否则最后一项 rel(Ts[i], Ts[i+1]) 就是 action 本身 —— 元数据泄漏，
            # 任何"用历史动作做条件"的后续开发都会把答案读进输入。
            hist_actions = []
            for k in range(N_HIST):
                j = i - N_HIST + k
                j = max(0, j)
                hist_actions.append(rel_action(tr["Ts"][j], tr["Ts"][j + 1], tr["fov"], tr["fov"], tr["speed"]).tolist())
            meta = {
                "action": [round(float(v), 5) for v in a],
                "hist_actions": [[round(float(v), 5) for v in h] for h in hist_actions],
                "action_dsl": frames[i + 1].get("action", "idle"),
                "meta": {"source": "synthetic", "scene": scene, "seed": 0, "t": i,
                         "fov": round(tr["fov"], 5), "speed": round(tr["speed"], 5)},
            }
            add(key + ".json", json.dumps(meta).encode("utf-8"))

            _ACT_SAMPLES.append(a)
            total += 1
            sample_in_shard += 1
            index["actions"][meta["action_dsl"]] = index["actions"].get(meta["action_dsl"], 0) + 1
            if sample_in_shard >= args.shard:
                tf.close()
                shard_idx += 1
                sample_in_shard = 0
                open_shard()

    tf.close()
    # P2（v2.15.8 修）：动作 7 维量纲差 ~20 倍（平移按速度归一、旋转是 raw 弧度）。
    # 不标准化的话 MSE 里旋转的梯度天然被淹没 → "turn 无效、forward 还行"
    # （report.md 症状①）。这里一次性统计每维标准差，存成 1/std 供两端使用：
    #   · dataset.py 会对 rel_action 再乘一次 action_scale
    #   · rollout.py 的 parse_actions 也乘同一个值（T6，训推一致）
    #   配置里显式给 `data.action_scale` 时以配置为准。
    if _ACT_SAMPLES:
        _a = np.asarray(_ACT_SAMPLES, dtype=np.float64)
        _std = _a.std(axis=0) + 1e-6
        index["action_scale"] = [round(float(v), 4) for v in (1.0 / _std)]
        print("动作标准化系数（1/std）：" + str(index["action_scale"]))
    index["samples"] += total
    index["shards"] = shard_idx + (1 if sample_in_shard > 0 else 0)
    index["size"] = [args.width, args.height]
    index["n_hist"] = N_HIST
    json.dump(index, open(index_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("完成：%d 条样本 / %d 个分片 / %.1fs → %s" % (total, index["shards"], time.time() - t0, args.out))
    print("动作分布：" + json.dumps(index["actions"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
