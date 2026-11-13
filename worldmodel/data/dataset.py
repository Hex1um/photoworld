"""webdataset tar 分片 → 训练窗口。

`worldmodel/train/trainer.py` 从本模块导入 `TarWindows / collate / split_dataset`，
但过去仓库里只有 `model/`、`train/`、`infer/` 三个包，没有 `data/`，
训练入口 `python -m worldmodel.train --config configs/s1_base.yaml` 直接
`ModuleNotFoundError`，rollout 也就永远拿不到 checkpoint。本模块补齐这一层。

分片格式（见 scripts/prepare_data.py）：每个样本一组同名前缀的文件
    {key}.init.jpg     初始帧
    {key}.h0.jpg ...   最近 n_hist 帧历史（旧 → 新）
    {key}.target.jpg   目标帧
    {key}.json         {"action": [7], ...}
图片解码成 (3, H, W) float32，范围 [-1, 1]（Tokenizer.encode 的约定）。

设计取舍：
  * torch 只在 `collate()` 里延迟导入 —— 数据集本身可脱离 torch 单测，
    也便于在数据生成机上先验证分片；
  * 默认不缓存解码结果。trainer 记录过 cache=True 吃 36GB 内存的事故，
    这里保持按需读取 + 最多 8 个分片句柄的 LRU。
"""
from __future__ import annotations

import io
import json
import os
import random
import tarfile
from collections import OrderedDict

import numpy as np
from PIL import Image

__all__ = ["TarWindows", "collate", "split_dataset"]

ACT_DIM = 7          # DiTConfig.act_dim
_MAX_OPEN_SHARDS = 8


def _decode_image(buf: bytes, width: int, height: int) -> np.ndarray:
    """JPEG/PNG → (3,H,W) float32 in [-1,1]。"""
    with Image.open(io.BytesIO(buf)) as im:
        im = im.convert("RGB")
        if im.size != (width, height):
            im = im.resize((width, height), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def _stack(batch: list[dict], key: str) -> np.ndarray:
    """纯 numpy 堆叠，供无 torch 环境的回归测试直接调用。"""
    return np.stack([np.asarray(b[key], dtype=np.float32) for b in batch], axis=0)


def collate(batch: list[dict]) -> dict:
    """DataLoader collate_fn：list[dict] → batched tensor dict。"""
    import torch  # 延迟导入：数据集与切分不依赖 torch
    return {
        key: torch.from_numpy(_stack(batch, key)).float()
        for key in ("init", "hist", "target", "action")
    }


class _Subset:
    """轻量视图，避免为了切分就把整个数据集复制一遍。"""

    def __init__(self, ds, ids):
        self.ds, self.ids = ds, list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ds[int(self.ids[i])]


def split_dataset(ds, val_ratio: float = 0.05, seed: int = 0):
    """按固定 seed 打乱后切 train/val；验证集至少 1 条，训练集至少 1 条。"""
    n = len(ds)
    if n == 0:
        raise RuntimeError(
            "训练分片为空：请先用 "
            "`python scripts/prepare_data.py --traj <轨迹目录> --out data/wm` 生成，"
            "或检查 configs/*.yaml 里 data.shard_dir 是否指向正确目录。"
        )
    order = list(range(n))
    random.Random(int(seed)).shuffle(order)
    n_val = min(max(1, int(round(n * float(val_ratio)))), n - 1) if n > 1 else 0
    return _Subset(ds, order[n_val:]), _Subset(ds, order[:n_val])


class TarWindows:
    """从 shard-*.tar 读取 (init, hist, target, action) 训练窗口。

    shard_dir    分片目录（含 shard-00000.tar ...）
    width/height 帧尺寸，与 prepare_data.py 一致
    n_hist       历史帧数（默认 4，与 prepare_data.N_HIST / DiTConfig.n_hist 一致）
    cache        True 时缓存解码后的样本；训练请保持 False（内存会爆）
    action_scale 可选标量或长度 7 的向量，对动作再乘一次尺度
    """

    def __init__(self, shard_dir: str, width: int, height: int, n_hist: int = 4,
                 cache: bool = False, action_scale=None):
        self.shard_dir = os.fspath(shard_dir)
        self.width, self.height = int(width), int(height)
        self.n_hist = int(n_hist)
        self.cache_enabled = bool(cache)
        self._cache = {}
        self._tars = OrderedDict()

        if action_scale is None:
            self.action_scale = None
        elif np.isscalar(action_scale):
            self.action_scale = np.full(ACT_DIM, float(action_scale), dtype=np.float32)
        else:
            vec = np.asarray(action_scale, dtype=np.float32).reshape(-1)
            if vec.size != ACT_DIM:
                raise ValueError("action_scale 必须是标量或长度 %d 的向量" % ACT_DIM)
            self.action_scale = vec

        self.shards = sorted(
            os.path.join(self.shard_dir, f) for f in os.listdir(self.shard_dir)
            if f.startswith("shard-") and f.endswith(".tar")
        ) if os.path.isdir(self.shard_dir) else []
        if not self.shards:
            raise RuntimeError(
                "找不到训练分片 %s/shard-*.tar。生成命令：\n"
                "  python scripts/prepare_data.py --traj <轨迹目录> --out %s"
                % (self.shard_dir, self.shard_dir)
            )

        wanted = {"init", "target", "meta"} | {"h%d" % k for k in range(self.n_hist)}
        self.samples = []
        for si, path in enumerate(self.shards):
            groups: dict[str, dict[str, str]] = {}
            with tarfile.open(path, "r") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    if name.endswith(".json"):
                        key, role = name[:-5], "meta"
                    elif name.lower().endswith((".jpg", ".jpeg", ".png")):
                        key, _, role = name.rsplit(".", 1)[0].rpartition(".")
                        if not key:
                            continue
                    else:
                        continue
                    groups.setdefault(key, {})[role] = name
            for key, files in sorted(groups.items()):
                if wanted.issubset(files):
                    self.samples.append((si, key, files))
        if not self.samples:
            raise RuntimeError(
                "%s 里的分片没有完整样本（需要 init/target/json + h0..h%d）。"
                "确认 prepare_data.py 的 N_HIST 与配置 n_hist 一致。"
                % (self.shard_dir, self.n_hist - 1)
            )

    def __len__(self):
        return len(self.samples)

    def _shard(self, idx: int):
        path = self.shards[idx]
        if path in self._tars:
            self._tars.move_to_end(path)
            return self._tars[path]
        tf = tarfile.open(path, "r")
        self._tars[path] = tf
        while len(self._tars) > _MAX_OPEN_SHARDS:
            _, old = self._tars.popitem(last=False)
            old.close()
        return tf

    def _read(self, shard_idx: int, name: str) -> bytes:
        tf = self._shard(shard_idx)
        f = tf.extractfile(tf.getmember(name))
        if f is None:
            raise IOError("分片成员不可读：" + name)
        return f.read()

    def __getitem__(self, idx: int) -> dict:
        if self.cache_enabled and idx in self._cache:
            return self._cache[idx]
        si, _key, files = self.samples[int(idx)]
        init = _decode_image(self._read(si, files["init"]), self.width, self.height)
        target = _decode_image(self._read(si, files["target"]), self.width, self.height)
        hist = np.stack([
            _decode_image(self._read(si, files["h%d" % k]), self.width, self.height)
            for k in range(self.n_hist)
        ], axis=0)
        meta = json.loads(self._read(si, files["meta"]).decode("utf-8"))
        action = np.asarray(meta.get("action", []), dtype=np.float32).reshape(-1)
        if action.size != ACT_DIM:
            raise ValueError(
                "样本 %s 的 action 维度是 %d，应为 %d（scripts/prepare_data.py 输出 7 维："
                "轴角3 + 平移3 + logFOV1）" % (_key, action.size, ACT_DIM)
            )
        if self.action_scale is not None:
            action = action * self.action_scale
        out = {"init": init, "hist": hist, "target": target, "action": action}
        if self.cache_enabled:
            self._cache[idx] = out
        return out

    def close(self):
        for tf in self._tars.values():
            tf.close()
        self._tars.clear()
