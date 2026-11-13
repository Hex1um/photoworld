#!/usr/bin/env python3
"""把图片训练分片**离线**预编码成 latent 分片（需要 torch + diffusers + CUDA）。

  python scripts/preencode_latents.py --shards data/wm --out data/wm_latent
  # 然后把配置里的 data.shard_dir 指向 data/wm_latent 即可

动机
----
`trainer.py` 原来每个 micro-batch 在线编码 init + n_hist 帧 + target 共 6 张图；
按 `configs/s1_base.yaml`（batch 2、grad_accum 16）每个 optimizer step 要 192 次
VAE encode，2000 步约 38.4 万次。latent 单样本只有 6×4×32×48×4B ≈ 147KB，
比 6 张 JPEG 还小，且训练时只需一次 numpy.frombuffer。

产物（不改动原有分片，另写一份）
--------------------------------
  shard-*.tar
    {key}.latent   float32，(2+n_hist, 4, H/8, W/8)，顺序 init / h0..hN-1 / target
    {key}.json     原 meta 加 `latent_shape`
  index.json       原内容加 `latent: true`

`worldmodel/data/dataset.py::TarWindows` 检测到 `.latent` 会自动走直读路径，
`trainer.py` 检测到 batch 里的 `z_init` 会跳过 VAE；没有 latent 的分片仍按原方式工作。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldmodel.data.dataset import _decode_image  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png")


def _groups(tf: tarfile.TarFile) -> dict:
    out = {}
    for m in tf.getmembers():
        if not m.isfile():
            continue
        name = m.name
        if name.endswith(".json"):
            key, role = name[:-5], "meta"
        elif name.lower().endswith(IMG_EXT):
            key, _, role = name.rsplit(".", 1)[0].rpartition(".")
            if not key:
                continue
        else:
            continue
        out.setdefault(key, {})[role] = name
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="原图片分片目录（含 shard-*.tar）")
    ap.add_argument("--out", required=True, help="输出 latent 分片目录")
    ap.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=8, help="每次 VAE encode 的帧数")
    args = ap.parse_args()

    import torch  # 延迟导入：无 torch 的机器仍可 --help / 做语法检查
    from worldmodel.model.tokenizer import Tokenizer

    srcs = sorted(f for f in os.listdir(args.shards)
                  if f.startswith("shard-") and f.endswith(".tar"))
    if not srcs:
        print("找不到分片：" + args.shards)
        return 2
    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer(args.vae, device=args.device)
    t0 = time.time()
    total = 0

    for fname in srcs:
        with tarfile.open(os.path.join(args.shards, fname), "r") as tf:
            members = {m.name: m for m in tf.getmembers() if m.isfile()}
            groups = _groups(tf)
            dst_path = os.path.join(args.out, fname)
            with tarfile.open(dst_path, "w") as out:

                def add(name: str, data: bytes):
                    ti = tarfile.TarInfo(name)
                    ti.size = len(data)
                    ti.mtime = int(time.time())
                    out.addfile(ti, io.BytesIO(data))

                for key, files in sorted(groups.items()):
                    if "meta" not in files:
                        continue
                    meta = json.loads(tf.extractfile(members[files["meta"]]).read().decode("utf-8"))
                    n_hist = len(meta.get("hist_actions", [])) or 4
                    roles = ["init"] + ["h%d" % k for k in range(n_hist)] + ["target"]
                    pics = []
                    for role in roles:
                        if role not in files:
                            break
                        buf = tf.extractfile(members[files[role]]).read()
                        with Image.open(io.BytesIO(buf)) as probe:
                            w, h = probe.size
                        pics.append(_decode_image(buf, w, h))
                    if len(pics) != len(roles):
                        print("跳过 %s：图片不全" % key)
                        continue

                    arr = np.stack(pics, axis=0).astype(np.float32)
                    x = torch.from_numpy(arr).to(args.device)
                    lats = []
                    with torch.no_grad():
                        for s in range(0, len(x), args.chunk):
                            lats.append(tok.encode(x[s:s + args.chunk]).cpu().numpy())
                    lat = np.concatenate(lats, axis=0).astype("<f4")
                    add(key + ".latent", lat.tobytes())
                    meta["latent_shape"] = list(lat.shape)
                    add(key + ".json", json.dumps(meta).encode("utf-8"))
                    total += 1
        print("%s → %d 样本" % (fname, total), flush=True)

    idx_path = os.path.join(args.shards, "index.json")
    if os.path.exists(idx_path):
        idx = json.load(open(idx_path, encoding="utf-8"))
        idx["latent"] = True
        json.dump(idx, open(os.path.join(args.out, "index.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
    print("完成：%d 样本 / %.1fs → %s" % (total, time.time() - t0, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
