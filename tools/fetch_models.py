#!/usr/bin/env python
"""下载浏览器端需要的 ONNX 模型到 models/。

  python tools/fetch_models.py                # 缺什么补什么
  python tools/fetch_models.py --semantics    # 只补语义模型

模型：
  segformer_b0_ade20k.onnx  SegFormer-B0 / ADE20K 150 类（约 15MB，量化）
      Xenova/segformer-b0-finetuned-ade-512-512 · onnx/model_quantized.onnx
  model_quantized.onnx      Depth-Anything-V2-Small（约 27MB，仓库已自带）
  yolov8n-seg.onnx          YOLOv8n-seg COCO（约 14MB，仓库已自带）
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")

TARGETS = {
    "semantics": {
        "file": "segformer_b0_ade20k.onnx",
        "mirrors": [
            "https://hf-mirror.com/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx",
            "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx",
        ],
        "min_bytes": 3 * 1024 * 1024,   # int8 量化后约 4.4MB
    },
    "depth": {
        "file": "model_quantized.onnx",
        "mirrors": [],
        "min_bytes": 10 * 1024 * 1024,
    },
    "yolo": {
        "file": "yolov8n-seg.onnx",
        "mirrors": [],
        "min_bytes": 5 * 1024 * 1024,
    },
}


def download(url: str, dest: str) -> None:
    tmp = dest + ".part"
    print("  GET " + url)
    req = urllib.request.Request(url, headers={"User-Agent": "photoworld/1.16"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = got * 100 / total
                sys.stdout.write("\r    %5.1f%%  %.1f/%.1f MB" % (pct, got / 1e6, total / 1e6))
                sys.stdout.flush()
    sys.stdout.write("\n")
    os.replace(tmp, dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantics", action="store_true", help="只下语义模型")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    a = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    keys = ["semantics"] if a.semantics else list(TARGETS)
    rc = 0
    for k in keys:
        t = TARGETS[k]
        dest = os.path.join(MODELS, t["file"])
        if os.path.exists(dest) and os.path.getsize(dest) >= t["min_bytes"] and not a.force:
            print("[skip] %s 已存在（%.1f MB）" % (t["file"], os.path.getsize(dest) / 1e6))
            continue
        if not t["mirrors"]:
            print("[warn] %s 未提供下载源，请确认仓库已自带" % t["file"])
            continue
        ok = False
        for url in t["mirrors"]:
            try:
                download(url, dest)
                if os.path.getsize(dest) >= t["min_bytes"]:
                    print("[ok]   %s  %.1f MB" % (t["file"], os.path.getsize(dest) / 1e6))
                    ok = True
                    break
            except Exception as e:
                print("  failed: %s" % e)
        if not ok:
            print("[fail] %s 下载失败，请检查网络或镜像" % t["file"])
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
