#!/usr/bin/env python3
"""rollout：初始图 + 动作 DSL → 视频（GIF + 帧目录）"""
import argparse, json, os, sys
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from worldmodel.infer.rollout import rollout, parse_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--actions", default="forward 20,turn_left 20,forward 20")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="out/rollout")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--size", default="384x256")
    a = ap.parse_args()

    w, h = [int(v) for v in a.size.lower().split("x")]
    im = Image.open(a.image).convert("RGB").resize((w, h), Image.LANCZOS)
    x = torch.from_numpy(__import__("numpy").asarray(im, dtype="float32") / 127.5 - 1.0).permute(2, 0, 1)[None]

    acts = parse_actions([s.strip() for s in a.actions.split(",") if s.strip()])
    print("动作 %d 帧" % len(acts))
    frames = rollout(x, acts, a.ckpt, steps=a.steps, cfg_scale=a.cfg)

    os.makedirs(a.out, exist_ok=True)
    imgs = []
    for i, f in enumerate(frames):
        arr = ((f.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
        img = Image.fromarray(arr)
        img.save(os.path.join(a.out, "%06d.jpg" % i), quality=92)
        imgs.append(img)
    if imgs:
        imgs[0].save(os.path.join(a.out, "rollout.gif"), save_all=True, append_images=imgs[1:],
                     duration=120, loop=0)
    json.dump({"frames": len(frames), "actions": a.actions, "steps": a.steps, "cfg": a.cfg},
              open(os.path.join(a.out, "rollout.json"), "w"), indent=1)
    print("完成 %d 帧 → %s" % (len(frames), a.out))


if __name__ == "__main__":
    main()
