#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 glb 世界渲染带真值位姿的轨迹（合成数据源，Python 版）。

    python scripts/render_traj.py --glb assets/world-alpine.glb --out data/syn \\
        --trajs 6 --frames 48

产出 data/syn/traj_XXXX/{frames/000000.jpg, poses.json}。
位姿生成与 trsArray 矩阵与 JS 版逐位一致（同一个 LCG 种子公式）。
"""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import time

import cdp as C

W, H = 384, 256


def make_pos(seed: int, n: int):
    """与 JS 版同款的确定性伪随机位姿，返回 (poses, acts)。"""
    s = seed * 9301 + 49297

    def rnd():
        nonlocal s
        s = (s * 9301 + 49297) % 233280
        return s / 233280.0

    poses, acts = [], []
    x = z = 0.0
    yaw = (rnd() - 0.5) * 0.5
    pitch = (rnd() - 0.5) * 0.12
    for _ in range(n):
        poses.append([x, 1.7, z, yaw, pitch])
        r = rnd()
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
        else:
            act = "idle"
        acts.append(act)
    return poses, acts


def trs_array(x, y, z, yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    vals = [
        cy, 0, sy, 0,
        sy * sp, cp, -cy * sp, 0,
        -sy * cp, sp, cy * cp, 0,
        x, y, z, 1,
    ]
    return [math.floor(v * 1e5 + 0.5) / 1e5 if v >= 0 else math.ceil(v * 1e5 - 0.5) / 1e5
            for v in vals]


def main() -> int:
    a = C.args_dict()
    glb = a.get("glb", "assets/world-alpine.glb")
    if not glb.startswith("/"):
        glb = "/" + glb
    out = a.get("out", "data/syn")
    n_traj = int(a.get("trajs", "4"))
    n_frames = int(a.get("frames", "48"))

    b = None
    try:
        b = C.launch(prefix="pwtraj-", w=W, h=H, edge=a.get("edge"),
                     gpu="swiftshader", url="about:blank")
        c = b.cdp
        C.enable_common(c, W, H)
        c.send("Page.navigate", {"url": "http://127.0.0.1:8099/tools/traj.html"})
        time.sleep(1.6)

        init = c.eval_raw("__traj.init(%s)" % json.dumps(glb, ensure_ascii=False))
        if init.get("exceptionDetails"):
            raise C.CDPError("init 失败：" + str(init["exceptionDetails"].get("text")))
        print("世界已加载", json.dumps((init.get("result") or {}).get("value"), ensure_ascii=False))

        fov_y = 2 * math.atan(math.tan(69.4 * math.pi / 360) * (H / W))
        os.makedirs(out, exist_ok=True)

        for t in range(n_traj):
            poses, acts = make_pos(t + 1, n_frames)
            r = c.eval_raw("__traj.runPath(%s, %s)"
                           % (json.dumps(poses, ensure_ascii=False), fov_y))
            if r.get("exceptionDetails"):
                raise C.CDPError("渲染失败：" + str(r["exceptionDetails"].get("text")))
            frames = (r.get("result") or {}).get("value") or []
            tag = "traj_" + str(t).zfill(4)
            frames_dir = os.path.join(out, tag, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            meta = {"fovY": fov_y, "fovX": 69.4 * math.pi / 180,
                    "width": W, "height": H, "frames": []}
            for i, b64 in enumerate(frames):
                frame_name = str(i).zfill(6) + ".jpg"
                with open(os.path.join(frames_dir, frame_name), "wb") as f:
                    f.write(base64.b64decode(b64))
                p = poses[i]
                meta["frames"].append({
                    "file": frame_name,
                    "camToWorld": trs_array(p[0], p[1], p[2], p[3], p[4]),
                    "action": acts[i],
                })
            with open(os.path.join(out, tag, "poses.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            print("%s  %d 帧" % (tag, len(frames)))
    except Exception as e:
        print("失败：" + str(e), file=sys.stderr)
        return 1
    finally:
        if b:
            b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
