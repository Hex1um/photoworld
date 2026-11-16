#!/usr/bin/env python
"""把本仓库的 SkillOpt 集成同步进 SkillOpt 克隆。

为什么要这个脚本：
  SkillOpt 是**独立的 clone**（photoworld 的同级目录 `tools_skillopt/`），
  我们的 env / config 改动如果只写在 clone 里，重克隆就丢了，也不进本仓库版本管理。
  所以权威副本放在本仓库的 `skillopt/`，用这个脚本推过去。

用法：
    python tools/sync_skillopt.py                # 推到 ../tools_skillopt
    python tools/sync_skillopt.py --dir D:\\path  # 指定 clone 位置
    python tools/sync_skillopt.py --check        # 只比对，不写
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ENV = os.path.join(ROOT, "skillopt", "envs", "photoworld")
SRC_CFG = os.path.join(ROOT, "skillopt", "configs", "photoworld.yaml")

REGISTER = '''    try:
        # PHOTOWORLD：把烘焙参数当技能训练（target 是确定性流水线，不是 LLM）
        from skillopt.envs.photoworld.adapter import PhotoWorldAdapter
        _ENV_REGISTRY["photoworld"] = PhotoWorldAdapter
    except ImportError:
        pass
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="SkillOpt clone 目录（默认 ../tools_skillopt）")
    ap.add_argument("--check", action="store_true", help="只比对不写")
    args = ap.parse_args()

    dst = args.dir or os.path.join(os.path.dirname(ROOT), "tools_skillopt")
    if not os.path.isdir(os.path.join(dst, "skillopt", "envs")):
        print(f"[x] 这不是 SkillOpt 仓库：{dst}")
        print("    先克隆： git clone https://github.com/microsoft/SkillOpt.git tools_skillopt")
        return 1

    changed = []
    # 1) env 包
    d_env = os.path.join(dst, "skillopt", "envs", "photoworld")
    os.makedirs(d_env, exist_ok=True)
    for name in sorted(os.listdir(SRC_ENV)):
        s, t = os.path.join(SRC_ENV, name), os.path.join(d_env, name)
        same = os.path.exists(t) and open(s, "rb").read() == open(t, "rb").read()
        if not same:
            changed.append(f"env/{name}")
            if not args.check:
                shutil.copyfile(s, t)
    # 2) 配置
    d_cfg = os.path.join(dst, "configs", "photoworld")
    os.makedirs(d_cfg, exist_ok=True)
    s, t = SRC_CFG, os.path.join(d_cfg, "default.yaml")
    same = os.path.exists(t) and open(s, "rb").read() == open(t, "rb").read()
    if not same:
        changed.append("configs/photoworld/default.yaml")
        if not args.check:
            shutil.copyfile(s, t)

    # 3) 注册（train.py / eval_only.py 的 _ENV_REGISTRY）
    for script in ("train.py", "eval_only.py"):
        p = os.path.join(dst, "scripts", script)
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        if "photoworld" in txt:
            continue
        anchor = '        _ENV_REGISTRY["officeqa"] = OfficeQAAdapter\n    except ImportError:\n        pass\n'
        if anchor in txt:
            txt = txt.replace(anchor, anchor + REGISTER, 1)
            changed.append(f"scripts/{script}（注册）")
            if not args.check:
                open(p, "w", encoding="utf-8").write(txt)
        else:
            print(f"[!] {script} 里没找到 officeqa 注册锚点，需手工加 photoworld")

    if not changed:
        print("[ok] 已经一致，无需同步。")
        return 0
    print(("[check] 需要同步：" if args.check else "[ok] 已同步：") + "、".join(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
