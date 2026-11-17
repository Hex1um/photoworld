#!/usr/bin/env python3
"""后端/管线修复回归（v2.15.8）。

    python tools/test_backend_fixes.py

对应外部审查报告 `findings_backend.md` 的 26 条。纪律：**每条修复都必须有一条
会 fail 的断言**，不接受"看截图/看日志"式的软检查。

分两类：
  · 静态断言：源码里不允许再出现某段旧写法（防止回退）
  · 功能断言：直接调用函数跑小输入，验证行为（build_solid 封边、build_points
    抽稀、ckpt_meta 键名、阈值单一来源、EXIF 参数）
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "pipeline", "v2"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402


def read(p: str) -> str:
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return f.read()


def code_only(p: str) -> str:
    """去掉行注释后的源码。

    很多修复的注释里会**引用旧写法**做说明（"这里原来写 xxx"），
    静态断言必须只看代码，否则会被自己的注释绊倒。
    """
    out = []
    for line in read(p).splitlines():
        q = line.split("#", 1)[0]
        out.append(q)
    return "\n".join(out)


PASS = []


def ok(msg: str) -> None:
    PASS.append(msg)
    print("PASS  " + msg)


# ---------------------------------------------------------------- P0 / pipeline
def test_reconstruct_p0():
    src = read("pipeline/reconstruct.py")
    # 2.1：bbox 不能再引用未定义的 pos
    assert not re.search(r'"min":\s*pos\.min', src), \
        "P0 回归：reconstruct.py 的 bbox 仍在引用未定义的 `pos`（world.json 永远不落盘）"
    assert '"min": pos_m.min' in src, "P0：bbox 应改用 pos_m"
    # 4.3：只允许一个 "camera" 键
    n = len(re.findall(r'^\s*"camera":\s*\{', src, re.M))
    assert n == 1, 'P2 回归：meta_out 里有 %d 个 "camera" 键（重复键会静默覆盖 eye_height）' % n
    # 3.11：三入口都要按 EXIF 摆正
    assert "ImageOps.exif_transpose" in src, "P1 回归：reconstruct.py 没按 EXIF 摆正"
    # 4.2：不能再在 resize 后的图上重算 hfov
    assert "hfov, _, m = hfov_from_image(img)" not in src, \
        "P2 回归：reconstruct.py 又在缩图后重算 hfov（覆盖 load_photo 的正确值）"
    ok("P0-2.1 / P2-4.2 / P2-4.3 / P1-3.11  reconstruct.py")


def test_api_py():
    src = read("pipeline/v2/api.py")
    assert re.search(r"^\s*sem = None$", src, re.M), \
        "P1 回归：api.py 的 sem 没初始化（语义关闭时 bake 抛 NameError）"
    assert "ImageOps.exif_transpose" in src, "P1 回归：api.py 没按 EXIF 摆正"
    assert "tempfile.mkstemp" in src, "P1 回归：GLB 临时文件又回到固定名（并发会读到半截）"
    assert '"_tmp_out.glb"' not in src, "P1 回归：仍存在固定名 _tmp_out.glb"
    # 3.1：pad 必须乘缩放比
    assert re.search(r"_sx = disp\.shape\[1\] / max\(1\.0, float\(src_size\[0\]\)\)", src), \
        "P1 回归：标定裁剪的 pad 没按 disp/src 的比例缩放"
    # 2.2：点云分支不得再引用 seg
    assert "!= seg.shape" not in src, "P0 回归：api.py 仍用 seg.shape 对齐掩码"
    ok("P1-3.1 / P1-3.2 / P1-3.3 / P1-3.11（api.py）/ P0-2.2")


def test_ldi():
    src = read("pipeline/v2/ldi.py")
    # 3.5：两处都要用降采样的 sky_s
    # 只看代码，不看注释（注释里会引用旧写法做说明）
    assert not re.search(r"if sky is not None and sky\.shape ==", src), \
        "P1 回归：ldi.py 又用全分辨率 sky 与降采样网格比形状（恒 False → 死代码）"
    assert src.count("if sky_s is not None:") >= 2, "P1：两处应都改用 sky_s"
    # 4.5：抽稀步长必须 ceil
    assert "int(max_points / pos.shape[0])" not in src, \
        "P2 回归：build_points 的抽稀步长又写成 int(N/max)（超预算时恒 0 → 永不抽稀）"
    assert "np.ceil(pos.shape[0] / float(max_points))" in src, "P2：抽稀步长应为 ceil"
    # 4.4：限幅必须在 cliff_mask 之前
    i_cap = src.index("zs = cap_depth_gradient(zs")
    i_cliff = src.index("cliff, ratio, edge_str = cliff_mask(")
    assert i_cap < i_cliff, \
        "P2 回归：cap_depth_gradient 又跑到 cliff_mask 之后（cliff 掩码基于未限幅深度）"
    ok("P1-3.5 / P2-4.4 / P2-4.5（ldi.py 静态）")


def test_build_solid_skirt():
    """3.6：封边必须是**活的** —— 旧实现的判据互斥，一条也建不出来。"""
    from ldi import build_solid

    h = w = 12
    mask = np.zeros((h, w), dtype=bool)
    # build_solid 要求 ≥24 个像素（ys.size < 24 直接返回 None），用 6×6
    mask[3:9, 3:9] = True
    P = np.zeros((h, w, 3), dtype=np.float32)
    uv = np.zeros((h, w, 2), dtype=np.float32)
    z = np.full((h, w), 10.0, dtype=np.float32)
    for y in range(h):
        for x in range(w):
            P[y, x] = (x * 0.1, y * 0.1, -10.0)
            uv[y, x] = (x / w, y / h)

    out = build_solid(P, uv, mask, z)
    assert out is not None, "build_solid 返回 None"
    n_idx = len(out["index"])
    assert n_idx % 3 == 0

    # 内部 quad 数（正+背各一份）
    inner = sum(1 for y in range(h - 1) for x in range(w - 1)
                if mask[y, x] and mask[y + 1, x] and mask[y, x + 1] and mask[y + 1, x + 1])
    base = inner * 2 * 6
    assert n_idx > base, ("P1 回归：build_solid 的封边仍是死代码"
                          "（总索引 %d = 正/背 %d，没有任何封边 quad）" % (n_idx, base))

    # 封边 quad 数 = 轮廓边数。按"中心到中心"的边算，n×n 像素块是 4(n-1) 条：
    #   竖边 n 列 × (n-1) 行，只有首尾两列是轮廓边 → 2(n-1)；横边同理 → 共 4(n-1)。
    #   （角上的"缺失边"两端是同一个顶点，退化，不需要单独封。）
    skirt = (n_idx - base) // 6
    assert skirt == 4 * (6 - 1), "封边 quad 数应为 20（6×6 方块），实测 %d" % skirt
    ok("P1-3.6  build_solid 封边 %d 个 quad（6×6 方块 = 4(n-1) 条轮廓边）" % skirt)


def test_build_points_thinning():
    """4.5：抽稀必须真的发生，且 size 必须同步切片（否则 glTF accessor 非法）。"""
    from ldi import build_points

    h = w = 40
    P = np.zeros((h, w, 3), dtype=np.float32)
    P[..., 2] = -10.0
    rgb = np.full((h, w, 3), 0.5, dtype=np.float32)
    valid = np.ones((h, w), dtype=bool)
    out = build_points(P, rgb, valid, step_used=1, max_points=400)
    n = out["pos"].shape[0]
    assert n <= 400, "抽稀没生效：%d 点 > 上限 400" % n
    assert out["size"].shape[0] == n, \
        "size 与 pos 点数不同步（%d vs %d）→ 非法 glTF" % (out["size"].shape[0], n)
    assert out["color"].shape[0] == n
    ok("P2-4.5  build_points 抽稀到 %d 点，size/color 同步" % n)


def test_depth_polarity():
    src = read("pipeline/depth.py")
    assert "d = 1.0 - d" not in src, \
        "P2 回归：极性翻转又回到加性 `1-d`（破坏 z ∝ 1/r 的乘性标定）"
    assert "polarity_confident and rho < 0" in src, "P2：翻转为乘性且需置信"
    ok("P2-4.1  depth 极性改为乘性翻转（1/d 再归一）")


def test_exif_sensor_size():
    from exif import hfov_from_image

    class FakeImg:
        size = (4000, 3000)
        def getexif(self):
            return {}

    a = hfov_from_image(FakeImg())[0]
    b = hfov_from_image(FakeImg(), sensor_size=(3000, 4000))[0]
    assert abs(a - b) < 1e-9, "sensor_size 参数未生效（无 EXIF 时两者应一致）"
    ok("P1-3.11  hfov_from_image 接受存储朝向尺寸")


def test_exif_portrait_fov():
    """P1（本轮）：竖拍 ±90° 的 FOV 口径与显示尺寸必须随旋转换算。

    26mm 等效、120×80 存储图、Orientation=6（竖拍显示 80×120）：
      · 横拍口径 hfov=2atan(36/52)=69.39°、vfov(80,120)=92.16°
      · 竖拍口径 hfov=2atan(24/52)=49.55°、vfov(80,120)=69.39°
    修复前两个调用点都没传 rotated90，会用横拍口径 → hfov 偏大 20°。
    """
    import math
    from PIL import Image
    from exif import hfov_from_image, vfov_of

    im = Image.new("RGB", (120, 80))
    ex = Image.Exif()
    ex[0xA405] = 26            # FocalLengthIn35mmFilm
    # PIL 需通过 save 往返才能让图片带上 getexif
    import io
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=ex)
    buf.seek(0)
    img = Image.open(buf)

    hf_land = hfov_from_image(img, sensor_size=(120, 80), rotated90=False)[0]
    hf_port = hfov_from_image(img, sensor_size=(120, 80), rotated90=True)[0]
    exp_land = 2 * math.atan(36.0 / (2 * 26))
    exp_port = 2 * math.atan(24.0 / (2 * 26))
    assert abs(hf_land - exp_land) < 1e-6, \
        "横拍 35mm 口径被破坏：%.4f != %.4f" % (hf_land, exp_land)
    assert abs(hf_port - exp_port) < 1e-6, \
        "竖拍仍按 36mm 长边算 hfov（应换 24mm 短边）：%.4f != %.4f" % (hf_port, exp_port)
    # 显示朝向 80×120：竖拍 vfov 应回到 69.39°，而不是 92.16°。
    vf_port = vfov_of(hf_port, 80, 120)
    assert abs(vf_port - exp_land) < 1e-6, \
        "竖拍 vfov 口径错：%.4f != %.4f" % (vf_port, exp_land)
    ok("P1（本轮）  EXIF 竖拍 ±90° FOV 按短边换算（69.39°→49.55°）")


def test_exif_portrait_resize():
    """P1（本轮）：reconstruct.load_photo 缩放按显示朝向，竖图不能被压扁。"""
    import io
    from PIL import Image
    import reconstruct

    img = Image.new("RGB", (120, 80), (120, 140, 160))
    ex = Image.Exif()
    ex[0x0112] = 6              # Orientation=6 → transpose 后显示 80×120
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=ex)
    path = os.path.join(ROOT, "out", "_exif_portrait_tmp.jpg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.getvalue())
    try:
        out, arr, _src, meta = reconstruct.load_photo(path, max_side=60)
        # 摆正后是 80×120（竖），等比缩到 max=60 → 40×60，绝不能是 60×40。
        assert out.size == (40, 60), \
            "竖拍 resize 仍用存储宽高：得到 %s，应为 (40, 60)" % (out.size,)
        assert arr.shape[:2] == (60, 40), "arr 形状与显示尺寸不一致：%s" % (arr.shape,)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    ok("P1（本轮）  load_photo 竖拍 resize 按显示朝向（60×40→40×60）")


def test_geom_meta():
    src = read("pipeline/geom.py")
    assert not re.search(r'"n":\s*int\(r_n\)', code_only("pipeline/geom.py")), \
        "P2 回归：geom 又把视差当样本数写进 n"
    ok("P2-4.6  geom 退化分支的 n 字段不再失真")


# ---------------------------------------------------------------- 工具链
def test_tools():
    t = code_only("tools/tune_per_image.py")
    assert "text=True" not in t, "P1 回归：tune_per_image 又用 text=True（GBK 会崩掉整轮）"
    sp = code_only("tools/score_params.py")
    assert "tasklist" not in sp, "P2 回归：锁探活又回到 Windows 专用的 tasklist"
    assert "os.kill(pid, 0)" in sp, "P2：锁探活应改 os.kill(pid, 0)"
    assert "获取渲染锁超时" in sp, "P2：等锁超时必须报错，不能无锁裸跑"
    ok("P1-3.12 / P2-4.11  工具链")


def test_threshold_single_source():
    import acc_report
    import score_params
    for k in ("stripe", "innerVoid", "voidFrac", "lowVoid"):
        assert abs(acc_report.INNER_T.get(k, acc_report.OUTER_T.get(k)) - score_params.THRESH[k]) < 1e-9, \
            "P2 回归：acc_report 与 score_params 的阈值 %s 不一致" % k
    ok("P2-4.9  验收与评分阈值单一来源")


# ---------------------------------------------------------------- worldmodel
def test_ckpt_meta_key():
    from worldmodel.ckpt_meta import resolve_hist_frames
    n, warn = resolve_hist_frames({"data": {"n_hist": 6}}, 4)
    assert n == 6 and warn, "P1 回归：resolve_hist_frames 没读 n_hist（保护仍是死代码）"
    n2, warn2 = resolve_hist_frames({"data": {"hist_frames": 5}}, 5)
    assert n2 == 5 and warn2 is None, "兼容键 hist_frames 应仍然可用"
    ok("P1-3.13  ckpt_meta 读 n_hist，保护生效")


def test_trainer_fixes():
    src = read("worldmodel/train/trainer.py")
    # 3.8：noise_proxy 不得再引用目标帧 z1
    m = re.search(r"z_self = \(1 - alpha\) \* z1", src)
    assert m is None, "P1 回归：noise_proxy 又把目标帧 z1 泄漏进历史"
    assert "z_self = z_hist[:, -1] + alpha" in src, "P1：z_self 应扰动真历史帧"
    # 3.7：resume 恢复 opt/sched
    assert 'opt.load_state_dict(ck["opt"])' in src, "P1 回归：续训不恢复 optimizer"
    assert 'sched.load_state_dict(ck["sched"])' in src, "P1 回归：续训不恢复 scheduler"
    assert '"opt": opt.state_dict()' in src, "P1：checkpoint 未保存 optimizer"
    # 4.8：历史噪声退火
    assert "progress" in src and "p_eff" in src, "P2 回归：历史噪声增强没有退火调度"
    ok("P1-3.7 / P1-3.8 / P2-4.8  trainer")


def test_data_and_eval():
    pd = read("scripts/prepare_data.py")
    assert "index[\"action_scale\"]" in pd, "P2：prepare_data 没写 action_scale（动作量纲差 20 倍）"
    assert "for i in range(0, n - 1):" in pd, \
        "P1 回归：数据窗口又从 i=N_HIST 开始（rollout 冷启动历史仍是分布外）"
    assert "frames[max(0, i - N_HIST + 1 + k)]" in pd, "P1：历史不足应用首帧补齐"
    assert "for i in range(N_HIST" not in pd, "P1 回归：窗口起点又回到 N_HIST"
    # 4.10：hist_actions 末项不得等于目标动作
    assert "j = i - N_HIST + k" in pd, "P2 回归：hist_actions 又包含目标动作本身"
    ev = read("scripts/eval.py")
    assert "torch.zeros(7)], args.ckpt" not in ev, \
        "P1 回归：eval 一步预测又喂 idle 动作（指标不可信）"
    assert "rel_action(_Ts[5], _Ts[6]" in ev, "P1：eval 应喂真实相对动作"
    ok("P1-3.9 / P1-3.10 / P2-4.7 / P2-4.10  数据与评测")


def main() -> int:
    tests = [
        test_reconstruct_p0, test_api_py, test_ldi,
        test_build_solid_skirt, test_build_points_thinning,
        test_depth_polarity, test_exif_sensor_size,
        test_exif_portrait_fov, test_exif_portrait_resize, test_geom_meta,
        test_tools, test_threshold_single_source,
        test_ckpt_meta_key, test_trainer_fixes, test_data_and_eval,
    ]
    for t in tests:
        t()
    print("\n全部通过：%d 项" % len(PASS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
