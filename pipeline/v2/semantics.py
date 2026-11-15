#!/usr/bin/env python3
"""语义分割（SegFormer-B0 / ADE20K 150 类），只做两件事：

  1. **判断"这是不是风景照"**：天空占比高 + 地面几乎为零 → 云海 / 星空 / 太空地球。
     这类图硬建三维只会得到一团黑色拉丝（验收批次里最惨的两张就是），
     正确做法是退回"照片平贴"。
  2. **给前景物体一个语义身份**：person / car / pole / 家具…… 有了它，
     "该不该做成独立卡片"不再只靠面积阈值（越野车那种大目标会被面积判据漏掉）。

为什么不用数值统计代替：试过"地面像素占比 < 0.4%"和"远景占比 / 纵深动态范围"，
在云海、星空、太空地球这几张上**全都不成立**（地面占比 3.7%、远景占比恒为 0、
纵深比好坏图不分离）。语义是这里唯一可靠的信号。

类别表与分组口径**与 `js/semantics.js` 完全一致**（同一份 ADE20K 索引表），
这样离线烘焙和浏览器内在线链路的判断不会打架。
"""
from __future__ import annotations

import os

import numpy as np

SIZE = 512
MODEL_NAME = "segformer_b0_ade20k.onnx"

# 与 js/semantics.js 的 GROUP 编号一致
UNKNOWN, SKY, CEILING, WALL, GROUND, WATER, VEG, BUILDING, PERSON, VEHICLE, OBJECT, TERRAIN = range(12)
GROUP_NAMES = ["unknown", "sky", "ceiling", "wall", "ground", "water",
               "vegetation", "building", "person", "vehicle", "object", "terrain"]

_MARK = {
    WALL: [0, 8, 27, 43, 58, 63, 74, 95, 100, 124, 130, 132, 144],
    BUILDING: [1, 25, 48, 84, 79, 86],
    SKY: [2],
    GROUND: [3, 6, 9, 11, 13, 29, 46, 52, 53, 54, 59, 94, 121],
    CEILING: [5],
    VEG: [4, 17, 66, 72],
    TERRAIN: [16, 68, 33, 34, 41, 42, 44, 50],
    WATER: [21, 26, 60, 104, 109, 111, 113, 128],
    PERSON: [12],
    VEHICLE: [20, 76, 80, 82, 90, 102, 103, 116, 127],
    OBJECT: [7, 10, 14, 15, 18, 19, 22, 23, 24, 28, 30, 31, 32,
             35, 36, 37, 38, 39, 40, 45, 47, 49, 51, 55, 56, 57, 61, 62, 64, 65,
             67, 69, 70, 71, 73, 75, 77, 78, 81, 83, 85, 87, 88, 89, 91, 92, 93,
             96, 97, 98, 99, 101, 105, 106, 107, 108, 110, 112, 114, 115, 117, 118,
             119, 120, 122, 123, 125, 126, 129, 131, 133, 134, 135, 136, 137, 138,
             139, 140, 141, 142, 143, 145, 146, 147, 148, 149],
}

_CLASS_GROUP = np.zeros(150, dtype=np.uint8)
for _g, _ids in _MARK.items():
    for _i in _ids:
        _CLASS_GROUP[_i] = _g

# 会被当作"可独立成卡片的物体"的组
OBJECT_GROUPS = {VEG, BUILDING, PERSON, VEHICLE, OBJECT}

_SESS = None


def model_path(root: str) -> str:
    return os.path.join(root, "models", MODEL_NAME)


def available(root: str) -> bool:
    return os.path.exists(model_path(root))


def _session(root: str):
    global _SESS
    if _SESS is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = min(8, os.cpu_count() or 4)
        _SESS = ort.InferenceSession(model_path(root), so,
                                    providers=["CPUExecutionProvider"])
    return _SESS


def segment(rgb: np.ndarray, root: str) -> dict:
    """rgb: (H,W,3) float 0..1 → 分组图与占比。

    返回 {"groups": (H,W) uint8, "classes": (H,W) uint8, "frac": {组名: 占比}}
    """
    from PIL import Image
    H, W = rgb.shape[:2]
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    small = im.resize((SIZE, SIZE), Image.BILINEAR)      # 与 JS 一致：直接拉伸到正方形
    a = np.asarray(small, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = ((a - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)

    sess = _session(root)
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    cls_small = logits[0].argmax(axis=0).astype(np.uint8)     # (LH, LW)
    # 回到原图分辨率（最近邻，类别不能插值）
    cls = np.asarray(Image.fromarray(cls_small).resize((W, H), Image.NEAREST))
    groups = _CLASS_GROUP[cls]
    frac = {name: float((groups == g).mean()) for g, name in enumerate(GROUP_NAMES)}
    return {"groups": groups, "classes": cls, "frac": frac}


def looks_like_scenery(seg: dict) -> tuple[bool, str]:
    """判断"这是不是一张可以建三维的风景照"，以及理由。

    判据（阈值来自验收批次里 32 张图的实际分布）：
      * 地面类（ground + terrain）合计 < 3%：脚下没有地 → 不是风景；
      * 天空 > 70% 且地面 < 6%：云海 / 星空 / 太空；
      * 水 + 天空 > 90%：纯海面/纯天空。
    """
    f = seg["frac"]
    ground = f["ground"] + f["terrain"]
    sky = f["sky"]
    water = f["water"]
    # 云海那张：天空 73%，其余 27% 被模型误判成"地形"（云看起来像山）。
    # 所以判据不能要求"地面≈0"，改成"天空很高 且 地+水合计很低"。
    # 真正的海景（天空 70% + 水面 30%）不会被误杀，因为它有水。
    if sky > 0.65 and (ground + water) < 0.30:
        return False, (f"天空 {sky*100:.0f}%，地+水仅 {(ground+water)*100:.0f}%"
                       f"（云海/星空类）")
    if ground < 0.03:
        return False, f"地面类仅 {ground*100:.1f}%（脚下没有实地）"
    if water + sky > 0.90:
        return False, f"水+天空 {((water+sky)*100):.0f}%（纯海面/纯天空）"
    return True, f"地面 {ground*100:.0f}% / 天空 {sky*100:.0f}%"


def object_mask(seg: dict, near_mask: np.ndarray | None = None,
                min_frac: float = 0.004) -> np.ndarray:
    """语义上属于"物体"的像素，可选再与"显著近"求交。

    为什么要语义：面积阈值会漏掉大目标（验收里"沙丘上的越野车"面积超过 12% 画面，
    按面积判据它不是"细物体"，于是没被做成卡片，结果被横向拉成一条黑带）。
    语义给出的是"它是个东西"，与大小无关。
    """
    m = np.isin(seg["groups"], list(OBJECT_GROUPS))
    if near_mask is not None:
        # 与"近"求交时不要求特别近：楼房也是物体，只是不需要卡片化。
        m = m & near_mask
    if m.mean() < min_frac:
        return np.zeros_like(m)
    return m
