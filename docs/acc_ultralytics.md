# Ultralytics 文档通读结论：有没有能用于"模型生成"的？

> 结论先说：**没有生成模型**（没有扩散、没有 3D 生成、没有 Gaussian Splatting）。
> 它是**感知（perception）**工具箱。但其中有两项能直接喂进我们的"照片 → 世界"管线，
> 一项值得用（分割做掩码），一项实测反而更差（米制深度）。

## 文档覆盖范围

| 板块 | 内容 | 与"生成世界"的关系 |
|---|---|---|
| Tasks | 检测 / 实例分割 / 语义分割 / **深度估计** / 分类 / 姿态 / OBB | 分割、深度**可用** |
| Models | YOLO26、YOLO11、YOLO27 预览、**SAM 3**、RT-DETR、FastSAM | SAM 可用 |
| Modes | train / val / predict / export / track / benchmark | 无关（部署向） |
| Solutions | 计数、热力图、安防、健身计数 | 无关 |
| Platform | 云端标注 / 训练 / 部署 | 无关 |

**没有**：diffusion、图像生成、NeRF、3DGS、mesh 重建。想"生成"只能用我们自己的
SDXL / COLMAP / TSDF 那条线。

## 直接可用的两项

### 1) 米制单目深度（`yolo26*-depth.pt`）—— 已接入，但实测更差

官方文档关键点：深度头是 `exp(logit)` 无界（0.02~150m），**输出即米**，权重内置全局尺度
校准；219 万张室内+室外训练，NYU v2 δ1=0.882 / RMSE=0.414m（n 版）。

实测（lego 800×800）：

```
输出 1.04~2.54 m，主体中位 1.17m、背景 1.66m，遮挡边界很干净（out/v4/yolo_depth_vis.jpg）
```

**接进 v4 的结果：更差。** `--depth yolo --mesh tsdf` 出来的是一排竖直薄片
（`out/v4/lego_yolo_offline.png`），fill 从 15% 变 29% 但形状散架。
原因：它给的是"每帧自己的米制尺度"，而 COLMAP 的尺度是全局单值；我们按帧拟合
`a = median(z_sfm / depth_m)`，帧间 `a` 的偏差直接把点云拉成扇面。

→ `--depth yolo` 保留为实验开关，**默认仍是 `da`（Depth-Anything）**。
要真用起来，得把 `a` 改成**全局单尺度**（所有帧联合拟合一个 `a`），这一步没做。

### 2) SAM 3 / FastSAM 分割 —— 值得用，尚未接

价值很明确：**给 SfM 做背景掩码**。我们的 lego 素材背景是一大片纯灰，
SIFT 在上面找不到特征、稠密匹配也会在背景上产生大量外点。用 SAM 把主体掩出来，
只在对主体区域提特征/匹配，理论上能显著提升点云密度与位姿精度。

次要用途：v3 的 `--bg-fill`（现在用 SD 重绘背景）可以先用 SAM 得到精确前景掩码，
再让扩散只补背景。

## 其他不适用于本项目的部分

* **检测/跟踪/计数/热力图**：与静态重建无关；
* **导出（ONNX/TensorRT/OpenVINO…）**：我们不做边缘部署；
* **Ultralytics Platform**：需要账号与云端，违背"本机离线"的前提；
* **许可**：AGPL-3.0，商用需企业许可 —— 我们自用没问题，但要注意。

## 权重下载的坑

`ultralytics` 自动下载权重会在本机代理下报
`SSLError CERTIFICATE_VERIFY_FAILED`。绕过方式（已验证）：

```bash
python tools/hf_fetch.py \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-depth.pt \
  tools/weights/yolo26n-depth.pt
```

然后 `YOLO_DEPTH_WEIGHT=tools/weights/yolo26n-depth.pt` 或直接按默认路径放。

## 下一步（按价值排序）

1. **SAM 掩码进 SfM**：`yolo11n-seg.pt` 或 SAM3 → 每帧前景掩码 → 只在掩码内提特征/匹配
   → 对比点云数量与光度一致性。这是最可能带来实质提升的一项。
2. **全局单尺度拟合**：把 `--depth yolo` 的 per-frame 尺度改成全局最小二乘，
   米制深度就能用上（顺带把世界标定成米）。
3. 语义分割做"天空/地面"分区：天空不参与 SfM、地面用平面约束 —— 对户外素材有用。
