# 开源件选型（2026-11-17）

> 目的：把"该自己写"和"该直接用开源"分开。本仓库的纪律是零构建、零依赖、纯 CPU 优先，
> 所以只收**许可证干净 + 能进浏览器或本地 GPU 服务**的件。
> 核实状态：`已核实` = 本轮通过公开资料/HF 模型卡确认；`待核实` = 来自外部报告，本轮未独立验证。

## 1. 现状对照

| 环节 | 现在用的 | 许可 | 结论 |
|---|---|---|---|
| 相对深度（浏览器内） | Depth-Anything-V2-**Small** ONNX 量化 27MB | Apache-2.0（已核实） | 保持。**注意：Base/Large/Giant 是 CC-BY-NC**，不可商用 |
| 语义分组 | SegFormer-B0 / ADE20K int8 4.4MB | 开源（随仓库分发） | 保持 |
| 实例切分 | YOLOv8n-seg | AGPL-3.0（Ultralytics） | **商用需注意**，见 §4 |
| 补洞/擦除 | MI-GAN 512 ONNX 29.5MB | MIT（已核实，© Picsart AI Research） | 保持；可选换官方 `migan_pipeline_v2.onnx`（整图+mask 一次出，省前后处理） |
| 点云/网格渲染 | 自研 WebGL2 splat | 自有 | 保持（零依赖是本项目卖点） |
| v2 离线烘焙深度 | Depth-Anything-V2-**Large**（`models/da2_large/`） | **CC-BY-NC**（已核实） | **商用红线**：要么只在本地研究用，要么换 MoGe-2 |
| 视锥外内容 | 程序化 fBm + 平铺 | 自有 | 只能兜底，不是补全 |

## 2. 建议接入：MoGe-2（MIT）——一次前向给"米制几何 + 内参"

这是本仓库最缺的一样东西。现在深度是**相对**的，靠"眼高 1.7m + 远锚 520/1500m + 鱼眼修正"
反解米制，FOV 固定 62°（无 EXIF 时按 26mm 猜），误差一层层放大：
报告里"10% 深度误差 → 27~32% 距离误差"就是这个链条。

- 仓库：[`microsoft/MoGe`](https://github.com/microsoft/MoGe)，**代码与权重均 MIT**（已核实）
- 权重（HF，`Ruicheng/moge-2-*-normal`）：`vitl` 331M / `vitb` 104M / **`vits` 35M**（已核实参数量）
- 一次前向输出：`points`（**米制**点图）、`depth`、`normal`、`mask`、`intrinsics`（归一化内参 → FOV）
- ONNX：官方 `docs/onnx.md` 提供了导出路径；社区已有转换好的现成品
  （`DakeQQ/Vision-Metric-Depth-Distance-Estimation-ONNX`，仓库 Apache-2.0，已核实存在）
- 成本参考（第三方实测，已核实来源）：720×1280，`MoGe-v2-vits` f32 在 i7 CPU 3.4s、
  MX150 独显 1.5s；`DepthAnything-v2-vits` 在 294×518 下 0.26~0.32s

**接入点（如果要做，按这个顺序）**

1. 新增 `js/moge.js`：ORT session（优先 WebGPU，回退 WASM），输入缩放到 14 的倍数、RGB/255；
2. 用输出的 `intrinsics` 替换 `HFOV = 62°` 这个常量（`world.js` 顶部 + `buildCartesianWorld` 的 `tanH`）；
3. 用米制 `depth` 直接替掉 `calibrateScale()` 的双锚点链（保留旧链做 `?calib=legacy` 回退）；
4. 法线图接到顶点法线（现在是从世界坐标邻居差分算的，噪声大）；
5. **验收**：`multiview.mjs` 的 `stripe/lowVoid` 之外，必须加一项"同机位比例误差"——
   拿 `worlds/*` 里已知的拍摄距离做人工对照，否则只会在指标上自嗨。

**为什么本轮没做**：权重不在本工作区（vits f32 约 140MB），接一个新 ONNX 后端要同时改
深度前处理、标定链、法线、缓存 key 与四条降级路径，属于 1~2 天的独立改动，和本轮的
"修静默失效 + 压拉丝"混在一起会让 A/B 失去归因能力。

## 3. 选型表：其余环节

| 需求 | 候选（许可） | 核实 | 备注 |
|---|---|---|---|
| 天空/前景抠图（替代启发式洪水填充） | **BiRefNet / BiRefNet_lite 512**（MIT，已核实，含浏览器 ONNX 导出） | 已核实 | 比 ADE20K 的 sky 类边界细得多；体积 ~40~90MB |
| 全景补全（视锥外内容） | **FLUX.2 [klein] 4B**（报告称 Apache-2.0） / Qwen-Image（Apache-2.0） / HY-Pano 2.0 | 待核实 | 必须服务端 GPU；中心原图区域要严格保留像素 |
| 单图→3DGS | WorldStereo / WonderWorld 思路（论文级） | 待核实 | 工程量大，先做全景 outpaint 再谈 |
| 3DGS 浏览器渲染 | **Spark.js**（WebGPU）/ GaussianSplats3D（WebGL）/ antimatter15/splat | 待核实 | 自研 splat 在 20 万点以内够用；到百万级再换 |
| 内参/重力交叉校验 | **GeoCalib**（ECCV'24） | 待核实 | 只在有 EXIF 或需要重力方向时值得接 |
| 图像修复/去水印 | LaMa（Apache-2.0）/ MI-GAN（MIT，已内置） | MI-GAN 已核实 | 先用手上的 |

## 4. 许可证红线（本轮已核实项）

- **Depth-Anything-V2**：`Small` = Apache-2.0；`Base/Large/Giant` = **CC-BY-NC-4.0**。
  本仓库 `pipeline/v2` 下的是 Large → **只能研究/自用**。要商用必须换 Small 或 MoGe-2。
- **MoGe-2 / MoGe-3**：代码与权重均 **MIT**，可商用。
- **MI-GAN**：**MIT**（© 2024 Picsart AI Research），可商用，注意随分发附 LICENSE。
- **YOLOv8n-seg（Ultralytics）**：**AGPL-3.0**。自用/开源没问题；闭源商用要买授权，
  或换 Apache-2.0 的分割模型（如 BiRefNet / SAM 系列权重）。
- **待核实**：FLUX.2 [klein]、Qwen-Image、GeoCalib、Spark.js、HY-Pano 的商用条款，
  上线前必须逐条读原仓库 LICENSE，本文件不作为法律意见。

## 5. 明确不建议再投入的路线（与外部报告一致）

- 再上生成模型**补"天空后面"**：天空没有视差，不是几何，补出来也是贴纸；
- 完整 LDI 全量重建：双层窄带能拿到大部分收益，全量 LDI 的工程量不划算（且本轮实测窄带无收益）；
- 更大的深度/分割模型：问题不在精度，在"边界与天空的语义没进几何管线"；
- 时域累积/超分：漫游下拖影明显，与纯 CPU 预算冲突。
