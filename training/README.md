# 训练与数据（v1.16）

本目录是「把 v1.16 的工程骨架喂成高质量生成系统」的训练侧说明。
四条能力分别对应四个可独立训练/替换的模型，互不阻塞：

| 能力 | 模型 | 是否开源 | 训练数据 | 现状 |
|---|---|---|---|---|
| 语义分类（树/建筑/天空/地面/天花板） | SegFormer-B0 / ADE20K | ✅ | ADE20K 150 类 | 直接跑预训练 ONNX，可选领域微调 |
| 单体单图转 3D | InstantMesh / TripoSR | ✅ | Objaverse / COCO 裁片 | 服务端接入点已留 |
| 无限地貌延伸 | Terrain Diffusion / InfiniteDiffusion | ✅ SIGGRAPH'26 | DEM / 高度图 | 服务端接入点已留，程序化兜底可用 |
| 动作可控世界模型 | Causal World DiT（本仓库） | ✅ | 合成轨迹 + 真实视频 | 原型已可训练 |

> 注意：`../train/` 是**早期废弃实验**的存档，与本目录无关，不要混用。

---

## 0. 先整理数据

```bash
# 把收集到的图集（风景 / 室内 / 360 全景）整理成 512×384、去重、切分
python training/prepare_landscape_dataset.py --src D:/datasets/raw --out data/landscapes
# 含 2:1 全景时，切成 4 个 90° 透视视角，天然带 360° 监督
python training/prepare_landscape_dataset.py --src D:/datasets/raw --out data/landscapes --panorama
```

**推荐数据源**（均可公开下载）：

- **ADE20K**（语义）：150 类，室内+户外，SegFormer 的原生训练集。
- **Places365 / COCO-Stuff**（语义泛化）：场景多样，补 ADE 的类别偏置。
- **DL3DV-10K / RealEstate10K**（世界模型）：真实多视角视频，位姿可用 VGGT/COLMAP 估。
- **360° 全景**（地貌/世界）：等距柱状图切透视视角，是「无限延伸」最便宜的 360° 监督。
- **DEM / 地形高度图**（Terrain Diffusion）：USGS/SRTM 30m 高程，仓库 30m 模型即基于此类数据。

---

## 1. 语义分割（修「天空/地面/天花板/树混淆」）

预训练权重已经够用，前端直接量化推理：

```bash
python tools/fetch_models.py --semantics     # segformer_b0_ade20k.onnx（~15MB）
```

若图集以某一类场景为主（雪林、海岸），做轻量领域适配：用 ADE20K 预训练权重，
在自建标注子集上微调 1~2 小时（单卡即可）。

> 经验：ADE20K 的 `tree` 在密集针叶林上召回一般。若侧拍发现树被当地形，
> 优先补 500~1000 张雪林/针叶林标注，比换更大模型收益高。

---

## 2. 单体转 3D（修「树/房屋被拉成鳍」）

默认走账单牌（零依赖、不拉伸）。要「各视角都合理」，接 InstantMesh：

```bash
git clone https://github.com/TencentARC/InstantMesh && cd InstantMesh
pip install -r requirements.txt
# 权重按仓库说明放 ckpts/
```

服务端 `server/object3d.py::generate` 已留接入点：裁片 → 前向 → 导出 GLB。
前端 `?obj3d=1&api=...` 会用它替换账单牌，并调用 `world.hideObjectCards(ids)`。

> 训练提示：树/建筑属于「细长/箱体」两类，通用 Objaverse 模型偏圆润。
> 用 COCO 裁片 + 少量自建树/房屋多视角数据微调 2~3k 步，轮廓会明显更准。

---

## 3. 无限地貌延伸（核心：扇环外自动生长）

服务端默认已用「全局连续 fBm + 四边硬约束」，能保证接缝一致。要 learned 质量：

```bash
git clone https://github.com/xandergos/terrain-diffusion
cd terrain-diffusion && pip install -r requirements.txt
# 需要 CUDA 版 torch；30m 模型在 HF: xandergos/terrain-diffusion-30m
```

`server/terrain.py::terrain_diffusion_extend` 已按 `WorldPipeline` 写好适配，
若该仓库 API 变动，只改这一处。`/health` 的 `terrain_diffusion` 会变 `true`，
`/api/extend` 的 `source` 会从 `procedural` 变成 `terrain_diffusion`。

要点：
- **确定性**：同一 `seed + tile 坐标` 必须生成同一结果（InfiniteDiffusion 的核心性质），
  否则走回头路会看到地貌跳变。
- **边界条件**：tile 四边必须与邻居严格一致，`terrain.py` 的带状混合 + 硬约束负责收边。
- **风格一致**：把「照片重建区域的地貌统计」（起伏幅度、主色、雪/沙）写进请求，
  否则生成区会与照片区风格断裂。

---

## 4. 动作可控世界模型（可选，Oasis 式）

本仓库 `worldmodel/` 已有 Causal World DiT 原型（129M）。放大训练：

```bash
node scripts/render_traj.mjs --glb assets/world-alpine.glb --out data/syn --trajs 400 --frames 64
python scripts/prepare_data.py --traj data/syn --out data/wm --shard 500
HF_HUB_OFFLINE=1 python -m worldmodel.train --config configs/s1_3000.yaml
```

真实数据混训：DL3DV/RealEstate10K 抽帧 + VGGT 估位姿，写成同一 `poses.json` schema，
`meta.source=real`，评测分层统计。

> 显存参考：8G 卡 batch 2 / grad_accum 16；4090 24G 可 batch 8~12。
> 浏览器端实时生成不现实（见 `server/README.md`），本模型定位是服务端 rollout / 离线质检。

---

## 5. 质检：每张图都要能侧拍

```bash
node tools/orbit_shot.mjs \
  --url "http://127.0.0.1:8099/?w=alpine&instant=1&api=http://127.0.0.1:8100" \
  --out shots/alpine --angles 8 --radius 14 --center 0,-30 --pitch -0.18
```

产出 `orbit_00..07.png` + `orbit_top.png`，重点看：

1. **单体**：侧看是否还是薄卡片/鳍；树冠轮廓是否完整。
2. **接缝**：相邻延伸 tile 之间有没有台阶或裂缝。
3. **类别**：天空/地面/天花板是否互相渗透（雪坡被当天空、水当地面）。
4. **边界**：走出照片扇环后，地貌是否连续、有没有断崖。

配合 `node tools/test_regression.mjs` 与 `python tools/test_regression.py` 做无浏览器回归。
