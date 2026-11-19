# Diffusion360 落地记录（单图 → 360°）

> 结论：**跑通了**，18 秒/张，8GB 显存够用；但**画质不足以替换现有管线**，
> 出厂仍用 `--pano-mode eq`。产物：`shots/diff360_alpine.png`。

## 环境（全部在仓库内的独立 venv，不影响主环境）

| 项 | 值 |
|---|---|
| 解释器 | `uv python install 3.11` → Python **3.11.18**（主环境是 3.13，装不了它的 numpy 1.23） |
| venv | `.venvs/diff360`（已加入 `.gitignore`） |
| torch | **2.11.0+cu128**（CUDA 可用） |
| diffusers | **0.26.0**（仓库钉死；主环境是 0.43） |
| numpy / transformers / hub | 1.23.5 / 4.41.2 / 0.23.5（hub 必须 <0.26，否则 diffusers 0.26 导入就炸） |
| 源码 | `third_party/diff360`（已加入 `.gitignore`） |
| 权重 | `third_party/diff360/models/`：`sd-base` + `sd-i2p` + `sr-base`（约 **7GB**，hf-mirror 下载，峰值 22MB/s） |

## 为了让它在 8GB / Windows 上跑，改了三处

1. **SR（超分）改成按需加载**：`image_to_360panorama_image_pipeline.py` 顶部原本
   `from basicsr... / from realesrgan...`，而 basicsr 1.4.2 在 torchvision 0.26 上
   导入即崩（`functional_tensor` 已被删）。改成 `enable_sr=False` 时整段跳过，
   代价是输出只有 512×1024（不上采样）。
2. **关掉 CPU offload**：`enable_model_cpu_offload()` + 手动 `encode_prompt` 会把
   token 留在 CPU，报 `Expected all tensors to be on the same device`。
   8GB 上 SD2.1 + i2p ControlNet 整块放得下，直接 `.to('cuda')` 更稳。
3. `torch` 必须锁 **2.11**：装 `realesrgan` 会顺手把 torch 升到 2.15 + torchvision 0.29，
   反而更装不上 basicsr，且破坏 CUDA 匹配。

## 实测数据

```
加载  9s（模型进显存）
推理  18s（24 步，512×512 输入 → 1024×512 全景）
显存  峰值约 5.5GB / 8GB
```

## 画质对比（这就是不换默认的原因）

| 维度 | 现有 `--pano-mode eq`（SDXL） | Diffusion360（SD2.1） |
|---|---|---|
| 照片区保真 | ✅ 与输入逐像素一致（投影纹理） | ❌ **中心区就被重画了**（Half Dome 变成另一座山） |
| 侧向结构 | ⚠️ 糊、缺细节 | ✅ 连贯（山、湖、岸线连续，**没有竖条**） |
| 接缝 | 无（单张生成） | ⚠️ 有 3~4 条竖直亮缝 |
| 天空 | 干净渐变 | ❌ 一条巨大的极光状亮弧（SD2 的太阳） |
| 天底 | 雾色 | ❌ 一圈"木地板"（输入被拉伸造成的） |
| 风格 | 照片 | 绘画感明显 |

## 如果以后要真正用它

优先级从高到低：

1. **把 mask 与输入几何对齐**：现在用的是官方示例 mask（90° 方形窗口 0.25 权重），
   我们的照片是 60°×40°，内容被重画就是这个原因；应该按 60/360×1024 与 40/180×512
   重新算窗口，并把输入**先投影到等距柱状**再喂；
2. 打开 SR（`upscale=True`，需要 sr-control + RealESRGAN，约 +2GB 显存 + 4GB 权重）；
3. 接进烘焙：`--pano-mode diff360` 走子进程调用 `.venvs/diff360/Scripts/python.exe`。

预计再 1~2 小时可以做到"中心保住 + 侧向连贯"，但**风格仍是 SD2 的绘画感**，
达不到"像原照片"。
