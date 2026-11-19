# 全景外推模型：本机可行性评估（2026-11-19）

## 结论先说

| 候选 | 推理显存 | 能否在本机跑（RTX 5070 Laptop **8GB**） | 结论 |
|---|---|---|---|
| **PanFusion** | **26.3 GB** | ❌ 差 3 倍以上 | 排除 |
| UniPano（ICCV'25） | 32.6 GB（训练）/ 更大 | ❌ | 排除 |
| PanoDiff | 36.0 GB | ❌ | 排除 |
| MVDiffusion 系 | 24 GB+ | ❌ | 排除 |
| DreamCube / SpND | 16.7 GB | ❌ | 排除 |
| SMGD | 8.0 GB | ⚠️ 临界，且是 512 级质量 | 不划算 |
| **Diffusion360** | **3.5 GB / 4s** | ✅ **唯一可行** | 值得一试 |

数据来源：《A Survey on Text-Driven 360-Degree Panorama Generation》(arXiv 2502.14799) 的方法对比表
（FID / 推理时间 / GPU 显存），以及 PanFusion 官方文档（训练 60.12 GB 峰值、推理 26.3 GB）。

## 本机实际情况（实测）

```
GPU      NVIDIA GeForce RTX 5070 Laptop，8 GB 显存，功耗上限 93W
算力     fp16 8192³ matmul = 31.4 TFLOP/s（电源方案已是「高性能」）
Python   3.13.14
torch    2.11.0+cu128（CUDA 12.8）
diffusers 0.43.0
numpy    2.5.4
```

## Diffusion360 的依赖可行性（`pip install --dry-run` 实测）

| 依赖 | 结果 |
|---|---|
| `xformers` | ✅ 有 0.0.36 |
| `triton-windows` | ✅ 有 3.9.0.post30（Windows 移植版，非官方 triton） |
| `py360convert` | ✅ 有 1.0.4 |
| `realesrgan==0.3.0` | ✅ 可装 |
| **`diffusers==0.26.0`** | ⚠️ 与我们现在的 0.43 冲突，需隔离环境 |
| **`numpy==1.23.5`** | ❌ **Python 3.13 没有 1.23 的 wheel**，会走源码编译并大概率失败 |

**这是唯一的硬阻塞点**：仓库是 2023 年的代码，钉死了 numpy 1.23 / diffusers 0.26。

## 两条可选路线

### A. 独立 Python 3.11 环境（推荐）
1. 用 python-build-standalone 下一个 3.11（约 30MB，走镜像）；
2. `venv` 里装 torch-cu128 + diffusers 0.26 + numpy 1.23 + xformers/triton-windows + py360convert；
3. 下载权重 `archerfmy0831/sd-t2i-360panoimage`（约 4GB，走 hf-mirror，实测 4~7 MB/s）；
4. 跑 `img2panoimg` 的「单图 → 360」管线，把我们的 alpine 照片喂进去；
5. 如果效果好，在我们的烘焙里加一个 `--pano-mode diff360`：调子进程（独立解释器）产全景，
   再走现有的地形/着色流程。

**工作量**：装环境 30~40 min，权重 15 min，接管线 40 min，调 1~2 轮 30 min → **约 2 小时**。
**风险**：triton-windows 与 torch 2.11 的匹配、realesrgan 的旧依赖链；最坏情况卡在安装。

### B. 不用（维持现状）
出厂用 `--pano-mode eq`（SDXL inpaint + 镜像结构初值）：
出生点与全部空中视角是照片级，转身/低头是干净的雾霭，但没有细节。

## 我的判断

- 想要"转身 180° 也像照片"，**只能走 A**，而且 A 里 Diffusion360 是唯一能在 8GB 上跑的选择。
- 预期收益要放平：它是 **SD2 时代**的 512×1024 模型 + Real-ESRGAN 放大，天花板低于我们现在的
  SDXL。它的优势在**结构**（环形一致性、接缝处理），不在**细节**。
- 所以 A 值得做，但如果目标是"看起来像原照片"，那仍然做不到——那是多图/视频重建
  （3D Gaussian Splatting 一类）的领域，不是单图扩散能补的。

## 环境获取路径（实测）

| 要素 | 现状 |
|---|---|
| 本机 Python | 只有 WorkBuddy 托管的 **3.13.12**（PATH 上没有 python/py） |
| uv / conda / py | 都没有 |
| 3.11 运行时 | 需要装：`pip install uv` → `uv python install 3.11`（uv 从 GitHub 拉预编译包；慢的话设 `UV_PYTHON_INSTALL_MIRROR` 走镜像）|
| Diffusion360 权重 | `hf-mirror.com/api/models/archerfmy0831/sd-t2i-360panoimage` **可达** ✅（约 4GB）|

所以路线 A 的第一步（搞一个 3.11 解释器）是可行的，额外约 10 分钟。
