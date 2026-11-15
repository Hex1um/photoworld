# pipeline/v2 —— 分层重建（LDI）

v1（`pipeline/reconstruct.py`）把一张照片做成**一层连续深度薄片**，于是山体在
深度不连续处被拉成"幕布"、近景塌成"地毯"。v2 把深度和几何都换掉：

| 模块 | 作用 | 关键点 |
|---|---|---|
| `depth_gpu.py` | GPU 高分辨率深度 | Depth-Anything-V2-**Large**，长边 1540，fp16，**guided-filter 上采样**（以原图为引导） |
| `calib.py` | 尺度标定 | 地面带拟合不可信时退回分位双锚点；俯仰落在搜索边界即拒收 |
| `ldi.py` | 分层几何 | 断崖处不连面（不拉满）、边界挤裙边（有厚度）、地面按几何判据压平、掠射面拉伸抑制 |
| `api.py` | 可调用接口 | 常驻模型 + 内容哈希缓存，给 `server/app.py` 的 `/api/reconstruct` 用 |
| `bake.py` | 命令行批处理 | 一张/多张照片 → `out/<name>/world.glb` + `world.json` |

## 依赖与权重

```bash
python -m pip install torch transformers scipy pillow numpy
```

权重（约 1.3GB，fp32 safetensors）：

```bash
# HF 直连在国内常被墙，用镜像（仓库其它地方同样用法）
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p models/da2_large
for f in config.json preprocessor_config.json; do
  curl -sL -o models/da2_large/$f \
    "https://hf-mirror.com/depth-anything/Depth-Anything-V2-Large-hf/resolve/main/$f"
done
curl -L -C - --retry 20 --retry-all-errors \
  -o models/da2_large/model.safetensors \
  "https://hf-mirror.com/depth-anything/Depth-Anything-V2-Large-hf/resolve/main/model.safetensors"
```

**不要把权重提交进仓库**（`.gitignore` 已忽略 `models/da2_large/`）。

## 用法

```bash
# 单张
python pipeline/v2/bake.py corpus_test/03_snowrange.jpg --out out/v2/snow

# 批量（每张一个子目录）
python pipeline/v2/bake.py "corpus_test/*.jpg" --out out/v2/all

# 调试：每层单独导一个 GLB，用来定位缺陷在哪一层
python pipeline/v2/bake.py photo.jpg --out out/x --debug-layers
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--far-split` | 120 | 近景/无视差背景的分界（米）。切在雾霭里，缝看不见 |
| `--edge-ratio` | 2.4 | 多大深度比视为断崖（越小剪得越狠，碎片越多） |
| `--stretch-clamp` | 12 | 纹理拉伸超该倍数就用大窗口中值平滑深度（0=关） |
| `--strong-median` | 7 | 上述平滑的窗口 |
| `--quantize` | 0 | 深度量化倍数（枝叶类可试 1.3，实测帮助有限） |
| `--no-flatten` / `--no-skirt` | — | A/B 用开关 |

## 视锥外扩（outpaint）

`--outpaint 1.7` 会把照片放进更大画布（左右 +35%、下 +22%、上 +8%）并把四周补出来，
中心原图区域**严格保留原始像素**；视野按 `hfov_new = 2·atan(tan(hfov/2)·W_new/W_old)`
换算，中心角分辨率不变。

两种模式：

| 模式 | 触发 | 效果 | 耗时 |
|---|---|---|---|
| 镜像延展 + 渐隐雾色 | 默认（`outpaint_model=False`） | 无空白、无刀切边，但没有新内容 | <0.2s |
| 扩散模型补全 | `outpaint_model=True` 且权重可用 | 四周真的长出新地貌 | 取决于模型 |

**权重注意**：不加门禁的可用权重是 `Lykon/dreamshaper-8-inpainting`（SD1.5 系，约 2GB）。
`diffusers/stable-diffusion-xl-1.0-inpainting-0.1` 也能下（6.5GB），但在 8GB 显存的笔记本上
单张 15 分钟都跑不完 —— 所以默认关闭模型模式，别默认去踩这个坑。

## 已知失败模式

**密集枝叶/细枝前景**（`corpus_test/08_autumn_fg.jpg`）仍然不行：深度本身是对的
（叶 1m / 背景树 30m），但几像素内跨 30 倍的深度在网格里必然把四边形剪成碎片。
正确表示是点云/splat —— 需要给 GLB 加载器加 `mode=0 (POINTS)` 支持（v2.1）。

## 回归

```bash
python tools/test_v2.py              # 26 条断言，2 秒，不需要 GPU
python tools/test_v2.py --with-model # 额外真跑一次推理（需 CUDA + 权重）
```
