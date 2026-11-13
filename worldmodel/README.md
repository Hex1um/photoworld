# 阶段 1：自回归帧预测世界模型（原型）

一张初始图 + 一串相机动作 → rollout 出一段连贯、动作可控的视频。
只验证范式，不含蒸馏 / 量化 / 流式服务 / refiner（阶段 2~4）。

## 一、数据

### 合成轨迹（本仓库自带产线）

```bash
node scripts/render_traj.mjs --glb assets/world-alpine.glb --out data/syn --trajs 120 --frames 48
```

从 `pipeline/` 重建出的 glb 世界渲染轨迹，**位姿是真值**，所以动作标签没有估计误差。

产物：

```
data/syn/traj_0000/
  frames/000000.jpg … 000047.jpg      384×256
  poses.json                          {fovY, fovX, width, height, frames:[{file, camToWorld[16], action}]}
```

`camToWorld` 是 4×4 **列主序**（相机→世界，平移在最后 4 个元素，与 WebGL 一致）。
`scripts/prepare_data.py::mat4_from_rowmajor` 必须先 `reshape(4,4)` 再转置；
早期版本漏了转置，`T[:3,3]` 取到齐次列 `[0,0,0]`，**所有动作的平移分量恒为 0**，
模型从未收到"前进"信号（已在 93f052e 修复）。

### 训练分片

```bash
python scripts/prepare_data.py --traj data/syn --out data/wm --shard 500
```

每个样本包含 `init.jpg` / `h0..h3.jpg` / `target.jpg` / `.json`，动作定义：

```
a_t = [axis-angle(3), Δt(3, 相机系, 除以轨迹中位速度), log(FOV 比值)(1)]   # 7 维
```

`Δt` 归一化后不同尺度的场景可比。窗口 `(frame_{t-4..t}, a_t, frame_{t+1})`，N=4。

分片由 `worldmodel/data/dataset.py::TarWindows` 读取（补齐了此前缺失、导致训练入口
`ModuleNotFoundError` 的包）：图片解码成 (3,H,W) float32 ∈ [-1,1]，按固定 seed 切 train/val，
`collate` 堆叠成 batched tensor。可用 `python tools/test_regression.py` 无 torch 验证读取与形状。

### 真实数据（可选，未接入）

RealEstate10K / DL3DV 抽帧后需用 VGGT/COLMAP 估位姿，写成同一个 `poses.json` schema 即可混训，
在 `.json` 的 `meta.source` 标 `real`，评测可分层统计。

## 二、Tokenizer

`worldmodel/model/tokenizer.py`：SD-VAE（`stabilityai/sd-vae-ft-mse`，83.7M，**冻结**）。
256×384 → 32×48×4，scaling factor 0.18215 写死在代码里。支持分块解码。

权重走 HF 镜像下载一次即可，之后 `local_files_only=True` 只读本地缓存：

```bash
HF_ENDPOINT=https://hf-mirror.com python -c "from worldmodel.model.tokenizer import Tokenizer; Tokenizer()"
```

## 三、模型

`worldmodel/model/dit.py`：**Causal World DiT，129.3M**

| 项 | 取值 |
|---|---|
| 潜变量 | 2×2 patchify → 每帧 16×24=384 token |
| 序列 | 1(初始) + 4(历史) + 1(目标) = 6 帧 = 2304 token |
| Transformer | hidden 768 / 12 层 / 12 头，**SDPA** |
| 位置编码 | RoPE 三轴 (t, y, x) |
| 注意力 | 时间维因果、帧内双向（bool 掩码） |
| 条件 | adaLN 注入 时间步 + 动作 + 域标签；初始帧与历史帧作为前缀 token |
| 目标 | rectified flow，v-prediction，单步 `x ← x + dt·v(x,t)` 从 t=0 积分到 1 |
| CFG | 训练 drop 动作 10%、drop 初始帧 10% |

## 四、训练

```bash
HF_HUB_OFFLINE=1 python -m worldmodel.train --config configs/s1_3000.yaml
```

所有超参在 `configs/*.yaml`。实测（RTX 5070 Laptop 8G，batch 2）：

```
data 0.11s  vae 0.09s  fwd 0.26s  bwd 0.26s   ≈ 0.8 s/step
```

**8G 显存下的取舍**：batch 2（前向峰值 4.56GB），有效 batch 靠 `grad_accum` 堆。
4090 24G 可以 batch 8~12，吞吐约 3~4 倍，5 万窗口 3000 步约 3 小时。

## 五、Rollout

```bash
python scripts/rollout.py --image data/syn/traj_0000/frames/000000.jpg \
  --actions "forward 20,turn_left 15,forward 10" --ckpt runs/s1_3000/last.pt --out out/roll
```

动作 DSL：`forward / back / turn_left / turn_right / look_up / look_down / idle` + 帧数。
Euler 8 步 + CFG 4.0。产出逐帧 JPEG + `rollout.gif`。

## 六、评测

```bash
python scripts/eval.py --ckpt runs/s1_3000/last.pt --out eval --n 8 --dir-steps 6
```

出 `eval/report.md`、`eval/metrics.json`、`eval/frames/`、`eval/long_rollout.gif`：

1. **一步预测**：PSNR / SSIM，与「复制上一帧」基线对比（验收要求 ≥ +3dB）
2. **动作可控性**：`turn_*` 用相位相关估整体位移符号；`forward` 用中心/边缘差能量判"放大"
3. **长 rollout**：40 帧帧间 SSIM 距离曲线

## 七、踩过的坑（都写进代码注释了）

| 现象 | 真相 | 修法 |
|---|---|---|
| "VAE 卡 70 秒" | 数据加载 `cache=True` 缓存了 36GB 解码张量 → 系统内存耗尽 → 疯狂换页。表象像 GPU 卡死 | `cache=False` |
| rollout 全是噪声 | EMA 固定 `decay=0.999`：300 步后影子权重 74% 仍是随机初值 | warmup `min(0.999,(1+t)/(10+t))` |
| bf16 下 device-side assert | `masked_fill(-inf)` 经 softmax 出 NaN | 改 bool 掩码（顺带让 SDPA 走高效路径） |
| 注意力吃掉 12.5GB | 手写 matmul+softmax | 换 `scaled_dot_product_attention`，B=2 降到 4.56GB |
| 数据加载每样本重开 tar | `extractfile(名字)` 是线性扫描 | TarInfo 索引，O(1) |
| 训练前 100 步极慢 | HF 镜像 429 限流，每次联网校验 VAE 白等 100s+ | `local_files_only=True` |

## 八、已知局限

- **无长期记忆**：感受野只覆盖初始帧 + 最近 4 帧，走出视锥后信息靠"猜"→ 阶段 2 加 memory bank
- **loop closure 不保证**：转一圈回来对不上 → 阶段 3 引入几何约束
- **长 rollout 漂移**：误差累积，40 帧后可见色偏与结构软化 → 阶段 4 refiner
- 只在**单一重建世界**的合成数据上训练，真实照片泛化未验证
- 256×384 分辨率仅够看结构，细节不足
