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

**训练与推理共用 `worldmodel/action_codec.py`**。`scripts/prepare_data.py` 和
`worldmodel/infer/rollout.py` 都调用它的 `rel_action` / `dsl_action`，后者把 DSL
名称转成与训练标签同构的 7 维（构造与 `render_traj.mjs::makePoses` 相同的相邻位姿）。
1.9.0 之前推理侧只手填 `a[1]/a[3]/a[5]` 三个标量：`forward` 被编码成"绕 y 轴转 + x 平移"，
`turn_left` 的轴角符号还是反的，模型学到的动作语义在推理时被整体换掉。
`python tools/test_regression_p2.py` 会复现合成轨迹标签并比对，最大偏差需 < 0.02。

分片由 `worldmodel/data/dataset.py::TarWindows` 读取（补齐了此前缺失、导致训练入口
`ModuleNotFoundError` 的包）：图片解码成 (3,H,W) float32 ∈ [-1,1]，按固定 seed 切 train/val，
`collate` 堆叠成 batched tensor。也可用 `.latent` 预编码分片直读（见第四节）。
可用 `python tools/test_regression.py` 无 torch 验证读取与形状。

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
| 序列 | 1(初始) + 4(历史) + 1(目标) = 6 帧 = 2304 token，另有 1 个动作 token（共 2305） |
| Transformer | hidden 768 / 12 层 / 12 头，**SDPA** |
| 位置编码 | RoPE 三轴 (t, y, x)，`worldmodel/model/rope.py` 保证 half-split 前后半频率配对 |
| 注意力 | 时间维因果、帧内双向（bool 掩码） |
| 条件 | adaLN 注入 时间步 + 域标签；动作为独立 token；初始帧与历史帧作为前缀 |
| 初始化 | block 的 adaLN gate 零初始化（残差恒等），输出头常规小随机初始化 |
| 目标 | rectified flow，v-prediction，Euler 8 步 `x ← x + dt·v(x,t)` 从 t=0 积分到 1 |
| CFG | 训练/推理共用 `uncond=True`：一次联合 drop 动作 + 初始帧 + 域标签（`drop_cond: 0.1`） |

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

### 可选：离线预编码 latent

默认每个 micro-batch 在线编码 `init + 4×hist + target = 6` 张图；按
`s1_base.yaml`（batch 2 / grad_accum 16）是 **每 optimizer step 192 次 VAE encode**，
2000 步约 38.4 万次。可在有 CUDA 的训练机上先跑一次：

```bash
python scripts/preencode_latents.py --shards data/wm --out data/wm_latent
# 把配置里的 data.shard_dir 改成 data/wm_latent
```

产物每个样本只有 `{key}.latent`（float32，`(6,4,32,48)`）+ `{key}.json`，约 147KB，
比 6 张 JPEG 更小。`TarWindows` 检测到 `.latent` 自动走 `numpy.frombuffer` 直读，
trainer 检测到 batch 的 `z_init` 就跳过 VAE，这个瓶颈归零；原分片不受影响，
验证循环也走同一条读取路径。该脚本需要 torch + diffusers，**本机无 torch 未实跑**，
通路本身由 `tools/test_regression_p2.py` 用合成 latent 分片验证。

验证 loss 现在临时换上 EMA 权重再评估（评估完恢复训练权重），与默认加载 EMA 的
rollout 同源；日志里的 `val loss ... (EMA)` 即此路径。

### 三项「省时不改算法」的开关（v1.17，全部 opt-in，默认行为不变）

| 配置键 | 默认 | 作用 | 代价 |
|---|---|---|---|
| `data.prefer_latent` | `false` | 自动切到 `{shard_dir}_latent`（缺失则回退并打印），训练循环内 VAE encode 归零 | 需离线预编码一次 |
| `train.ss_mode: noise_proxy` | `forward` | scheduled sampling 不再多跑一次 DiT 前向：`z_self = (1-α)z1 + αε`，α~U(0.05,0.25) | 无（`s3.yaml` p=0.5 时省掉 +16.7% MACs） |
| `train.val_max_samples` | `0`（全量） | 固定 seed 抽固定子集验证，`val_every` 到点时不再遍历全量 val | 指标方差更小、开销降一个量级 |
| `train.ema_every` | `1` | 每 N 步更新一次 EMA，衰减用 `decay**N` 补偿 | 无（显存访存减少） |

`configs/s1_fast.yaml` 是 `s1_base` + 上述四项的现成组合（权重同构，checkpoint 可互读）。
回归：`python tools/test_trainer_opts.py`（分片自动切换 / latent 直读 / 固定 val 子集 /
EMA 降频 / noise_proxy 不产生额外前向）。

**没有采用**：`flex_attention` 块稀疏（掩码掉的 41.7% 注意力对可跳过）需要训练机实测
数值等价后再开，且会替换 SDPA 的 math path，收益 13.9% MACs 但风险高于收益，
留作下一版在真实分片上做 A/B。

## 五、Rollout

```bash
python scripts/rollout.py --image data/syn/traj_0000/frames/000000.jpg \
  --actions "forward 20,turn_left 15,forward 10" --ckpt runs/s1_3000/last.pt --out out/roll
```

动作 DSL：`forward / back / turn_left / turn_right / look_up / look_down / idle` + 帧数。
每个 DSL 名称经 `action_codec.dsl_action` 展开成与训练标签同构的 7 维动作；
幅度和速度归一化可用 `parse_actions(..., speed=, move_step=, yaw_step=, pitch_step=)` 覆盖，
以匹配具体数据集。Euler 8 步（**中点** `t=(s+0.5)dt`）+ CFG 4.0，
无条件分支走 `uncond=True`。产出逐帧 JPEG + `rollout.gif`。

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
| 动作完全不可控 | 训练标签 7 维 6DoF，推理 DSL 只填 `a[1]/a[3]/a[5]` 三个标量，维度与符号都错 | 共用 `action_codec.dsl_action` |
| 位置编码等变性被破坏 | 三轴频率拼成 60 维再补零，与 half-split 旋转的前后半不配对（单测误差 0.984） | 频率表拆到 d/2 通道后复制，误差降到 8.9e-16 |
| 开训最初若干步 block 不学 | 输出头零初始化使 `dL/dx = W_headᵀ·grad = 0` | gate 零初始化 + head 常规初始化 |
| CFG 引导方向不对 | 训练 drop 动作/初始帧/域，推理只把 action 置零（`act_embed` 还有 bias） | 统一 `uncond` 路径 |

## 八、已知局限

- **无长期记忆**：感受野只覆盖初始帧 + 最近 4 帧，走出视锥后信息靠"猜"→ 阶段 2 加 memory bank
- **loop closure 不保证**：转一圈回来对不上 → 阶段 3 引入几何约束
- **长 rollout 漂移**：误差累积，40 帧后可见色偏与结构软化 → 阶段 4 refiner
- 只在**单一重建世界**的合成数据上训练，真实照片泛化未验证
- 256×384 分辨率仅够看结构，细节不足
