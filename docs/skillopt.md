# 用 SkillOpt 训练"烘焙参数选择"这项技能

[SkillOpt](https://github.com/microsoft/SkillOpt)（Microsoft，NeurIPS 2026 Spotlight）把
**skill 文档当可训练参数**：评分 rollout → 优化器模型提议文本编辑 → 验证门控接受。
它不改模型权重，产出是一个 `best_skill.md`。

我们用它的原因：PHOTOWORLD 有一堆互相牵制的旋钮（`far_split` / `ground_radius` /
`seg_edge_tau` / `skirt_cap` …），今天的调参经验全散在 CHANGELOG 和代码注释里，
**没有可复现的"该按什么规则选"**。SkillOpt 正好能把这件事变成有门控的实验。

## 结构

```
tools_skillopt/                     # SkillOpt 本体（clone 在 photoworld 的同级目录）
  skillopt/envs/photoworld/
    adapter.py                      # EnvAdapter：rollout = 烘焙+渲染+评分
    dataloader.py                   # 读 data/photoworld/{train,val,test}/items.json
    seed_skill.md                   # 种子技能（含 params yaml 块）
  configs/photoworld/default.yaml   # 训练配置（deepseek + openai_compatible 后端）
photoworld/
  tools/score_params.py             # 单图快速评分器（烘焙 → 4 机位 → 指标 → hard/soft）
  data/photoworld/{train,val,test}/ # 4 / 1 / 1 张的划分
```

关键设计：**target 不是 LLM**。这里的"执行者"是一条确定性流水线，
rollout 只从 skill 文档里读 yaml 参数 → 烘焙 → 渲染 → 打分。
优化器模型只负责改 skill 文本，所以评分的可信度完全来自我们自己的指标。

## 运行：双击「PHOTOWORLD 训练」

桌面上的 `PHOTOWORLD 训练.bat` 会依次做四件事（**不需要**手工开终端）：

1. 起静态站 `127.0.0.1:8199`（评分器渲染要用）；
2. 起评分服务 `127.0.0.1:8300 --warm`（深度模型只加载一次）；
3. 轮询 `/health`，等 `loaded=true` 才算就绪；
4. 前台跑训练，进度直接打在窗口里，同时写 `logs/train.log`（UTF-8）。

收尾用 `PHOTOWORLD 训练 -停止.bat`：按 `logs/pids_train.json` 里的 PID 精确关闭，
不误杀别的 python。

只想起服务不跑训练：`PHOTOWORLD 训练.bat -SkipTrain`。

### 命令行等价写法

```bash
# 终端 1：静态站
python -m http.server 8199 --bind 127.0.0.1
# 终端 2：评分服务（常驻！训练脚本自己起的子进程会被回收）
python tools/score_server.py --port 8300 --warm
# 终端 3：训练
cd ../tools_skillopt
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1 \
OPENAI_COMPATIBLE_API_KEY=$DEEPSEEK_API_KEY OPENAI_COMPATIBLE_MODEL=deepseek-chat \
PW_SCORE_SERVER=http://127.0.0.1:8300 \
python scripts/train.py --config configs/photoworld/default.yaml
```

### 训练循环现在真的转起来了

三个关键修复（都不是"调参"，是"环境/契约"层面的）：

1. **轨迹要写成对话**（`adapter._write_trajectory`）：
   分析器读的是 `predictions/<id>/conversation.json`，内容是"做了什么 + 环境反馈"。
   早先只把指标数字放结果字典里 → 每次都是 0 条编辑。
   现在每条轨迹带：用了哪些参数、每项指标 vs 阈值（标 ok/FAIL）、判定结果，
   结果字典里再补 `task_description` / `fail_reason`（如"lake：lowVoid 0.1616 未达标"）。
2. **评分在训练进程内跑**（`inproc: true`）：不依赖任何常驻服务。
   评分服务（HTTP）在本机反复被回收，症状是 `Remote end closed` / `WinError 10061`，
   整轮 rollout 变 0 分。进程内跑还省掉每张图一次的模型加载。
3. **验证集必须"有失败可分"**：原先 val 用了本来就满分的图（hard=1），
   任何编辑都不可能"严格提升"→ 门控必然全拒（实测 `action=reject`）。
   换成有超阈值的图后，STEP 1 就产出了 3 条编辑、合并成 2 条、技能从 1422 增到 1994 字符。

观察到的实际数据（一轮 rollout）：

| 图 | soft | 失败项 |
|---|---|---|
| snow | 1.000 | — |
| coast | 0.910 | `streakBlobs` 268 |
| lake | 0.803 | `lowVoid` 0.162、`voidRunMax` 0.446、`voidBlobMax` 0.478 |
| desert | 0.975 | `voidRunMax` 0.605 |

单图评分可以单独跑，便于手工试参数：

```bash
cd photoworld
python tools/score_params.py --image worlds/desert.jpg --base http://127.0.0.1:8199 \
  --params '{"far_split": 90, "ground_radius": 300}'
```

## 判定口径

与 `docs/acc32b.md` 完全一致（阈值改了要两边同步）：

| 指标 | 阈值 | soft 权重 |
|---|---|---|
| `stripe` | ≤ 2.0 | 0.40 |
| `innerVoid` | ≤ 0.06 | 0.20 |
| `voidFrac` | ≤ 0.10 | 0.20 |
| `lowVoid` | ≤ 0.08 | 0.20 |

四项全过 → `hard=1`。`soft` 按超阈值程度扣减。

## 成本

`score_params.py` 默认**关掉 SD 外扩**（`outpaint=1.0`）：外扩一张要 40~150 秒，
rollout 会被拖到不可用。关掉后单图约 60~80 秒（含深度模型加载）。

要更快的循环，可以把评分器改成常驻进程（模型只加载一次），这样单图能降到 10~20 秒。
这是下一步的优化点。
