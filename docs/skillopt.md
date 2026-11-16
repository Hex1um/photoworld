# 用 SkillOpt 训练"烘焙参数选择"这项技能

[SkillOpt](https://github.com/microsoft/SkillOpt)（Microsoft，NeurIPS 2026 Spotlight）把
**skill 文档当可训练参数**：评分 rollout → 优化器模型提议文本编辑 → 验证门控接受。
它不改模型权重，产出是一个 `best_skill.md`。

我们用它的原因：PHOTOWORLD 有一堆互相牵制的旋钮（`far_split` / `ground_radius` /
`seg_edge_tau` / `skirt_cap` …），今天的调参经验全散在 CHANGELOG 和代码注释里，
**没有可复现的"该按什么规则选"**。SkillOpt 正好能把这件事变成有门控的实验。

## 结构

```
photoworld/                         # 本仓库（权威副本在这里）
  skillopt/
    envs/photoworld/
      adapter.py                    # EnvAdapter：rollout = LLM 出参→烘焙+渲染+评分
      dataloader.py                 # 读 data/photoworld/{train,val,test}/items.json
      seed_skill.md                 # 种子技能（含 params yaml 块）
    configs/photoworld.yaml         # 训练配置（llm_target: true, views: 8）
  tools/score_params.py             # 单图评分器（烘焙 → 8 机位 → 指标 → hard/soft）
  tools/sync_skillopt.py            # 把上面的 skillopt/ 推到同级 clone
  data/photoworld/{train,val,test}/ # 数据划分（本仓库未入库，见 docs/ISSUES.md）
../tools_skillopt/                  # SkillOpt 独立 clone（训练实际从这里跑）
  scripts/train.py                  # 由 sync_skillopt.py 同步 env/config 后运行
```

关键设计：**target 就是 LLM**（`llm_target: true`，见 configs/photoworld.yaml）。
每轮由 target 读技能准则、输出一组烘焙参数 → 确定性流水线烘焙 → 渲染 → 打分；
不达标时把实测指标喂回 target 再要一组（`llm_rounds: 2`）。
优化器模型负责改 skill 文本。评分的可信度来自我们自己的指标（见下方"判定口径"）。

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
python scripts/train.py --config configs/photoworld.yaml
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

以 `tools/score_params.py` 的 `THRESH` / `WEIGHT` 为准（改阈值要同步 `tools/acc_report.py`）：

| 指标 | 阈值 | soft 权重 |
|---|---|---|
| `stripe` | ≤ 2.0 | 0.24 |
| `innerVoid` | ≤ 0.06 | 0.12 |
| `voidFrac` | ≤ 0.10 | 0.12 |
| `lowVoid` | ≤ 0.08 | 0.12 |
| `voidBlobs` | ≤ 260 | 0.12 |
| `streakBlobs` | ≤ 200 | 0.16 |
| `voidRunMax` | ≤ 0.50 | 0.08 |
| `voidBlobMax` | ≤ 0.22 | 0.04 |

八项全过 → `hard=1`。`soft` 按超阈值程度扣减，权重合计 1.0。

**聚合口径**：8 项判据一律取所有机位里的**最差值**（`views: 8` 环拍），
单个机位的贯穿缝/大洞不会被另外 7 个好视角摊薄；`detail`/`darkFrac` 等
只用于展示的指标仍取均值。指标缺项一律判无效（hard=0/soft=0），不给满分。
注意此口径自 v2.15.4 起生效，跨版 soft 分数不可直接比较。

## 成本

`score_params.py` 默认**关掉 SD 外扩**（`outpaint=1.0`）：外扩一张要 40~150 秒，
rollout 会被拖到不可用。关掉后单图约 60~80 秒（含深度模型加载）。

评分器已把 `api.py` 做成进程内常驻单例（v2.15.2 起）：深度模型只加载一次，
重复参数的烘焙可直接命中内容哈希缓存。此前每次评分都重建模块、重载约 1.3GB 权重。
