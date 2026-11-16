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

## 运行

```bash
# 1) 静态站（score_params 要用它渲染）
cd photoworld && python -m http.server 8199 --bind 127.0.0.1 &

# 2) 训练（DeepSeek 走 OpenAI 兼容协议）
cd ../tools_skillopt
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1 \
OPENAI_COMPATIBLE_API_KEY=$DEEPSEEK_API_KEY \
OPENAI_COMPATIBLE_MODEL=deepseek-chat \
python scripts/train.py --config configs/photoworld/default.yaml
```

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
