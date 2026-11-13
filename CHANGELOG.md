# 更新日志

本文件记录面向使用者的功能性变更：修了什么症状、根因是什么、影响多少、怎么验证。
按重要性排序，格式 `## [版本] - 日期`。

## [1.8.0] - 2026-11-13

基座：拉取远端 `main` 到 `93f052e`（17:31 +08）后动工。远端在本次工作前已自行修掉
`opt.ring` 崩溃并加入浏览器冒烟测试，本版在此基础上补齐训练数据层、修复 4 项 P1 算法问题，
并把「修完能不能验证」这件事做成可执行脚本。静态资源缓存串 **1.7.1 → 1.8.0**（index.html 12 处）。

### P0：训练链路与代码包

- **训练入口不再必然崩溃**。`worldmodel/train/trainer.py` 一直 `from worldmodel.data.dataset import ...`，
  但仓库里只有 `model/ train/ infer/` 三个包：跑 `python -m worldmodel.train --config configs/s1_base.yaml`
  必然 `ModuleNotFoundError`，rollout 也就永远拿不到 checkpoint。现补齐 `worldmodel/data/`：
  按 `scripts/prepare_data.py` 的 tar 分片格式读取 `init / h0..h3 / target / 7 维 action`，
  图片解码成 (3,H,W) float32 ∈ [-1,1]（与 `Tokenizer.encode` 约定一致），提供确定性 train/val 切分、
  DataLoader collate，以及缺数据时可直接复制执行的生成命令。torch 只在 collate 里延迟导入。
- **顺带修掉一个会让上面修复失效的陷阱**：`.gitignore` 里的 `data/` 是裸规则，会连
  `worldmodel/data/` 一起忽略 —— 代码写得再对也推不上仓库。已改成 `/data/`（只忽略根目录数据），
  并忽略 `__pycache__/ *.pyc`。
- `js/world.js` 的 `opt.ring`（未定义变量）由本版之前的远端提交修复为 `opts && opts.ring`；
  本版加了静态断言防止它再回来。修复前的症状很反直觉：**深度模型加载成功时反而"重建失败"、
  进度卡在 74%，只有模型失败回退旧方案才能进世界**。
- 文档一致性：`worldmodel/README.md` 原来把 `camToWorld` 写成"行主序"，与 93f052e 修掉的
  "平移恒为 0"根因（实为列主序、需 reshape 后转置）相矛盾，已更正并补上 `data/` 读取层说明；
  `pipeline/README.md` 更正了"任何尺度错误都会掉分"的错误承诺。

### P1：几何与深度链路

- **地面不再被"前拉约束"拉离 y=0 平面**（`js/world.js`）。旧实现是 3 轮四邻域 min 腐蚀，
  而地面行 `t = eye/(-sinφ)` 在靠近地平线时相邻行比值轻易超过 1.25 —— 即使深度图完全平滑
  也会触发吸附，吸附后按当前行重算 `gy`，地面浮起（实测 1,083 个顶点、最大 **1.64m**，
  是"走动尖刺"同源伪影）。现改为：地面行既不参与吸附、也不作为吸附目标，其余区域只做单轮
  单侧补洞，并在生成阶段把地面顶点显式钉在 `y=0`。
  用真实 ONNX 深度（alpine）复现：被吸附顶点 **3,828 → 990**，其中地面顶点 **1,083 → 0**，
  地面 `max|y|` **1.64m → 0.0000m**，三角数保持 15,240 不变；带人工断崖的深度仍有 1,321 个
  非地面顶点被吸附，说明补洞能力没有丢。
- **前端深度图与离线管线现在是同一套约定**（`js/depth.js` + `js/world.js` + `js/app.js`）。
  修复前前端把任意比例照片直接拉伸到 518×518、用全图 min-max 归一化、按 40 行判极性，
  而离线管线明确"绝不拉伸"、用等比 padding + 2/98 分位 + 行均值相关系数双判据。
  现把离线流程移植到浏览器侧，输出与照片同比例的深度图（1536×968 → 518×326），
  `world.js` 按宽/高分别采样，深度预览同步按宽高取样。
  真实 ONNX 对照（同一权重、alpine）：JS 后处理与离线**逐像素 MAE = 0.000000**；
  前端与离线深度 MAE **0.0977 → 0.0006**、相关系数 **0.9919 → 1.0000**；
  喂给前端 `distOf` 后距离中位差 **19.32m → 0.09m**、差 >30m 的像素 **42.2% → 0%**。
- **双锚点标定的 1500m 不再被 600m 上限吃掉**（`pipeline/geom.py` / `reconstruct.py`）。
  旧实现 `horizon_far=1500` 解出 `z = A·r^(-p)`，却在 `metric_depth` 里 `clip` 到默认 600m，
  **46.1% 像素被压成同一平面**；简单放开上限又因 `r→0` 幂律发散到 1e5m。
  现让上限自动联动（`far = 1.2 × horizon_far`，默认 **600 → 1800m**，也可 `--far` 指定），
  改用软饱和 `z = A_eff·(r + r0)^(-p)`（r0 = (A/far)^(1/p)），并按
  `A_eff = A·(1 + r0/r_near)^p` 修正幅度，**近端锚点 4.792m 精确保持不变**。
  实测触顶像素 **46.1% → 37.7%**、p90 **600m → 1800m**、全部有限值；
  新增 `--fov` 覆盖时的 EXIF 独立相机、`--verify-fov` 参数与 `p > 2` 退化告警
  （当前 p=2.284：深度 10% 误差会放大成距离 27.2% 误差）。
  `world.json` 的 `metric` 不再谎称 True（ONNX 是相对深度 + 几何标定），改为
  `false` / `depth_kind: relative-calibrated`，并写入 `far / r0 / a_effective`。
- **验收门终于能发现尺度与视场错误**（`pipeline/eval.py`）。旧自检声称"任何尺度、视场错误
  都会掉分"，实际生成与泼点用同一个相机，全局尺度在 `pos/fwd` 比值里抵消：
  坐标整体 ×0.5、FOV 错 25% 都是 **99.0dB / SSIM 1.0 纹丝不动** —— 所以之前所有尺度改进
  都测不出来。现新增两道独立参照：
  ① `ground_scale_check` 用相机眼高这个绝对量卡地面平面（应 `y ≈ -eye`）；
  ② `cam_verify` 用独立来源的相机交叉泼点，并要求交叉覆盖率不低于自洽渲染的 90%
  （常数贴图的 PSNR 可能看不出错位，相对覆盖率更可靠）。
  回归实测：×0.5 → PSNR 仍 52.9dB 但尺度门 `rel_error=0.50` 判失败；
  FOV 错 25% → 自洽 52.9dB、独立相机 **7.8dB / 覆盖 0.59** 判失败。
  端到端跑 `alpine`：自检 35.73dB / 尺度门 `rel_error=0.0522` OK；
  故意给错的 `--verify-fov 55` → 交叉门覆盖 0.76 / PSNR 18.1dB 判失败。

### 验证与工具

- `tools/test_regression.mjs`：无浏览器回归（几何地面平面、断崖补洞、等比 padding、后处理极性/分位）；
  `node tools/test_regression.mjs --depth <真实深度.f32>` 可附带真实深度图。
- `tools/test_regression.py`：无 GPU / 无 torch 回归（合成分片 → dataset、尺度门、交叉门、深度上限）。
- `tools/bump_cache.mjs`：统一升 `index.html` 的 `?v=` 缓存串，`--check` 可挂 CI，避免新旧 JS 混用。
- 本次实测：两套回归全绿；浏览器冒烟 **5/5 PASS**（alpine / desert / asset-glb / ring-on / skip-sky，
  深度模式均能进世界）；`pipeline/reconstruct.py worlds/alpine.jpg` 端到端产出 8.9MB GLB、2 层 / 305,458 三角。

### 仍然存在的限制（未在本版处理）

- 本机没有 torch，训练全链路未实跑；`worldmodel/data` 用合成 tar 分片验证读取与形状，
  真实大数据集的吞吐仍需在训练机上确认。
- 天空/远景仍有 37.7% 像素到达 1800m 上限，它们属于 `backdrop` 无视差背景层（分层设计）；
  要根治 p=2.284 的误差放大与"沙盘感"，仍需 README 已指出的度量深度后端（Depth Pro / UniK3D）。
- 前端尺度仍用经验曲线 `distOf`（近 ~2.5m / 远 ~250m），离线已用物理标定，两条链路的
  **尺度模型**尚未统一；P2 的世界模型动作/RoPE/CFG 问题、P3 的碰撞高度场与重复代码本版未动。
- 相机视场的交叉验收需要独立来源（EXIF、`--fov` 覆盖或 `--verify-fov`）；没有独立信息时
  单视图自检在数学上无法约束 FOV，文档已更正原先过度承诺的说法。
