# 更新日志

本文件记录面向使用者的功能性变更：修了什么症状、根因是什么、影响多少、怎么验证。
按重要性排序，格式 `## [版本] - 日期`。

## [1.13.0] - 2026-11-14

合并 v1.12（未单独发布）的工程质量项，并把**默认重建路径从「视锥/球面参数化」
整体换成「直角坐标反投影」**。静态资源缓存串 **1.11.0 → 1.13.0**（index.html 12 处）。

### 为什么必须换掉视锥模型

v1.11 的几何是「屏幕行=仰角、深度=半径」的球面网格，再乘一个 `gVis` 视锥掩码；
视锥内才建面、边缘用 `aEdge` 融色盖住。这套结构有三个不可修复的副作用：

- 视锥边界是**人为裁切**，任何放宽都会露出镜像拉伸条纹（v1.11 已实测 38% 时一片灰弧）；
- 边缘顶点沿射线排布，深度断崖处只能靠「前拉吸附」硬拉，结果就是**尖刺/鳍**；
- 地平线以下若强制 `y=0`，半张照片就是**平铺贴图**；若交给深度，又和球面半径打架。

结论：视锥参数化本身会产生尖刺与平铺，融色只是把症状盖住。

### 读到的方案（github.com/google/three-live、topics/image-to-3d、topics/scene-generation）

| 来源 | 可迁移的结论 |
| --- | --- |
| `google/three-live`（`compileFromImage` / `from-image`） | **REFERENCE \| RENDER \| OVERLAY** 视觉闭环、`--visual-turns` 自纠错、spatial lint（有限边界/网格数/覆盖/重叠）。→ 落到 `tools/quality_gate.py` 与回归断言。 |
| `Tencent-Hunyuan/HunyuanWorld-Voyager` | 单图 → **RGB-D 序列 → 世界坐标点云**，world cache + 点剔除保证全局一致。→ 反投影点云是正解，不是球面网格。 |
| `ZiYang-xie/WorldGen`、`SkyworkAI/Matrix-3D` | RGBD → mesh / 3DGS，360° 可探索，**先粗几何 scaffold 再补细节**。→ 有限区域 + 地面低通 = 轻量 scaffold。 |
| `WHU-USI3DV/VistaDream` | 单视图先建 coarse 3D scaffold，再做 RGB-D inpainting 补洞。→ 断崖跳面 + 背景层实验（默认关）。 |
| `VAST-AI-Research/MIDI-3D`、`mschneider456/WorldMesh` | 多实例/网格条件生成，**遮挡关系要显式建模**。→ 跳面避免把前景背景连成一张面。 |
| `facebookresearch/map-anything`、`HunyuanWorld-Mirror` | 度量 3D / 法线与深度联合输出。→ 法线 3×3 平滑 + 双锚点标定。 |

**没有采用**：Hunyuan3D / TRELLIS / 3DGS / 视频扩散等需要 GPU 与大权重，和「零依赖、
27MB ONNX、CPU 可跑」的约束冲突；mid-air 的数据集与训练也不在本次范围。

### 实际改动

1. **新增 `buildCartesianWorld`（默认路径）**：相机内参反投影 `P = eye + dir(u,v)·z(u,v)`，
   只在**有限直角区域** `x∈[-60,60]m、z∈[-90,0]m、y∈[-2,40]m` 内建面；没有 `gVis`、
   没有 `panoU` 镜像、没有球面半径。
2. **深度预处理**：2/98 分位归一化 → 5×5 中值（压单像素尖刺）→ 地平线以下横向 7 列 /
   纵向 5 行各向异性低通（压掉水面倒影造成的「漂浮碎片」）。
3. **断崖跳面**：四边形四角距离比 > 2.6 时跳过（`pipeline/geom.py` 同一判据），
   不再用前拉吸附硬拉成鳍。
4. **法线**：世界坐标邻居差分叉积 + 3×3 平滑，统一朝上；顶点色烘焙太阳 ndl + 断崖 AO。
5. **行走高度场**：把反投影顶点烘进 1m XZ 网格（格内取最高点 + 两轮邻域补洞），
   玩家站在真实几何上；`walkRadius=22m`。
6. **边界**：照片投影成梯形，alpha 2% 渐隐；不融色（融色会形成灰带）。
   `?cart=0` 回退 v1.11 的视锥路径，`?v1=1` 回退经验曲线，便于 A/B。
7. **质量门**：`tools/quality_gate.py` 生成 REFERENCE | RENDER | OVERLAY 三联图，
   输出 `gray_mae / mean_centered_mae / edge_iou / luma_ratio`；回归新增 4 条断言。
8. **工程清理**：删除 `enterWorld` 里**第二份逐字相同**的 10 次曝光预热循环
   （第一份已收敛后白跑最多 10 帧全屏渲染）；回归静态断言锁住。

### 验证

- `node tools/test_regression.mjs` 全绿：直角重建 `6650 三角 / 96×60 顶点全有限、法线归一化、
  索引全部落在 ±60m × −90m 区域内`；v1.11 视锥路径与 `?ldi=1` 实验层仍可复现。
- `python tools/test_regression.py` P0/P1 + P2 全绿；`node --check` 全部通过；
  `bump_cache --check` 12 处版本串一致。
- `node tools/smoke.mjs` **5/5 PASS**（alpine / desert / asset-glb / ring-on / skip-sky）。
- 走动 6m 画面 MAE **62.21 → 66.07 /255**（v1.11 → v1.13），视差进一步拉开；
  湖面碎片、视锥尖刺均为 0（截图逐张检查）。
- `quality_gate.py` 三联图：`ab/quality_v111.png` / `ab/quality_v113.png`。

### 仍存在的限制（如实说明）

- **有限区域的边界可见**：照片投影成梯形，窗口外是程序化地面/雾；这是「有限直角区域」
  方案的固有边界，不是伪影。要做到无边界必须做 inpainting / 360° 生成（需生成模型）。
- 单目深度仍是相对深度，尺度来自地面平面 + 远端先验；水面倒影会被当成几何（已低通）。
- 走动范围 22m；再远离开照片覆盖区，边界与重复会更明显。

## [1.11.0] - 2026-11-13

基座：远端 `main`（v1.10.0）。静态资源缓存串 **1.10.0 → 1.11.0**（index.html 12 处）。
本版只解决一件事：**生成出来的世界像一张平铺在地面上的照片卡片**。

### 症状与根因（先定位代码，再改代码）

| 症状 | 根因 | 位置 |
| --- | --- | --- |
| 中远景像贴在背景上的一张照片，没有前后关系 | `distOf` 经验曲线给出 80~280m，随后 `lerp(t, 128, smoothstep(18,46,t))` 把 **46m 外全部吸到 128m 固定圆柱**，可见像素几乎等距 | `js/world.js::buildRelief` |
| 下半张照片是一块绝对平的地面 | 地平线以下一律 `gy=0`、`t=eye/(-sinφ)`，**深度图完全不参与**；`heightAt` 还被乘 0.12 | 同上 / `build()` |
| 四周一圈硬切边，像插在水面上的卡片 | 视锥掩码只留 12% 渐隐带，并在 `gVis<0.5` 处直接砍面，露出与照片无关的程序化地面 | 同上 |
| 走动几乎没有视差 | 上面三条叠加，几何都堆在同一半径上 | — |

### 参考方案（GitHub 检索结果）

- `vt-vl-lab/3d-photo-inpainting`（CVPR 2020）：单图 RGB-D → **Layered Depth Image**，靠分层与遮挡修补产生真视差，而不是一张连续曲面 → 本项目「近/远分层」与「断崖不要连成拉伸面」的思路来源。
- `OneeMe/splat-drawing` / `imichaelnorris/image-3d`：浏览器内 Depth-Anything-V2 → 单图 2.5D 点云/splat，确认「深度必须真正参与几何」而不是只做颜色。
- `jhaabhijeet864/DepthWizard`、`AyberkA0/pbrcelain-material-generator`：Z 尺度（vertical exaggeration）、2/98 分位裁剪、局部均衡、去 bowl/dome → 竖直夸张与深度低通的依据。
- `s-macke/VoxelSpace`：明确写出 2.5D 高度场**无法表达建筑/树木等竖直物体**，所以不能只调高度系数，必须用真三维网格。
- 仓库内现成答案：`pipeline/geom.py` 的**双锚点地面平面标定 + 断崖跳四边形**、`js/glbworld.js` 的 **backdrop 无视差远景层**（README 已写明这是消掉沙盘感的关键）。这两条本项目自己已验证，直接移植到在线链路，优先级高于外部方案。

### 实际改动

1. **距离改为物理标定**（`calibrateScale`，与 `pipeline/geom.py` 同一口径）：近端锚点来自地面平面几何 `z=eye/(-sinφ)`，远端锚点是声明式先验 520m，解 `z = A·(d+r0)^(-p)`；标定退化或 `?v1=1` 才回退旧经验曲线。**删掉 128m 圆柱收敛**，并把 `t` 上限从 420m 放到 700m。
2. **近/远分层**：>160m 的四边形拆到 `reliefBackdrop`，渲染时跟随相机平移（复用 GLB 资产模式的 `uBackdrop`）；近景保持世界坐标与真视差。
3. **地面不再是死平的 y=0**：用「比同排地面基线明显更近的深度」挤出近景物体；配 0.10 死区、5×5 低通、9 行低通、镜像接缝衰减、≤1.4m 上限。平坦水面/地面保持完全水平（第一版没有低通，湖面被挤成一圈圈波纹条纹，已回退重做）。行走高度场同步跟随挤出面，避免玩家穿过几何；`walkRadius` 12 → 24m。
4. **卡片硬边改为融色**：新增顶点属性 `aEdge(vec2)`，`FS_MAIN` 的 mode 4 在视锥侧边把照片纹理融进天光/地面色、底边融进地面色；渐隐带 12% → 25%，但只在 `gVis≥0.5` 建面，避免视锥外的镜像拉伸纹理露出来。缺 `aEdge` 的网格（GLB 资产、程序化地面）默认 0，行为不变。
5. **体积感**：法线明暗范围 0.55~1.16 → 0.42~1.34，深度断崖处加接触阴影 AO；竖直夸张 1.10（`?vg=` 可调），前拉吸附阈值在标定模式下 1.25 → 1.9（否则连续地形被误判成断崖，吸成台阶）。
6. **A/B 开关**：`?v1=1` 旧路径、`?flat=1` 关近景挤出、`?vg=1.35` 调竖直夸张。

### 验证

- `node tools/test_regression.mjs` 全绿，新增判据：平坦地面 `max|y|=0.000m`（无波纹）、断崖深度 `max|y|=0.244m`（确有挤出）、纵深 `p10=31m / p90=578m (×18.6)`、**128m 圆柱占比 0.36%**、近/远分层与 `?v1=1` 旧路径可复现。
- `python tools/test_regression.py` P0/P1 + P2 全绿；`node --check` 全部 JS 通过；`tools/bump_cache.mjs --check` 12 处版本串一致。
- `node tools/smoke.mjs` **5/5 PASS**（alpine / desert / asset-glb / ring-on / skip-sky，含 GLB 资产模式，验证新 shader 属性默认值不破坏旧网格）。
- 真实 Edge 截图 A/B（`shot_v111_compare.png`）：同一机位走动 6m，画面 MAE **39.64 → 62.21 /255**（×1.57），去均值 **45.45 → 62.91**（×1.38）——视差确实被拉开，不是只改了静态首帧。
- 三个内置世界（alpine / desert / coast）逐张检查：无波纹条纹、无灰色大块、无点状拉伸条纹。

### 刻意没做 / 已知残留

- **LDI 裙边与遮挡修补**（3d-photo-inpainting 的核心）未做：物体仍是单层连续曲面，绕到侧面仍会看出「卡片」，根治需要断崖处跳过四边形 + 生成侧面/背景补洞，属于下一版的结构改造。
- 宽渐隐（38%）与视锥外建面都试过，分别露出灰弧与点状条纹，已回退并把原因写在 `world.js` 注释里；当前 25% + 融色是截图 A/B 后保留的折中。
- 单目深度仍是相对深度，尺度来自地面平面 + 远端先验，和离线管线一样不是真测距；天空/远景仍会到达标定上限。

## [1.10.0] - 2026-11-13

基座 `d3401ca`（本日 22:11），动工前与推送前均核对远端无并发提交。本版做算法分析报告
**第六章 P3 工程清理**，不改任何深度/几何/训练算法。静态资源缓存串 **1.9.0 → 1.10.0**
（index.html 12 处）。

**先说明一个前提**：报告第六章列出的「重复曝光预热循环」「asset 模式碰撞写死 `()=>0`」
「GLB 地面与眼高叠加成 3.4m」，远端在 v1.8.0～v1.9.0 期间已自行修掉（曝光预热现在只有
一份 8 次循环；`enterAsset` 用 `buildHeightGrid` 从网格烘中位数高度图并用 3×3 邻域补洞）。
本版**不重复修**，只把「重复定义不得回来」写成静态断言。下面才是本版实际改的内容。

### 重复定义与死代码：删掉不会改变行为、但会误导维护者的部分

- `js/world.js` 曾有**两份逐字相同**的 `makeSkyTexture`，函数声明提升使后者静默覆盖前者，
  改错哪一份都会「改了没效果」。现只保留一份。
- 删掉算完从未使用的变量：`buildRelief` 里的 `w = smoothstep(-0.03, 0.02, sp)`、
  `build()` 里的 `skyBottom`、`geom.build_mesh` 里的 `idx = []`。
- 删掉 `pipeline/geom.py` 的 `estimate_pitch` / `_fit_scale`（约 50 行）：二者互相调用、
  全仓库零调用点，是早期「搜索俯仰角」标定方案，主流程早已换成 `calibrate_scale`
  的地面带 + 双锚点。需要复现实验可从 git 历史找回。
- `tools/test_regression.mjs` 新增静态断言锁住上述几项，防止复制粘贴再次引入。

### 资产模式的两个一致性问题

- **曝光预热相机看错了位置**：`enterAsset` 写死 `eye: [0, 1.7, 0]`，但 GLB 是相机系、
  地面约在 `y=-1.7`，玩家实际眼位是 `heightAt(0,0) + 1.7 ≈ 0`。旧代码等于从**高 1.7m 的
  另一个视角**测画面亮度去定曝光，和进入世界后看到的画面不是一回事。现改为
  `groundY + world.eyeHeight`。
- **高度图越界会让人弹起来**：`buildHeightGrid.at()` 在网格外直接 `return 0`，而网格内
  地面约 `-1.7m`；一旦走到边界，眼高瞬间抬高 1.7m。现改为钳到边缘格继续双线性插值。
  正常游玩受 `walkRadius=55`（网格半径 90）限制走不出去，这属于边界防御。

### 同一张照片重复进入，不再重复推理

`js/depth.js` 加了内存 LRU（最多 4 张）：key 是「图片尺寸 + 16×16 缩略图内容指纹」，
不是纯 URL —— CDN 同名换图、data URL 重建都不会误命中。内置 `alpine` 一次 CPU 推理约
5 秒，返回同一世界时现在直接复用深度图。回归用 mock ONNX session 实测：连续两次
`estimate` 的 `session.run` 次数 **1 → 1**（第二次 `cached=true`），`clearCache()` 后
**→ 2**。返回值 `data` 为只读约定，上层 `buildRelief` / 预览都只采样不写入。

### 历史实验资产标注

`train/`（964KB `weights.json` + 3 个 `.f32` + 早期小网络脚本）全仓库**零运行时引用**，
主线早已换成 ONNX Depth-Anything-V2-Small 与 `worldmodel/`。新增 `train/README.md`
说明其 deprecated 状态与用途；**不删文件**，避免破坏想复现当时实验的人。

### 本版刻意未做的 P3 项

以下都会改变画质或生成分布，本机无法验证收益，不擅自上线：把 518×518 降到 392×392
（算力 0.57× 但损失细深度）、WebGPU/WebNN EP 优先 + WASM 回退、IndexedDB 跨刷新缓存、
推理移入 Web Worker、`extend_fov` 的镜像 + 强模糊外延（侧视拉伸条带，需截图 A/B）。

### 验证

`node tools/test_regression.mjs` 9 项全绿（含新增 P3 两条）；`python tools/test_regression.py`
P0/P1 + P2 全绿；浏览器冒烟 **5/5 PASS**（alpine / desert / asset-glb / ring-on / skip-sky）；
`pipeline/reconstruct.py` 端到端产出 305,458 三角，自检 35.73dB / 尺度门 0.0522 OK；
JS 与 Python 全量 `py_compile` / `node --check` 通过。

## [1.9.0] - 2026-11-13

基座 `78cb9fe`（本日 18:17），动工前核对远端无新提交。本版只修世界模型训练侧的
**P2 五项**，不碰前端几何与 P3 工程清理。静态资源缓存串 **1.8.0 → 1.9.0**（index.html 12 处）。
所有改动都能用 `python tools/test_regression.py` 在**无 torch / 无 GPU** 的机器上验证。

### 5.1 动作语义：训练学到的控制，推理时终于用得上

- **修复前的症状**：按 `forward / turn_left` 下达指令，画面不按指令动；动作可控性评测
  基本无从谈起。根因不是模型容量，而是两侧各写了一份动作编码：训练标签是从真值位姿
  算出的 7 维 `[轴角3, 相机系平移3/speed, logFOV1]`，推理 DSL 却只手填
  `a[1]/a[3]/a[5]` 三个标量。
- 具体错位：`forward` 训练里是相机系 **−z 平移 ≈ −1**，推理里被写成 `a[3] = −0.12`
  （x 平移）；`turn_left` 训练的轴角 y 是 **−0.05**，推理给的是 **+0.05，符号反了**；
  `look_up` 训练落在 `a[0] = −0.02`，推理却写进 `a[5]`（z 平移）。
- 修法：新增 `worldmodel/action_codec.py` 作为唯一来源，`prepare_data.py` 与
  `parse_actions()` 都调用同一个 `rel_action`；DSL 名称先构造与 `render_traj.mjs`
  相同的相邻位姿再编码，符号与耦合自动一致。回归复现 300 帧合成轨迹标签比对：
  `turn_left / turn_right / look_up / idle` 与 DSL 的最大偏差 **< 0.02**，
  forward 归一化平移模长 **1.000**（训练均值 0.92，方向因 yaw 累积漂移而分散）。
- **顺带修掉一个会抵消上面成果的验收 bug**：`scripts/eval.py::phase_shift` 的互功率谱
  写反成 `F(a)·conj(F(b))`，返回的是 a 相对 b 的位移 —— 合成图内容右移 6 像素会报
  `dx=-6`。相机左转时场景应右移，旧判据会把**正确**的 `turn_left` 系统性判成失败。
  改为 `F(b)·conj(F(a))`，并用合成位移单测锁住符号。

### 5.2 RoPE：位置编码恢复旋转等变性

- 旧实现把 `(t,y,x)` 三轴各 20 维频率拼成 60 维再补 4 个零，而 `apply_rope` 用
  half-split 旋转（第 j 与第 j+32 通道配对），要求前后半频率相同。前半是
  "t 的 20 + y 的前 12"，后半是"y 的后 8 + x 的 20 + 常数"，完全不配对。
- 新增 `worldmodel/model/rope.py` 把 `d/2` 个频率通道分给三轴后复制成 `d` 维。
  单测「整体平移后注意力分数只依赖相对位移」的误差从 **0.984 降到 8.9e-16**；
  该单测对旧实现会失败，能防回归。动作 token 的位置同时改为沿用目标帧**起始**位置，
  与因果掩码把它归入最后一帧的约定一致。

### 5.3 初始化：训练第一步梯度就贯通所有层

- 旧代码把输出头权重与偏置全部置零，`dL/dx = W_headᵀ·grad = 0`，
  开训最初若干步**所有 Transformer block 的梯度恒为零**，只有 head 在拟合常数速度场。
- 改为标准 DiT 做法：零初始化每个 block 的 adaLN gate（第 3、6 段），残差路径初始为
  恒等映射；输出头改为常规小随机初始化（`std=0.02`）。

### 5.4 CFG 与采样：三条条件路径统一成一条

- **修复前的症状**：CFG 4.0 的引导方向不对，`val loss` 与 rollout 表现不同源。
  训练时 `drop_action` / `drop_init` 是两次独立 10% dropout，推理的无条件分支却只传
  `action*0`（`act_embed` 仍有 bias），初始帧与域标签都没丢，三条路径互不等价。
- 现在训练与推理共用 `uncond=True`：一次联合 drop 动作 + 初始帧 + 域标签，
  配置键统一为 `drop_cond: 0.1`（8 个 yaml 已同步，保留 `drop_action` 读取回退）。
  Euler 采样从**左端点** `t=s·dt` 改为**中点** `t=(s+0.5)dt`（末步 0.875 → 0.9375）。
  验证循环临时换上 EMA 权重再评估、评估完恢复训练权重，与默认加载 EMA 的 rollout 同源。

### 5.5 训练吞吐：VAE 在线编码可归零

- 按 `configs/s1_base.yaml`（batch 2 / grad_accum 16），旧路径每个 optimizer step
  要 **192 次 VAE encode**，2000 步约 **38.4 万次**，且每步重复解码 JPEG。
- 新增可选的 `scripts/preencode_latents.py`（训练机上跑一次）与 `.latent` 分片格式：
  单样本 6×4×32×48 float32 ≈ **147KB**，比 6 张 JPEG 更小。
  `TarWindows` 检测到 `.latent` 走 `numpy.frombuffer` 直读，trainer 检测到 `z_init`
  就跳过 VAE，**该瓶颈归零**。原图片分片行为完全不变，验证循环也走同一读取路径。

### 验证与遗留限制

- `python -m py_compile` 全部通过；`tools/test_regression.py`（P0/P1 + 自动串联 P2）
  与 `tools/test_regression_p2.py` 全绿，均不需要 torch。
- **本机没有 torch / CUDA / VAE 权重**：训练全链路、`preencode_latents.py`、
  rollout 与 `scripts/eval.py` 均未在真实训练机上实跑；本版验证的是语义定义、
  数学性质与代码结构，吞吐与收敛收益需在训练机上确认。
- 报告 5.4 里的经验性采样改动（SDEdit 式噪声起点、CFG rescale、idle 预热历史、
  logit-normal 时间采样）**本版未做**：它们会改变生成分布，无 torch 环境无法验证收益，
  贸然改默认有回归风险。冷启动时历史窗口仍由初始帧复制 4 份填充。
- DSL 动作幅度默认取合成轨迹的典型值（`speed=0.09`、`move_step=0.09`、
  `yaw_step=0.05`、`pitch_step=0.02`）；换数据集时用
  `parse_actions(..., speed=, move_step=, ...)` 或配置对齐，否则动作量纲会失配。
- worldmodel 产物**仍未接入前端**：线上画面依旧由 WebGL 渲染 3D 网格（属阶段 2 / P3）。

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
