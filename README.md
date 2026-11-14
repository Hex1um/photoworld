# PHOTOWORLD — 走进照片里的世界

把一张风景照片还原成"拍摄那一刻你面前的三维场景"，可以自由走动、转头、走近看。纯前端、零构建，深度推理与重建全部在浏览器本地完成，照片不上传。

## 用法

```bash
# 静态站点，任意 http 服务即可（不能直接用 file:// 打开，浏览器会阻止读取像素）
python -m http.server 8099 --bind 127.0.0.1
# 打开 http://127.0.0.1:8099/
```

- `?w=alpine|desert|coast|winter` 直接进入某个内置世界
- `?img=corpus/xxx.jpg` 直接打开任意**同源**图片进世界（批量质检用，不经画廊）
- `?instant=1` 关闭所有过渡（截图/自动化用）
- `?skip=sky,far,near` 关闭某一层调试
- 背面剔除（v1.17）：重建表面默认只画正面，转到侧后方看到的是世界幕与雾，
  而不是糊成一片的「纸片背面」；`?cull=0` 回到 v1.16 的双面渲染做 A/B
- 地面远处渐隐（v1.17）：地面顶点按到相机的水平距离渐隐到地面基色，
  压掉极坐标网格的放射状拉丝；`?gblend=0` 关闭做 A/B
- `?ring=1` 打开"视锥外程序化地平雾带"实验（**观感仍不达标，默认关闭，不建议实际开启**）
- `?x=6&z=0&yaw=0` 指定出生位置，用来对比走动视差（截图/评测用）
- 重建 A/B：`?v1=1` 回到 v1.10 的经验曲线 + 死平地面，`?flat=1` 只关近景挤出，`?vg=1.35` 调竖直夸张
- 语义分组（v1.16）：默认开，`?sem=0` 关闭；结果写到 `body.dataset.semstats`
- 扇环延伸（v1.16）：默认开，`?ext=0` 关闭，`?extr=3..8` 调保留半径（默认 5 块 × 32m），
  `?api=http://127.0.0.1:8100` 指向本地 GPU 生成服务（不填则纯程序化）
- **实时变化（v1.18，类 Oasis）**：HUD 里的「实时变化」按钮，开启后两个滑杆
  · `混乱度 chaos`（`?chaos=`）0 = 整个世界按照片这一个主题生成（连续无缝）；
    1 = 每个扇区的地貌与之前、与邻居**完全无关**（主题原型来自 1268 张图集的聚类）
  · `记忆力 memory`（`?memory=`）1 = 生成后基本不再改变；0 = **眼前不动、身后迅速变化**：
    视锥外的扇区按 ~1.8 次/秒的概率重掷（epoch+1），转身看到的是新地貌
  · 也可以直接 `?live=1&chaos=0.6&memory=0.3` 进世界；状态记在 localStorage
- 单体真网格（v1.16，可选）：`?obj3d=1&api=http://127.0.0.1:8100`，服务端 InstantMesh/TripoSR
- 质检：`?dbgpos=1` 把重建网格挂到 `world.dbgPos`；`?skip=floor,cards,shell,ground` 分层关；
  `node tools/probe.mjs --url ... --exprfile tools/diag_jumps.js` 查竖直跳变

操作：`WASD` 行走 · 鼠标环视（点击画面锁定指针，或按住左键拖动）· `Shift` 奔跑 · 滚轮调视野 · `Esc` 暂停 · `H` 隐藏 HUD。触屏为左半屏摇杆 + 右半屏拖动视角。

## 原理

核心是**单目深度重建**，不是把照片贴在地面上：

1. **深度估计**（`js/depth.js`）
   浏览器内跑 Depth-Anything-V2-Small（ONNX 量化版 27MB，ort-web + WASM 单线程）。图片缩放到 518×518 → 归一化 → 推理出每个像素的相对深度 → 自动判定极性（天空必须在远端）。首次加载 27MB，之后走浏览器缓存。

2. **直角坐标反投影重建**（`js/world.js` → `buildCartesianWorld`，v1.13 起默认）
   不再用「屏幕行=仰角、深度=半径」的视锥/球面参数化 —— 那套结构在边界必然产生
   尖刺与平铺贴片。现在按 video-to-3D 方案的同一口径重建：
   - **反投影**：`P = eye + dir(u,v)·z(u,v)`，`dir` 由相机内参（HFOV 62°）给出；
     每个像素落到世界坐标，几何就是世界坐标本身。
   - **有限直角区域**：只在 `x∈[-60,60]m、z∈[-90,0]m、y∈[-2,40]m` 内建面；
     没有 `gVis` 视锥掩码、没有镜像平铺、没有球面半径。
   - **深度预处理**：2/98 分位归一化 → 5×5 中值（压单像素尖刺）→ 地平线以下横向 7 列 /
     纵向 5 行各向异性低通（压掉水面倒影造成的「漂浮碎片」）。
   - **双锚点标定**：沿用 v1.11 的成果，深度 → 米（`z = A·(d+r0)^(-p)`）。
   - **断崖跳面**：四边形四角距离比 > 2.6 时跳过（`pipeline/geom.py` 同一判据），
     不把前景与背景连成拉伸的鳍。
   - **法线/光照**：世界坐标邻居差分叉积 + 3×3 平滑、统一朝上；顶点色烘焙太阳 ndl 与断崖 AO。
   - **行走高度场**：反投影顶点烘进 1m XZ 网格（格内取最高点 + 两轮邻域补洞），
     玩家站在真实几何上；`walkRadius = 22m`。
   - 边界就是照片投影的梯形，alpha 2% 渐隐；`?cart=0` 可回到 v1.11 的视锥路径做 A/B。

3. **光与空气**
   太阳方位取天空最亮处，雾色取地平线色；光照在线性空间计算（sRGB 解码 → 光照 → ACES → sRGB 编码）；入景前用 64×36 探针 FBO 迭代反解曝光，让画面平均亮度对齐原片。

4. **后期**（`js/glutil.js` FS_POST）
   FXAA、暗角、极轻微颗粒。

模型不可用时自动回退到早期的"天际线 + 分层贴片"方案（`buildCurtain`），离线或加载失败也能进世界。

![v1.10 平铺卡片 → v1.11 标定纵深 + 融色边缘](shot_v111_compare.png)

4. **语义分组**（`js/semantics.js`，v1.16）
   SegFormer-B0 / ADE20K 150 类（int8 ONNX，4.4MB），映射到 12 个几何分组
   （天空/天花板/墙/地面/水面/植被/建筑/人/车/器物/山体/未知）。职责边界是 A/B 试出来的：
   **语义只负责天空 alpha、天花板几何、单体实例切分**；地面/水面几何仍由深度 + 地平线决定
   —— 语义地面是斑块状的，参与几何会在斑块边界立起无数薄墙。模型失败自动回退启发式
   （深度 + 地平线 + 前景框），只能分天空/地面/天花板。

5. **扇环外延伸**（`js/extend.js`，v1.16）
   重建只覆盖相机前方 `x∈[-55,55]m, z∈[-100,0]m`。走出这块之后由 tile 流式接管：
   **全局连续 fBm**（同一函数、同一坐标 → 相邻 tile 天然无缝，不需要接缝求解，
   可以 O(1) 随机访问任意 tile）+ 四边高度硬约束（边界继承重建高度）。
   延伸地面的**颜色**来自重建时烘的 48×32 照片地面颜色场，所以雪地外面还是雪。
   接管 `world.heightAt` 后玩家真的站在延伸地形上，不再"一步踏进雾里"。
   可选 `POST /api/extend` 换成 Terrain Diffusion / InfiniteDiffusion 的 learned 后端。

## 参考方案（v1.11 在 GitHub 上检索的结论）

| 仓库 | 结论 / 借鉴点 |
| --- | --- |
| `vt-vl-lab/3d-photo-inpainting`（CVPR 2020） | 单图 RGB-D → Layered Depth Image，靠**分层 + 遮挡修补**产生真视差；本项目据此拆出远景层，并明确"不要把所有像素连成一张连续曲面"。 |
| `OneeMe/splat-drawing`、`imichaelnorris/image-3d` | 浏览器内 Depth-Anything-V2 → 单图 2.5D 点云/splat；确认深度必须真正参与几何，而不是只影响颜色。 |
| `jhaabhijeet864/DepthWizard`、`AyberkA0/pbrcelain-material-generator` | Z 尺度夸张、2/98 分位裁剪、局部均衡、去 bowl/dome；用于竖直夸张与深度低通。 |
| `s-macke/VoxelSpace` | 明确写出 2.5D 高度场无法表达建筑/树木等竖直物体，只调高度系数没有用。 |
| 仓库内 `pipeline/geom.py`、`js/glbworld.js` | 双锚点地面平面标定 + 断崖跳四边形、backdrop 无视差远景层：本项目自己已验证的方案，优先移植到在线链路。 |

**没有采用的方案**：3D Gaussian Splatting（`antimatter15/splat`、`apple/ml-sharp`）与更大的度量深度后端
会破坏"零依赖 / 27MB 模型 / CPU 可跑"的约束；LDI 裙边与遮挡修补需要额外侧面几何或 inpainting，
留作下一版结构改造（见「已知限制」）。

## 两条产线

| 产线 | 何时用 | 产物 | 尺度 |
|---|---|---|---|
| **离线重建**（`pipeline/`） | 想要"能自由行走、尺度真实"的世界 | `world.glb` + `world.json` | 地面平面约束（相对深度）或度量深度 |
| **在线预览**（`js/`，本页） | 打开即用、无需等待、无需 GPU | 浏览器内实时生成 | 相对深度 + 近似尺度 |

```bash
python -m pip install onnxruntime numpy pillow
python pipeline/reconstruct.py worlds/alpine.jpg --out out/alpine   # → world.glb
```

离线管线的设计与参数见 [`pipeline/README.md`](pipeline/README.md)。

### 资产模式（加载离线产物）

```
https://…/?glb=assets/world-alpine.glb      # 直接加载 glb，跳过在线重建
```

`js/glbworld.js` 自己解析 GLB（不引入 three.js）：取出 POSITION/NORMAL/TEXCOORD_0/索引与内嵌 JPEG，
喂给现有渲染器，交互、天空、雾、曝光、FXAA 全部复用。仓库里的 `assets/world-alpine.glb`
就是 `pipeline/` 用 `worlds/alpine.jpg` 生成的样例（198,290 三角 / 5.9MB，分 2 层）。

远景层（`backdrop`）在渲染时跟随相机平移 —— 远景本就不该有视差，
这样走动时山不会"滑过来"，是消掉沙盘感的关键；近景 70m 内是真几何，视差正确。

### 多层 + 顶点色资产

`glbworld.js` 除了 `TEXCOORD_0` 也读 `COLOR_0`（VEC3/VEC4 都吃），因此**无贴图的纯顶点色 GLB 也能直接加载**：
> ?glb=xxx.glb → 没有内嵌 JPEG 时自动走顶点色材质（`mode 1`），天光取默认值。

层名约定：叫 `backdrop` 的层做无视差远景，其余层（`near` / `prop` / 任意名）按顺序合并成一条近景通道，
所以 1 层、2 层、3 层以上的产物都能走同一条渲染路径，不需要改渲染器。

资产模式会在加载时从网格顶点烘一张 2m 精度的高度图（每格取**中位数**，对"树冠抬高"和"地下尖刺"都稳），
因此离线产物不带 `world.json` 也能正常贴地行走，不会漂在半空。

## 冒烟测试

```bash
python -m http.server 8137 --bind 127.0.0.1
node tools/smoke.mjs --base http://127.0.0.1:8137            # 5 例：4 个内置世界 + 资产模式
node tools/smoke.mjs --base … --shots /tmp/smoke --only alpine
```

判据是硬的：`body[data-ready] === '1'` 且 `window.__errs` 为空、无 `data-err`，任何一例失败退出码 1。
一行改名（例如 `opts`→`opt`）就能让主路径 100% 进不去世界，这个脚本 20 秒能抓到。

不开浏览器也能跑算法回归（地面平面 / 断崖补洞 / 深度预处理 / 验收门 / 训练数据层 /
动作编码 / RoPE 等变性 / 预编码 latent / 重复定义与死代码 / 深度图缓存 / 缓存串）：

```bash
node tools/test_regression.mjs
python tools/test_regression.py            # 末尾会自动跑 P2 回归
python tools/test_regression_p2.py         # 也可单独跑世界模型部分
node tools/bump_cache.mjs --check
```

离线重建的自检分三层：贴图与 UV 错位看 PSNR，**尺度错误**看地面平面尺度门（相机眼高是绝对量），
**视场错误**必须由 `--verify-fov` 或 `--fov` 覆盖 EXIF 提供独立相机做交叉泼点。
单视图自洽自检在数学上无法约束尺度/FOV，别只看 99dB 就认为尺度对了。详见 `CHANGELOG.md`。

## 多视角批量质检（v1.17；v1.20.3 补清晰度基线）

每张图取 **16 个方位 + 1 个俯视**，逐张算客观指标，用来回答「这张照片重建出来的地貌
在别的角度看还站不站得住」：

```bash
node tools/multiview.mjs --base http://127.0.0.1:8099 \
  --list reports/mv_sample.txt --views 16 --gl hw \
  --out shots/mv --json reports/mv.json \
  --baseline reports/clarity_baseline.json
```

- `--list` 每行一个相对 base 的图片路径（`corpus/xxx.jpg`），`--corpus <dir>` 可直接整目录
- `--gl hw|sw`：hw = D3D11 真 GPU（默认），sw = SwiftShader 软件光栅。
  **两者观感差很多**（软件光栅在掠射角会拉出放射条纹），指标只能同后端对比
- 几何指标：`voidFrac` 空底色占比（洞/露底）、`innerVoid` 中央 ROI、`lowVoid` 下 40% 地面、
  `borderVoid` 外圈、`darkFrac` 塌黑、`stripe` 竖直/水平能量比（竖幕/拉花）、
  `detail` 平均梯度（保留兼容，但它分不出「糊片」和「雾」，别只看它）
- 清晰度六项（`tools/clarity.mjs`，越大越清晰）：`hfEnergy` 3×3 模糊残差、`farHF` 远景带、
  `centerHF` 中央 ROI、`lapVar` / `centerLapVar` 拉普拉斯方差（×1000）、`localContrast`
  5×5 局部标准差。**A/B 优先看 `centerHF` 与 `centerLapVar`**：天空和程序化地面是低频区，
  全图均值会被稀释。`node tools/test_clarity.mjs` 用合成图锁住「清晰 / 糊片 / 雾」的区分度
- `--baseline <file>` 第一次跑写入基线，之后**完全同参数**再跑即输出逐指标 A/B 表
  （基线均值 / 当前均值 / 变化 / 改善与退化的图数）。报告里带 `capture` 采集条件
  （分辨率、dpr、光栅后端、视角数、稳定时长、指标版本），任一不一致直接拒绝对比，
  避免把「换了后端」误读成清晰度变化
- 照片贴图 A/B：URL 加 `?tex=0` 回到旧的 420px 分析画布，`?tex=4096` 抬高上限；
  默认 `displayMaxEdge=2048`。`body.dataset.texW/texH` 会写入实际贴图尺寸，
  用 `tools/probe.mjs` 可直接断言
- 空底色用「画面主色」判定（16 级直方图最大桶），不依赖雾色 —— 夜景里雾色接近黑，
  用雾色比色会把整屏误判成雾

## 数据驱动调参（图集标定）

`tools/` 下是一条完整的语料 → 评测 → 调参链路（仅本地使用，不参与发布）：

```bash
python tools/harvest.py --target 1200 --out corpus --pages 16 --min-w 800
python tools/normalize.py corpus                    # 统一成 420 宽基线 JPEG（Node 才能全量解码）
node tools/eval.mjs --corpus corpus --sets baseline,strict,loose
```

`harvest.py` 是**内存校验**的：只有 PIL 真解码通过（≥700×400、横构图）的字节才落盘，
文件名取内容哈希（天然去重）。旧版先写 `.part` 再删，热链返 404/502 时会留下
一堆几十 KB 的假 jpg（一次实测 969 个文件里只有 461 张能读）。

`eval.mjs` 用 Node 里的 `jpeg-js` 解码，直接调与浏览器完全相同的 `analyzePixels`，统计地平线截断率、天空识别失效率、高度饱和率、低谷率等指标。默认阈值来自 503 张语料上 7 组参数的扫描结果（score 0.704 vs 基线 0.645）；v1.17 把语料扩到 1000+ 张复核了同一组参数。

`tools/shot.mjs` 是基于 CDP 的真实等待式截图工具（等页面 `data-ready` 再截），比 `--virtual-time-budget` 可靠。

## 目录

```
index.html
css/style.css
js/mat.js          矩阵 / 随机数
js/glutil.js       着色器与 WebGL 封装
js/reconstruct.js  照片分析（analyzePixels 纯函数，浏览器与 Node 共用）
js/depth.js        深度模型推理
js/world.js        几何：深度表面重建 / 分层回退 / 程序化道具
js/semantics.js    语义分组（SegFormer-B0 / ADE20K 150 类 → 12 个几何分组）
js/extend.js       扇环外延伸（全局 fBm tile + 边界无缝 + 区域底板）
js/object3d.js     单体真网格升级（可选，需服务端 InstantMesh/TripoSR）
js/renderer.js     渲染管线 + 后期 + 曝光探针
js/player.js       第一人称控制
js/audio.js        程序化风声
js/app.js          画廊 / 载入 / 主循环 / HUD
models/            Depth-Anything-V2-Small 量化 ONNX（27MB）
                   + segformer_b0_ade20k.onnx（int8，4.4MB，python tools/fetch_models.py --semantics）
vendor/ort/        onnxruntime-web（WASM 单线程）
worlds/            内置场景图
train/             历史实验：早期本地小网络（deprecated，运行时无引用，见 train/README.md）
tools/             语料采集 / 归一化 / 批量评测 / 多视角质检 / CDP 截图 / 算法回归 / 缓存升版
worldmodel/        Causal World DiT（action_codec 统一训练/推理动作，data/ 读 tar 与 .latent）
server/            本地 GPU 生成服务（FastAPI）：/api/extend、/api/object3d，见 server/README.md
training/          数据集整理与训练脚本（四条能力对应四个可训练模型），见 training/README.md
CHANGELOG.md       面向使用者的更新日志
```

## 视锥实时生成（v1.18）

扇环外不再是「镜像副本」或「一套固定噪声铺满 360°」，而是**只生成相机视锥覆盖的格子**：

| 层 | 做什么 | 代码 |
|---|---|---|
| 候选 | 以玩家为中心 9 格（288m）扫一遍，按「是否在视锥内 → 距离」排序 | `extend.js::update` |
| 预算 | 视锥内缺口 > 8 时单拍补 8 块（每块 ~1–3ms），否则 4 块；**转身时优先补身后** | 同上 |
| 保底 | 玩家周围 1.6 格无条件生成，避免转身瞬间脚下是空的 | 同上 |
| 主题 | 24 个扇区（15°）× epoch；原型提供配色/起伏/频率，hash 提供形状种子 | `themeFor / tintOf` |
| 记忆 | 视锥外扇区按 `(1-memory)²×1.8` 次/秒重掷，只删不重建（看不见的不烧 CPU） | `decayMemory` |
| 行走 | `heightAt` 是 (x,z,sector,epoch) 的解析函数，**不限制移动范围**，脚下永远与渲染同源 | `themedHeight` |

混乱度=0 时 `themedHeight` 逐值等于 v1.16 的 `outsideHeight`（回归测试守住这条），
所以「同一主题」不会引入任何接缝；混乱度=1 时每个扇区换一套原型与噪声种子，边界允许硬跳变
（这正是"完全无关"的定义）。

参考：Oasis（Decart）逐帧生成 + 短上下文导致「走远了 / 转身后地貌就变了」是其公开已知的短板，
官方方向是滑动窗口与显式记忆；本项目的 `记忆力` 就是把这个**失效模式变成可调参数**。
分块流式（load near / unload far）与「coherence from shared fields」的做法参考了
常规无限地形工程实现（Minecraft/No Man's Sky 式 chunk streaming）。

## 已知限制

- 单目深度是**相对深度**，不是真实测距：前端仍用经验映射（近端 ~2.5m、远端 ~250m），
  离线管线已改用地面平面 + 双锚点物理标定（上限与先验联动），两条链路的尺度模型尚未统一。
- 深度不连续处（前景遮挡背景）做单轮单侧"前拉"补洞，边缘偶有轻微拉伸；地面行豁免，
  保证 `y=0` 平面不被吸附破坏。
- 双锚点在默认 1500m 先验下 p≈2.28，深度 10% 误差会放大成距离 27.2% 误差；
  天空/远景仍会到达 1800m 上限（属 `backdrop` 无视差背景层），根治需换度量深度后端。
- v1.13 的直角区域只重建照片投影的那一片梯形，背向与侧后方没有几何（程序化地面 + 雾）；
  不再用镜像平铺，所以不会出现重复山形，但也没有 360° 内容。
- 深度图缓存只在内存（最多 4 张）：同一会话内返回已进过的世界免推理，刷新页面后仍需重算；
  跨刷新持久化（IndexedDB）与降分辨率推理尚未做，两者都需要先按截图 A/B 确认画质损失。
- 在线重建模式的照片区域是**有限直角区域**（默认 `x∈[-55,55]m、z∈[-100,0]m`），
  **玩家移动不限制**（`walkRadius=0`）。区域外由 `js/extend.js` 的 tile 流式接管
  （v1.16）：高度用全局 fBm + 边界硬约束，颜色用照片地面色场，所以走出去地形连续、
  颜色接得上；但延伸内容是**统计性推断**，不是照片里的真实地貌。
  要做到"地貌按已有景观继续长"必须接 learned 后端（`server/` 已留接口）。
- 相机视场与重建视场对齐（v1.16.1）：相机视场由 `world.reliefMeta.vfov` 决定，
  不再是固定的 66°。改这条之前，任何图片重建出来都只占屏幕中间 ~80%，四周是天穹的灰。
- 地平线以上做 1.35× 竖直夸张（`?yg=` 可调）；室内/俯拍（地平线 <0.42 或 >0.72）
  自动改用短基线测距（0.6~18m）并关闭夸张。
- 资产模式（`?glb=`）的行走范围是 55m。
- **只生成正面一个扇环**（v1.15）：`buildSurround` 的 5 份镜像副本默认关闭，
  需要时用 `?surround=1` 打开；扇环外是延伸 tile（v1.16）+ 雾。
- 远山与水面之间可能有一条横向的"推断岸线"：山脚像素深度顶在 220m 上限、
  水面的 y=0，两者之间是近乎垂直的面片，贴图会被竖直拉花。v1.16.1 用
  「竖直跨度 / 水平跨度 > 2.5 视为墙」把这些面剪掉，再用区域内底板（照片地面色混雾色）
  兜住洞，读作雾里的远岸——**是兜底，不是真的补出了岸线几何**。
- **近景物体语义识别**：`js/objects.js` 跑 YOLOv8n-seg（EdgeFirst Model Zoo，
  COCO 80 类，14MB，`models/yolov8n-seg.onnx`），识别结果写到 `body.dataset.objects`
  （如 `person,backpack`）；模型失败自动回退纯深度分层。
- 近处物体不做距离延伸（t 限幅 + y 限幅 + 下方邻域中值），远景可放到 220m；
  地面按「平地 + 眼高」反解，近处不会被拉成横向长条。
- 细密植被（草叶、树枝）深度不精确，近景可能略显"纸片感"。
- **仍是单层连续曲面**：断崖处已跳过四边形、不再有拉伸鳍，但没有做侧面裙边与背景补洞
  （`?ldi=1` 的实验层会填洞，但 A/B 里出现色块，默认关）；绕到物体侧面仍能看出它是照片。
- 视锥外不再由镜像补全（v1.13 的直角区域只覆盖照片投影的梯形），转身会看到程序化地面与雾；
  360° 全铺需要 `extend_fov` 级别的外推与修补，属于生成式方案的范畴。
