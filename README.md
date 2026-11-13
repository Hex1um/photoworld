# PHOTOWORLD — 走进照片里的世界

把一张风景照片还原成"拍摄那一刻你面前的三维场景"，可以自由走动、转头、走近看。纯前端、零构建，深度推理与重建全部在浏览器本地完成，照片不上传。

## 用法

```bash
# 静态站点，任意 http 服务即可（不能直接用 file:// 打开，浏览器会阻止读取像素）
python -m http.server 8099 --bind 127.0.0.1
# 打开 http://127.0.0.1:8099/
```

- `?w=alpine|desert|coast|winter` 直接进入某个内置世界
- `?instant=1` 关闭所有过渡（截图/自动化用）
- `?skip=sky,far,near` 关闭某一层调试

操作：`WASD` 行走 · 鼠标环视（点击画面锁定指针，或按住左键拖动）· `Shift` 奔跑 · 滚轮调视野 · `Esc` 暂停 · `H` 隐藏 HUD。触屏为左半屏摇杆 + 右半屏拖动视角。

## 原理

核心是**单目深度重建**，不是把照片贴在地面上：

1. **深度估计**（`js/depth.js`）
   浏览器内跑 Depth-Anything-V2-Small（ONNX 量化版 27MB，ort-web + WASM 单线程）。图片缩放到 518×518 → 归一化 → 推理出每个像素的相对深度 → 自动判定极性（天空必须在远端）。首次加载 27MB，之后走浏览器缓存。

2. **三维表面重建**（`js/world.js` → `buildRelief`）
   按 (方位角 × 仰角) 生成网格，每个顶点按自己的深度放到空间里：
   - 地平线**以上**：半径来自深度，山体/树木获得真实前后关系，走动时正确位移；
   - 地平线**以下**：按"平地 + 1.7m 眼高"反解半径，脚下就是照片里那块地，透视严格正确；
   - **前拉约束**：相邻格半径不允许突变，压掉深度断崖拉出的竖直"鳍"；
   - **镜像接缝**：360° 由镜像平铺补全，接缝处融合两侧半径；
   - 底缘以下冻结半径并渐隐，避免放射状拉伸。

3. **光与空气**
   太阳方位取天空最亮处，雾色取地平线色；光照在线性空间计算（sRGB 解码 → 光照 → ACES → sRGB 编码）；入景前用 64×36 探针 FBO 迭代反解曝光，让画面平均亮度对齐原片。

4. **后期**（`js/glutil.js` FS_POST）
   FXAA、暗角、极轻微颗粒。

模型不可用时自动回退到早期的"天际线 + 分层贴片"方案（`buildCurtain`），离线或加载失败也能进世界。

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
就是 `pipeline/` 用 `worlds/alpine.jpg` 生成的样例（195,180 三角 / 5.6MB）。

## 数据驱动调参

`tools/` 下是一条完整的语料 → 评测 → 调参链路（仅本地使用，不参与发布）：

```bash
python tools/harvest.py --target 520 --out corpus   # 从 360 图搜按 20 类场景抓横构图大图
python tools/normalize.py corpus                    # 统一成 420 宽基线 JPEG（Node 才能全量解码）
node tools/eval.mjs --sets baseline,strict,loose    # 多组参数在同一批图上的量化对比
```

`eval.mjs` 用 Node 里的 `jpeg-js` 解码，直接调与浏览器完全相同的 `analyzePixels`，统计地平线截断率、天空识别失效率、高度饱和率、低谷率等指标。当前默认阈值来自 503 张语料上 7 组参数的扫描结果（score 0.704 vs 基线 0.645）。

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
js/renderer.js     渲染管线 + 后期 + 曝光探针
js/player.js       第一人称控制
js/audio.js        程序化风声
js/app.js          画廊 / 载入 / 主循环 / HUD
models/            Depth-Anything-V2-Small 量化 ONNX（27MB）
vendor/ort/        onnxruntime-web（WASM 单线程）
worlds/            内置场景图
tools/             语料采集 / 归一化 / 批量评测 / CDP 截图
```

## 已知限制

- 单目深度是**相对深度**，尺度靠经验映射（近端 ~2.5m、远端 ~250m），不是真实测距。
- 深度不连续处（前景遮挡背景）做了"前拉"处理，边缘偶有轻微拉伸。
- 360° 背向画面靠镜像平铺补全，会出现重复的山形。
- 移动范围限制在半径 26m（`walkRadius`），再远就离开了照片能覆盖的信息。
- 细密植被（草叶、树枝）深度不精确，近景可能略显"纸片感"。
