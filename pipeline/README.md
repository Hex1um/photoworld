# 重建管线（离线）

把照片在本地重建为可加载的三维资产，浏览器只负责渲染。

```
照片 ──► 深度 ──► 尺度标定 ──► 点云 ──► 网格 ──► world.glb (+ world.json)
                    ▲
              地面平面约束
```

## 快速开始（CPU，无需 GPU）

```bash
python -m pip install onnxruntime numpy pillow
python pipeline/reconstruct.py worlds/alpine.jpg --out out/alpine
# → out/alpine/world.glb   out/alpine/world.json
```

约 3 秒 / 5.8MB（1536×968 输入，20 万三角）。自检会打印覆盖率、PSNR、SSIM。

## 两种后端

| 后端 | 深度类型 | 依赖 | 尺度 | 状态 |
|---|---|---|---|---|
| `--backend onnx` | 相对视差 | onnxruntime（CPU） | 由地面平面幂律拟合反推 | ✅ 已验证 |
| `--backend depthpro` | **度量（米）** | torch + ml-depth-pro 权重 | 直接是米 | 代码就绪，待装权重 |
| `--backend vggt` | 多视图 | torch + VGGT-1B | 由基线确定 | 代码就绪，待装权重 |

`environment.yml` 是重后端的锁定版本清单。

**重要**：相对深度撑不起真实尺度。ONNX 后端虽然用地面平面把近景钉到了 4.6m，
但中远景仍会被压成"沙盘"（山体只有 ~10m 远）。要满足"无微缩感"，必须换度量深度（Depth Pro）。

## 关键设计

1. **等比处理**：照片先等比缩放再 padding 到 518×518 推理，绝不拉伸；深度再裁回原比例。
2. **尺度来自几何**：地平线以下的像素都落在同一水平面上，距离 `z = eye / sinθ` 精确已知。
   用整条地面带拟合 `z = A·r^(-p)`（稳健两轮剔除），参数完全由这张图决定，不是经验曲线。
3. **视场外延拓**：横向延伸到 120°，按边界列深度连续外推 + 轻度雾化，**不做镜像平铺**，
   所以不会出现重复山形与接缝条纹。
4. **锐利轮廓**：跨深度断崖（比值 > `--edge-ratio`）的四边形直接跳过，不做跨边缘平滑、
   不做"前拉"补洞，所以树梢/山脊不会糊成一片。
5. **自检**：用泼点渲染回到原相机，与输入照片逐像素比对（覆盖率 / PSNR / SSIM），
   任何尺度、视场、等比处理上的错误都会让它掉分。

## 产物

`world.glb` — glTF 2.0，单 mesh + 内嵌 JPEG 贴图，PBR 材质，可直接被 three.js / 原生 WebGL 加载。
坐标系为**相机系**：x 右、y 上、z 向前（-Z 为视线方向），原点在拍摄点，单位米。

`world.json` — 相机内参、尺度标定参数、包围盒、网格统计、自检指标，以及
`walkable.height`（60m×60m 的 64×64 表面高度网格，供前端碰撞用）。

## 参数

```
--backend onnx|depthpro|vggt   深度后端
--eye 1.7                      相机高度（米），尺度标定的唯一人工输入
--fov 0                        覆盖 EXIF 的水平视场角（度），0=自动
--target-fov 120               延拓到的水平视场角
--max-tris 480000              三角上限（超出自动降采样）
--edge-ratio 1.6               多大深度比视为断崖
--tex-max 2048                 贴图长边上限
```
