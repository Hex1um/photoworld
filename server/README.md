# 本地生成服务（可选）

前端默认是纯静态、零构建、可离线运行的：没有本服务时，扇环延伸走浏览器内的
程序化 fBm，单体走账单牌。本服务只负责两件浏览器做不动的事：

| 接口 | 作用 | 后端 |
|---|---|---|
| `POST /api/extend` | 扇环外 tile 地貌延伸 | Terrain Diffusion（可选）/ 程序化边界约束（默认） |
| `POST /api/object3d` | 近景单体单图转 3D | InstantMesh / TripoSR（可选） |
| `GET /health` | 后端探测 | — |

## 启动

```bash
cd server
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8100
# 另开一个终端起静态站
cd ..
python -m http.server 8099 --bind 127.0.0.1
```

前端用法：

```
http://127.0.0.1:8099/?api=http://127.0.0.1:8100          # 只开地貌延伸
http://127.0.0.1:8099/?api=http://127.0.0.1:8100&obj3d=1 # 同时开启单体真网格
http://127.0.0.1:8099/?ext=0                              # 关闭延伸（对照）
```

> 跨域已开 `allow_origins=["*"]`，本地开发无需额外配置。

## 接口约定

### POST /api/extend

请求：

```json
{
  "seed": 12345,
  "tx": 0, "tz": -4,
  "ts": 32,
  "res": 17,
  "edges": {
    "left":  [17 个数], "right": [17 个数],
    "top":   [17 个数], "bottom":[17 个数]
  },
  "learned": true
}
```

响应：

```json
{ "heights": [289 个数], "res": 17, "source": "procedural", "tx": 0, "tz": -4 }
```

`edges` 是 tile 四边的高度采样（来自已重建地貌或相邻已生成 tile）。
服务端生成结果**在四条边界上与输入严格一致**（内部已用带状混合过渡），
因此相邻 tile 无接缝。`source` 为 `terrain_diffusion` 或 `procedural`。

### POST /api/object3d

`multipart/form-data`：`image` = 单体裁片 PNG，`bbox` = 可选 `"x,y,w,h"`（归一化）。
成功返回 `model/gltf-binary`；未安装后端返回 `501`，前端保留账单牌。

## 接入高质量的 learned 后端

### 1) Terrain Diffusion（无限地形，SIGGRAPH 2026）

```bash
git clone https://github.com/xandergos/terrain-diffusion
cd terrain-diffusion
pip install -r requirements.txt        # 需要 CUDA 版 torch
```

然后确认 `terrain.py::terrain_diffusion_extend` 里的 `WorldPipeline` 导入可用
（该仓库 API 随版本变化，若签名不同，改这一处即可）。`/health` 的
`terrain_diffusion` 会变为 `true`。

### 2) InstantMesh / TripoSR（单体转 3D）

```bash
# InstantMesh（质量优先）
git clone https://github.com/TencentARC/InstantMesh
pip install -r InstantMesh/requirements.txt
# TripoSR（速度优先）
git clone https://github.com/VAST-AI-Research/TripoSR
pip install -r TripoSR/requirements.txt
```

把权重放到对应仓库的 `ckpts/`，并在 `object3d.py::generate` 里接上前向与 GLB 导出
（目前只做了后端探测与裁片，推理函数留了明确的接入点，避免把未验证的代码伪装成可用）。

## 为什么不做成 Oasis 式的浏览器实时生成

Oasis 3 是 512×768×3 @22FPS、<200ms 延迟，跑在 B200 集群 + 自研 DOS 优化栈上。
浏览器 WebGPU 只有在本模型**大幅缩水**（256×384、1~4 步蒸馏、int8 量化）时
才可能跑到 3~8FPS，质量明显下降。所以本项目采用：

- **基础重建**：浏览器本地（已有，照片不上传）
- **生成式延伸**：本地 GPU 服务（本目录）
- **可选降级**：无服务时浏览器内程序化延伸
