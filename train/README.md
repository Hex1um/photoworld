# 历史实验资产（deprecated）

本目录是**早期本地小网络**的训练与推理残留，与当前主线无关，请按存档看待：

- 当前前端深度估计走 `js/depth.js` + `models/model_quantized.onnx`
  （Depth-Anything-V2-Small，浏览器内 ONNX Runtime WASM 推理）。
- 当前世界模型训练走 `worldmodel/`（Causal World DiT + rectified flow），
  数据由 `scripts/prepare_data.py` 打成 tar 分片。
- 全仓库检索确认：`train/` 下的文件**没有任何运行时引用**，`index.html` 不加载它们，
  构建与部署也不需要它们。

目录内容：

| 文件 | 说明 |
| --- | --- |
| `gen.py` / `train.py` | 早期小网络的样本生成与训练脚本 |
| `predict.py` / `dump.py` | 推理与权重导出 |
| `weights.json`（约 964 KB） | 导出的权重，运行时不再读取 |
| `in.f32` / `out_c.f32` / `out_h.f32` | 中间张量快照 |
| `preview/` | 当时的预览输出 |

保留它们只是为了让当时的实验可从 git 历史复现；若需要减小仓库体积，
可直接删除本目录（不影响任何功能）。新增代码请勿依赖这里的文件。
