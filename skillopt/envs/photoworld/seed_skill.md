# PHOTOWORLD 烘焙参数

你的工作只有一件事：**调整下面这段 yaml 里的数字**，让渲染指标全部达标。
不要写建议、不要加段落、不要解释。所有修改都必须体现为 `params:` 里某个数变了。

```yaml
params:
  far_split: 100
  ground_radius: 400
  seg_edge_tau: 0.10
  skirt_cap: 1.2
  card_flatten: true
```

## 允许的键与范围

| 键 | 范围 | 作用 |
|---|---|---|
| `far_split` | 60 ~ 160 | 近景/远景分界（米）。太小：中景被当远景、无视差；太大：远景参与视差、边缘拉丝 |
| `ground_radius` | 150 ~ 600 | 程序化地面铺多远。太小：转身看到空白；太大：三角面暴涨 |
| `seg_edge_tau` | 0.05 ~ 0.20 | 地物分割灵敏度。细杆/树枝要更灵敏（0.08~0.12）；过大会把建筑切碎（拉丝块数上升） |
| `skirt_cap` | 0.4 ~ 3.0 | 剪影处"厚度"。太大会在近处变成一块醒目色板 |
| `card_flatten` | true / false | 把近处细物压成卡片。放宽会伤到别的区域，一般保持 true |

## 判据（越小越好，全部达标才算通过）

| 指标 | 阈值 | 含义 |
|---|---|---|
| `stripe` | ≤ 2.0 | 拉丝强度 |
| `streakBlobs` | ≤ 200 | 拉丝块数（竖条连通域个数） |
| `innerVoid` | ≤ 0.06 | 中景空洞占比 |
| `voidFrac` | ≤ 0.10 | 露空白占比 |
| `voidBlobs` | ≤ 260 | 空白块数 |
| `voidBlobMax` | ≤ 0.22 | 单块空白最大占比 |
| `voidRunMax` | ≤ 0.50 | 最长横向空段（贯穿缝） |
| `lowVoid` | ≤ 0.08 | 脚下空洞占比 |

## 调整思路（按失败项对症下药）

* `lowVoid` / `voidRunMax` 高（脚下塌、有贯穿缝）→ 先动 `far_split`（分层交界是最常见来源）。
* `streakBlobs` / `stripe` 高（竖条多）→ `seg_edge_tau` 往小调（分割更细），或 `skirt_cap` 往小调。
* `voidFrac` / `voidBlobs` 高（到处是小洞）→ `ground_radius` 往大调，让程序化地面铺更远。
* `voidBlobMax` 很高（一块大洞）→ 通常是某一层没覆盖到，`far_split` 往大调。

## 输出格式

只输出一段 yaml 代码块，内容形如：

```yaml
params:
  far_split: 120
  ground_radius: 320
  seg_edge_tau: 0.08
  skirt_cap: 1.0
  card_flatten: true
```
