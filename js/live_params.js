/* 生物群系原型（由 tools/train_live_params.mjs 从图集训练，勿手改）
 *
 * 训练集：_norm · 2069 张（train 1655 / val 414）
 * 验证：颜色 MAE 0.0415（无原型基线 0.1058，改善 60.8%）
 *
 * 用途：实时变化模式里「混乱度」把当前扇区往某套原型推，
 *      于是新生成的地貌与照片无关，但仍然像真实世界的某个生物群系。
 * 字段：tint 线性 RGB 地面色 / amp 起伏倍率 / rough 频率倍率 / weight 语料占比
 */
(function (global) {
  'use strict';
  var DATA = {
    "version": "1.18.0",
    "trainedAt": "2026-11-14T07:17:27.304Z",
    "report": {
      "generatedAt": "2026-11-14T07:17:27.304Z",
      "corpus": "_norm",
      "samples": 2069,
      "train": 1655,
      "val": 414,
      "k": 6,
      "valColorRmse": 0.286,
      "valColorMae": 0.0415,
      "baselineMae": 0.1058,
      "improvement": 0.608
    },
    "prototypes": [
      {
        "tint": [
          0.0314,
          0.0341,
          0.0367
        ],
        "amp": 0.746,
        "rough": 0.879,
        "weight": 0.229,
        "n": 379
      },
      {
        "tint": [
          0.2698,
          0.3028,
          0.2945
        ],
        "amp": 1.34,
        "rough": 1.531,
        "weight": 0.0743,
        "n": 123
      },
      {
        "tint": [
          0.1196,
          0.1153,
          0.0902
        ],
        "amp": 1.24,
        "rough": 1.483,
        "weight": 0.2181,
        "n": 361
      },
      {
        "tint": [
          0.5511,
          0.5548,
          0.5409
        ],
        "amp": 0.886,
        "rough": 1.066,
        "weight": 0.0477,
        "n": 79
      },
      {
        "tint": [
          0.069,
          0.0663,
          0.0579
        ],
        "amp": 0.98,
        "rough": 1.167,
        "weight": 0.3142,
        "n": 520
      },
      {
        "tint": [
          0.2624,
          0.2194,
          0.1511
        ],
        "amp": 0.942,
        "rough": 1.16,
        "weight": 0.1166,
        "n": 193
      }
    ]
  };
  global.PW = global.PW || {};
  global.PW.liveParams = DATA;
})(window);
