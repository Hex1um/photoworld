/* 生物群系原型（由 tools/train_live_params.mjs 从图集训练，勿手改）
 *
 * 训练集：_norm · 1268 张（train 1014 / val 254）
 * 验证：颜色 MAE 0.0459（无原型基线 0.1017，改善 54.9%）
 *
 * 用途：实时变化模式里「混乱度」把当前扇区往某套原型推，
 *      于是新生成的地貌与照片无关，但仍然像真实世界的某个生物群系。
 * 字段：tint 线性 RGB 地面色 / amp 起伏倍率 / rough 频率倍率 / weight 语料占比
 */
(function (global) {
  'use strict';
  var DATA = {
    "version": "1.18.0",
    "trainedAt": "2026-11-14T05:34:43.067Z",
    "report": {
      "generatedAt": "2026-11-14T05:34:43.067Z",
      "corpus": "_norm",
      "samples": 1268,
      "train": 1014,
      "val": 254,
      "k": 6,
      "valColorRmse": 0.3006,
      "valColorMae": 0.0459,
      "baselineMae": 0.1017,
      "improvement": 0.549
    },
    "prototypes": [
      {
        "tint": [
          0.0946,
          0.0919,
          0.074
        ],
        "amp": 1.024,
        "rough": 1.27,
        "weight": 0.2643,
        "n": 268
      },
      {
        "tint": [
          0.2773,
          0.3347,
          0.3357
        ],
        "amp": 1.265,
        "rough": 1.495,
        "weight": 0.069,
        "n": 70
      },
      {
        "tint": [
          0.1231,
          0.1335,
          0.1028
        ],
        "amp": 1.291,
        "rough": 1.528,
        "weight": 0.2357,
        "n": 239
      },
      {
        "tint": [
          0.061,
          0.0589,
          0.0516
        ],
        "amp": 0.82,
        "rough": 0.984,
        "weight": 0.2456,
        "n": 249
      },
      {
        "tint": [
          0.5363,
          0.5131,
          0.4581
        ],
        "amp": 0.89,
        "rough": 1.101,
        "weight": 0.0424,
        "n": 43
      },
      {
        "tint": [
          0.2816,
          0.2279,
          0.1145
        ],
        "amp": 0.961,
        "rough": 1.203,
        "weight": 0.143,
        "n": 145
      }
    ]
  };
  global.PW = global.PW || {};
  global.PW.liveParams = DATA;
})(window);
