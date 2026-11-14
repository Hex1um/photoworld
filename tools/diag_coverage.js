/* 质检表达式：统计 relief 网格各区域「有四边形」的比例，用来定位画面里
 * 某块区域为什么是空的（顶点在、但索引被裁掉了？还是根本没几何？）。
 * 用法：node tools/probe.mjs --url "...&dbgpos=1" --exprfile tools/diag_coverage.js
 */
(function () {
  var d = window.__pw.world.dbgPos;
  if (!d) return JSON.stringify({ err: 'no dbgPos' });
  var GW = d.GW, GH = d.GH, idx = d.idx;
  var BX = 8, BY = 6;                       // 8×6 分区
  var cell = [];
  for (var c = 0; c < BX * BY; c++) cell.push({ quad: 0, total: 0 });
  for (var t = 0; t < idx.length; t += 6) {
    var a0 = idx[t];
    var i = a0 % GW, j = (a0 - i) / GW;
    var bu = Math.min(BX - 1, Math.floor(i / GW * BX));
    var bv = Math.min(BY - 1, Math.floor(j / GH * BY));
    cell[bv * BX + bu].quad++;
  }
  for (var j2 = 0; j2 < GH; j2++) for (var i2 = 0; i2 < GW; i2++) {
    var bu2 = Math.min(BX - 1, Math.floor(i2 / GW * BX));
    var bv2 = Math.min(BY - 1, Math.floor(j2 / GH * BY));
    cell[bv2 * BX + bu2].total++;
  }
  var rows = [];
  for (var bv3 = 0; bv3 < BY; bv3++) {
    var line = [];
    for (var bu3 = 0; bu3 < BX; bu3++) {
      var cc = cell[bv3 * BX + bu3];
      line.push((cc.quad / Math.max(1, cc.total)).toFixed(2));
    }
    rows.push(line.join(' '));
  }
  return JSON.stringify({ grid: rows });
})()
