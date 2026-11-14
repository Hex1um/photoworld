/* 质检表达式：统计 relief 网格里「可见顶点」之间相邻的竖直跳变，
 * 定位「竖条 / 鳍 / 竖幕」的成因。
 * 注意：必须过滤 alpha（天空区域被剔除，y 值无意义，不过滤会得到大量误报）。
 * 用法：node tools/probe.mjs --url "...&dbgpos=1" --exprfile tools/diag_jumps.js
 */
(function () {
  var d = window.__pw.world.dbgPos;
  if (!d) return JSON.stringify({ err: 'no dbgPos (?dbgpos=1 未开)' });
  var GW = d.GW, GH = d.GH, pos = d.pos, gT = d.gT, alpha = d.alpha;
  var vis = function (k) { return alpha ? alpha[k] >= 0.06 : true; };
  var bad = [];
  var i, j;
  for (j = 0; j < GH; j++) {
    for (i = 0; i < GW - 1; i++) {
      var a = j * GW + i, b = a + 1;
      if (!vis(a) || !vis(b)) continue;
      var dy = Math.abs(pos[a * 3 + 1] - pos[b * 3 + 1]);
      if (dy > 6) bad.push({ dir: 'H', i: i, j: j, dy: +dy.toFixed(1), y0: +pos[a * 3 + 1].toFixed(1), y1: +pos[b * 3 + 1].toFixed(1), t0: +gT[a].toFixed(1), t1: +gT[b].toFixed(1), z0: +pos[a * 3 + 2].toFixed(1), z1: +pos[b * 3 + 2].toFixed(1) });
    }
  }
  for (j = 0; j < GH - 1; j++) {
    for (i = 0; i < GW; i++) {
      var a2 = j * GW + i, b2 = a2 + GW;
      if (!vis(a2) || !vis(b2)) continue;
      var dy2 = Math.abs(pos[a2 * 3 + 1] - pos[b2 * 3 + 1]);
      if (dy2 > 6) bad.push({ dir: 'V', i: i, j: j, dy: +dy2.toFixed(1), y0: +pos[a2 * 3 + 1].toFixed(1), y1: +pos[b2 * 3 + 1].toFixed(1), t0: +gT[a2].toFixed(1), t1: +gT[b2].toFixed(1), z0: +pos[a2 * 3 + 2].toFixed(1), z1: +pos[b2 * 3 + 2].toFixed(1) });
    }
  }
  bad.sort(function (p, q) { return q.dy - p.dy; });
  var hist = {};
  for (var k = 0; k < bad.length; k++) {
    var dk = Math.min(200, Math.floor(bad[k].dy / 10) * 10);
    hist[dk] = (hist[dk] || 0) + 1;
  }
  var rowHist = {};
  for (var m = 0; m < bad.length; m++) {
    var vv = Math.round(bad[m].j / (GH - 1) * 20) / 20;
    rowHist[vv] = (rowHist[vv] || 0) + 1;
  }
  var visCount = 0;
  for (var q2 = 0; q2 < GW * GH; q2++) if (vis(q2)) visCount++;
  return JSON.stringify({ visible: visCount, total: bad.length, histY: hist, histRow: rowHist, top: bad.slice(0, 6) });
})()
