#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小页面验证 FS_MAIN 的 mode 10 分支能否在 WebGL2 编译通过。

ISSUES #3 的纪律：mode 10 曾因 undeclared identifier 编译失败（整站白屏），
不要再直接改主 renderer 试错 —— 这里起一个 only-compile 的最小页面，
把 glutil.js 里的真实 VS_MAIN/FS_MAIN 抽出来（含候选 mode 10 分支）编译，
读 gl.getShaderInfoLog / getProgramInfoLog 判定。

    python tools/test_shader_mode10_compile.py

退出码 0 = 编译+链接全通过；1 = 有 GLSL 错误（打印日志）。
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cdp  # noqa: E402


def extract_template(name: str) -> str:
    """从 glutil.js 里抽 `const NAME = `后跟的反引号模板字符串。"""
    src = open(os.path.join(ROOT, "js", "glutil.js"), encoding="utf-8").read()
    m = re.search(r"const\s+%s\s*=\s*`(.*?)`\s*;" % re.escape(name), src, re.S)
    if not m:
        raise RuntimeError("找不到 shader 模板：" + name)
    return m.group(1)


def main() -> int:
    vs = extract_template("VS_MAIN")
    fs10 = extract_template("FS_MAIN")   # 直接编译 glutil.js 里的真实源码
    if "mode == 10" not in fs10:
        raise RuntimeError("FS_MAIN 里没有 mode 10 分支（尚未接入 glutil.js？）")

    html = """<!doctype html><html><body><canvas id=c></canvas><script>
window.__result = null;
try {
  var gl = document.getElementById('c').getContext('webgl2');
  if (!gl) { window.__result = {ok:false, why:'no webgl2'}; }
  else {
    var VS = %s, FS = %s;
    function comp(type, src){
      var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      return {ok: gl.getShaderParameter(s, gl.COMPILE_STATUS), log: gl.getShaderInfoLog(s), sh:s};
    }
    var v = comp(gl.VERTEX_SHADER, VS), f = comp(gl.FRAGMENT_SHADER, FS);
    var linkLog = '', linkOk = false;
    if (v.ok && f.ok) {
      var p = gl.createProgram(); gl.attachShader(p, v.sh); gl.attachShader(p, f.sh);
      gl.linkProgram(p); linkOk = gl.getProgramParameter(p, gl.LINK_STATUS);
      linkLog = gl.getProgramInfoLog(p);
    }
    window.__result = {ok: v.ok && f.ok && linkOk,
      vsLog: v.log, fsLog: f.log, linkLog: linkLog,
      vsOk: v.ok, fsOk: f.ok, linkOk: linkOk};
  }
} catch(e) { window.__result = {ok:false, why: String(e)}; }
document.body.dataset.ready = '1';
</script></body></html>""" % (json.dumps(vs), json.dumps(fs10))

    tmp_html = os.path.join(ROOT, "out", "_shader_mode10_test.html")
    os.makedirs(os.path.dirname(tmp_html), exist_ok=True)
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(html)

    b = cdp.launch(headless=True, gpu="swiftshader", no_proxy=True)
    try:
        c = b.cdp
        cdp.enable_common(c, 640, 480, metrics=False)
        c.send("Page.navigate", {"url": "file:///" + tmp_html.replace("\\", "/")})
        import time
        for _ in range(60):
            try:
                raw = c.ev("JSON.stringify(window.__result || null)")
                if raw and raw != "null":
                    res = json.loads(raw)
                    break
            except Exception:
                pass
            time.sleep(0.25)
        else:
            print("FAIL  等待编译结果超时")
            return 1
    finally:
        b.close()
        try:
            os.remove(tmp_html)
        except OSError:
            pass

    if res.get("ok"):
        print("PASS  FS_MAIN + mode 10 分支：WebGL2 编译 + 链接全通过")
        return 0
    print("FAIL  GLSL 编译/链接失败：")
    print("  vsOk=%s fsOk=%s linkOk=%s why=%s" %
          (res.get("vsOk"), res.get("fsOk"), res.get("linkOk"), res.get("why")))
    for k in ("vsLog", "fsLog", "linkLog"):
        if res.get(k):
            print("  %s: %s" % (k, res[k].strip()))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
