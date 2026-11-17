"""PHOTOWORLD 本地生成服务（可选）

职责：为前端提供两件浏览器做不动的事——
  1) POST /api/extend    扇环外地貌延伸（Terrain Diffusion / 程序化边界约束）
  2) POST /api/object3d  单体物体单图转 3D（InstantMesh / TripoSR，可选）

前端保持零构建：不传 ?api= 时完全走浏览器内的程序化延伸，
传 ?api=http://127.0.0.1:8100 时自动升级为服务端生成。

启动：
  cd server
  pip install -r requirements.txt
  uvicorn app:app --host 127.0.0.1 --port 8100
"""
from __future__ import annotations

import os
import threading

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import terrain
import object3d

# v2 重建管线（pipeline/v2）：缺依赖时不能让整个服务起不来
try:
    import sys as _sys, importlib.util as _ilu
    _p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline", "v2")
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
    _spec = _ilu.spec_from_file_location("v2_api", os.path.join(_p, "api.py"))
    _v2 = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_v2)
except Exception:                              # noqa: BLE001
    _v2 = None
# P1（v2.15.8）：CUDA 只有一把，bake 必须串行。
# 旧代码把 bake 直接丢进 run_in_executor：两个请求并发时多线程同时调 CUDA +
# 共享 _model/_cache（api.py 的 LRU 还会 KeyError）。ISSUES 附录 A 记过这个事故
# （ThreadingHTTPServer + CUDA → 非主线程调 CUDA 直接崩、客户端只见连接被关），
# tools/score_server.py 换单线程规避了，但这条主链路一直没有等价保护。
_BAKE_LOCK = threading.Lock()

app = FastAPI(title="photoworld generation service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 本地静态站点，允许任意来源
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtendReq(BaseModel):
    seed: int = 1
    tx: int = 0
    tz: int = 0
    ts: float = 32.0
    res: int = 17
    edges: dict | None = None
    learned: bool = True


@app.get("/health")
def health():
    return {
        "ok": True,
        "terrain_diffusion": _has_terrain_diffusion(),
        "object3d": object3d.backend_name(),
        "gpu": _gpu_name(),
        "reconstruct": _reconstruct_available(),
    }


def _reconstruct_available():
    if _v2 is None:
        return {"ok": False, "reason": "pipeline/v2 未能加载（缺 torch/transformers/scipy？）"}
    try:
        return _v2.available()
    except Exception as e:                     # 缺依赖/缺权重都要能优雅降级
        return {"ok": False, "reason": str(e)}


@app.post("/api/reconstruct")
async def api_reconstruct(image: UploadFile = File(...), opts: str = Form("")):
    """照片 → 分层 GLB（v2 管线，GPU 上跑的离线重建）。

    这是"点一张照片就走进新的世界"的主链路：前端把原图 POST 上来，这里用
    常驻的 ViT-L 深度模型 + LDI 建网格，直接回 GLB 字节。前端拿 blob URL 交给
    现有的资产加载器即可，不需要改渲染路径。
    """
    avail = _reconstruct_available()
    if not avail.get("ok") or _v2 is None:
        return JSONResponse(status_code=501,
                            content={"error": "reconstruct_unavailable",
                                     "hint": avail.get("reason", "")})
    data = await image.read()
    over = {}
    if opts:
        try:
            import json as _json
            over = {k: v for k, v in _json.loads(opts).items() if k in _v2.DEFAULTS}
        except Exception:
            over = {}
    try:
        import asyncio
        def _locked_bake():
            with _BAKE_LOCK:
                return _v2.bake(data, **over)
        glb, manifest = await asyncio.get_running_loop().run_in_executor(
            None, _locked_bake)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "bake_failed", "detail": str(e)})
    return Response(content=glb, media_type="model/gltf-binary",
                    headers={"X-Pipeline": "photoworld-v2",
                             "X-Elapsed": str(manifest.get("elapsed_s", "")),
                             "X-Layers": str(len([k for k in manifest.get("layers", {})
                                                  if not k.startswith("_")]))})


@app.get("/api/reconstruct/info")
async def api_reconstruct_info(manifest: str = ""):
    """调试用：回显上一次/指定 manifest（当前从简，只回能力探测）。"""
    return {"available": _reconstruct_available()}


@app.post("/api/extend")
def api_extend(req: ExtendReq):
    res = max(3, min(129, int(req.res)))
    heights, source = terrain.extend(
        seed=req.seed, tx=req.tx, tz=req.tz, ts=req.ts, res=res,
        edges=req.edges, prefer_learned=req.learned,
    )
    return {
        "heights": heights.reshape(-1).tolist(),
        "res": res,
        "source": source,
        "tx": req.tx, "tz": req.tz,
    }


@app.post("/api/object3d")
async def api_object3d(image: UploadFile = File(...), bbox: str = Form("")):
    if not object3d.available():
        return JSONResponse(
            status_code=501,
            content={"error": "no_backend",
                     "hint": "安装 InstantMesh / TripoSR 后重启；未安装时前端使用账单牌"},
        )
    data = await image.read()
    bb = None
    if bbox:
        try:
            bb = [float(v) for v in bbox.split(",")]
        except Exception:
            bb = None
    glb, reason = object3d.generate(data, bb)
    if glb is None:
        return JSONResponse(status_code=500, content={"error": reason})
    return Response(content=glb, media_type="model/gltf-binary",
                    headers={"X-Backend": object3d.backend_name()})


def _has_terrain_diffusion() -> bool:
    try:
        import terrain_diffusion  # noqa: F401
        return True
    except Exception:
        return False


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("PW_HOST", "127.0.0.1"),
                port=int(os.environ.get("PW_PORT", "8100")))
