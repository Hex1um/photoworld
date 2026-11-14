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

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import terrain
import object3d

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
    }


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
