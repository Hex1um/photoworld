"""COLMAP 接入：用 COLMAP 求位姿，我们只做后面的融合/网格/渲染。

为什么必须接（实测结论）：手写增量 SfM 在真实环绕素材上会把相机解成
半径 10.3/13.0/8.5（真值恒为 4.03），RMSE 却只有 0.86px —— 位姿整体错、自洽性却很好。
COLMAP 有完整的 bundle adjustment，这才是"照片出世界"的正确起点。

约定对齐（这里很容易错，写清楚）：
  * COLMAP `images.txt` 的 `QW QX QY QZ TX TY TZ` 就是 **world→camera** 的旋转与平移；
  * 相机坐标 = x 右、y 下、**z 前**，与 cv2 一致，也正是本项目 `_project` 用的约定（z = Xc[2]）；
  * `cameras.txt` 给出 fx/fy/cx/cy（甚至畸变），比我们猜的 `f = 1.2·max(w,h)` 准得多；
  * `points3D.txt` 直接给稀疏点 + 颜色，省掉自己三角化。

用法：
    from v4 import colmap
    if colmap.available():
        model = colmap.run(img_dir, work_dir)     # 跑三步 CLI
        cams, poses, pts, cols = colmap.load_model(model / "sparse" / "0")
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def find_binary() -> str | None:
    """找 colmap 可执行文件：环境变量 → 仓库内 tools/colmap → PATH。"""
    env = os.environ.get("COLMAP_EXE")
    if env and Path(env).exists():
        return env
    for p in (ROOT / "tools" / "colmap" / "COLMAP.bat",
              ROOT / "tools" / "colmap" / "bin" / "colmap.exe",
              ROOT / "tools" / "colmap" / "colmap.exe",
              ROOT / "third_party" / "colmap" / "bin" / "colmap.exe",
              ROOT / "third_party" / "colmap" / "colmap.exe"):
        if p.exists():
            return str(p)
    return shutil.which("colmap")


def available() -> bool:
    return find_binary() is not None


def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def run_dense(img_dir: str, work: str, model: Path, quiet: bool = False) -> Path | None:
    """COLMAP 稠密重建：去畸变 → patch_match_stereo → stereo_fusion。

    为什么值得多花这两分钟：我们自己的"稠密点"是**单目深度反投影**（猜的），
    而 patch_match 是**真多视图立体匹配**——同一块表面被多张照片看到才算数。
    输出 `fused.ply` 可直接当世界点云。
    """
    exe = find_binary()
    if not exe:
        return None
    base = [exe] if exe.lower().endswith(".exe") else ["cmd", "/c", exe]
    w = Path(work)
    dense = w / "dense"
    dense.mkdir(parents=True, exist_ok=True)
    log = w / "colmap.log"
    steps = [
        ("去畸变", base + ["image_undistorter", "--image_path", str(Path(img_dir).resolve()),
                        "--input_path", str(model), "--output_path", str(dense),
                        "--output_type", "COLMAP"]),
        ("稠密匹配", base + ["patch_match_stereo", "--workspace_path", str(dense),
                         "--workspace_format", "COLMAP",
                         "--PatchMatchStereo.geom_consistency", "true"]),
        ("融合", base + ["stereo_fusion", "--workspace_path", str(dense),
                       "--workspace_format", "COLMAP",
                       "--input_type", "geometric",
                       "--output_path", str(dense / "fused.ply")]),
    ]
    for name, cmd in steps:
        if not quiet:
            print("      COLMAP %s…" % name)
        if _run(cmd, log) != 0:
            print("      COLMAP %s 失败（继续用稀疏点）" % name)
            return None
    p = dense / "fused.ply"
    return p if p.exists() and p.stat().st_size > 1024 else None


def read_ply(path: str | Path):
    """读 binary_little_endian PLY（COLMAP fused.ply 用的就是这种）。"""
    import numpy as np
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
            if len(header) > 200:
                raise SystemExit("PLY 头异常：%s" % path)
        n_vert = 0
        props = []
        fmt = "binary_little_endian"
        for h in header:
            if h.startswith("element vertex"):
                n_vert = int(h.split()[-1])
            elif h.startswith("property"):
                parts = h.split()
                props.append((parts[-1], parts[1]))
            elif h.startswith("format"):
                fmt = h.split()[1]
        if fmt != "binary_little_endian":
            raise SystemExit("只支持 binary_little_endian PLY（%s）" % fmt)
        dtype = []
        for name, typ in props:
            m = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1",
                 "uint8": "u1", "int": "i4", "uint": "u4", "short": "i2", "ushort": "u2"}
            dtype.append((name, m.get(typ, "f4")))
        arr = np.frombuffer(f.read(n_vert * np.dtype(dtype).itemsize), dtype=np.dtype(dtype))
    names = [n for n, _ in props]
    P = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    N = None
    if all(c in names for c in ("nx", "ny", "nz")):
        # fused.ply 自带法线（MVS 估的，方向一致）——泊松重建靠它，别自己再猜
        N = np.stack([arr["nx"], arr["ny"], arr["nz"]], axis=1).astype(np.float32)
    if all(c in names for c in ("red", "green", "blue")):
        C = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        C = np.full((len(P), 3), 0.6, np.float32)
    return (P, C) if N is None else (P, C, N)


def run(img_dir: str, work: str, matcher: str = "exhaustive", quiet: bool = False) -> Path:
    """feature_extractor → matcher → mapper。返回 sparse 模型目录。"""
    exe = find_binary()
    if not exe:
        raise SystemExit("找不到 colmap 可执行文件（设 COLMAP_EXE 或放到 tools/colmap/）")
    w = Path(work)
    db = w / "database.db"
    sparse = w / "sparse"
    w.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)
    log = w / "colmap.log"
    base = [exe] if exe.lower().endswith(".exe") else ["cmd", "/c", exe]

    steps = [
        ("特征提取", base + ["feature_extractor", "--database_path", str(db),
                          "--image_path", str(Path(img_dir).resolve()),
                          "--ImageReader.single_camera", "1",
                          "--SiftExtraction.use_gpu", "1"]),
        ("匹配", base + ["exhaustive_matcher", "--database_path", str(db),
                        "--SiftMatching.use_gpu", "1"] if matcher == "exhaustive" else
                 base + ["sequential_matcher", "--database_path", str(db),
                         "--SiftMatching.use_gpu", "1"]),
        ("重建", base + ["mapper", "--database_path", str(db),
                        "--image_path", str(Path(img_dir).resolve()),
                        "--output_path", str(sparse)]),
    ]
    for name, cmd in steps:
        if not quiet:
            print("      COLMAP %s…" % name)
        rc = _run(cmd, log)
        if rc != 0:
            raise SystemExit("COLMAP %s 失败（退出码 %d），日志：%s" % (name, rc, log))

    models = sorted([p for p in sparse.iterdir() if p.is_dir()])
    if not models:
        raise SystemExit("COLMAP 没产出模型（可能是重叠不足），日志：%s" % log)
    # 取点数最多的那个模型
    best, best_n = models[0], -1
    for m in models:
        f = m / "points3D.bin"
        n = f.stat().st_size if f.exists() else 0
        if n > best_n:
            best, best_n = m, n
    return best


def _read_cameras(path: Path) -> dict[int, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cid, model, W, H = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = params[0]; cx, cy = params[1], params[2]
            elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            else:
                fx = fy = params[0]; cx, cy = params[1], params[2]
            out[cid] = {"w": W, "h": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "model": model}
    return out


def _read_images(path: Path) -> dict[str, tuple]:
    """COLMAP images.txt：2 行一条（第二行是 2D 观测，跳过）。返回 name → (R, t)。"""
    out = {}
    import numpy as np
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if not l.startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        name = " ".join(parts[9:])
        # 四元数（w,x,y,z）→ R
        n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]],
            dtype=np.float64)
        out[Path(name).name] = (R, np.array([[tx], [ty], [tz]], np.float64))
        i += 1        # 跳过 2D 观测行
    return out


def _read_points(path: Path):
    import numpy as np
    P, C = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            P.append([float(parts[1]), float(parts[2]), float(parts[3])])
            C.append([float(parts[4]) / 255.0, float(parts[5]) / 255.0, float(parts[6]) / 255.0])
    if not P:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
    return np.array(P, np.float32), np.array(C, np.float32)


def load_model(model_dir: str | Path, frames):
    """读 COLMAP 模型，组装成 v4 的 res 结构（cams / poses / pts / colors）。"""
    import numpy as np
    md = Path(model_dir)
    # 优先读 TXT；没有就用 model_converter 转
    if not (md / "cameras.txt").exists():
        exe = find_binary()
        base = [exe] if exe and exe.lower().endswith(".exe") else ["cmd", "/c", exe or "colmap"]
        txt_dir = md.parent / (md.name + "_txt")
        txt_dir.mkdir(parents=True, exist_ok=True)
        _run(base + ["model_converter", "--input_path", str(md),
                     "--output_path", str(txt_dir), "--output_type", "TXT"],
             md.parent / "colmap.log")
        md = txt_dir

    cams_txt = _read_cameras(md / "cameras.txt")
    imgs = _read_images(md / "images.txt")

    class Cam:
        def __init__(self, w, h, K):
            self.w, self.h, self.K = w, h, K

    cams, poses = [], []
    hit = 0
    for f in frames:
        key = Path(f.name).name
        cam_any = next(iter(cams_txt.values()))
        p = imgs.get(key)
        if p is None:
            cams.append(Cam(cam_any["w"], cam_any["h"],
                            np.array([[cam_any["fx"], 0, cam_any["cx"]],
                                      [0, cam_any["fy"], cam_any["cy"]], [0, 0, 1]], np.float64)))
            poses.append(None)
            continue
        hit += 1
        # 每帧的实际分辨率可能和 COLMAP 记录的不同（我们做过缩放），按比例改内参
        sx = f.rgb.shape[1] / cam_any["w"]
        sy = f.rgb.shape[0] / cam_any["h"]
        K = np.array([[cam_any["fx"] * sx, 0, cam_any["cx"] * sx],
                      [0, cam_any["fy"] * sy, cam_any["cy"] * sy], [0, 0, 1]], np.float64)
        cams.append(Cam(f.rgb.shape[1], f.rgb.shape[0], K))
        poses.append(p)

    pts, cols = _read_points(md / "points3D.txt")
    print("      COLMAP 模型：%d 帧命中 / %d，内参 %.0f px，稀疏点 %d"
          % (hit, len(frames), cams[0].K[0, 0] if cams else 0, len(pts)))
    return cams, poses, pts, cols
