"""动作编码的唯一来源：训练（prepare_data）与推理（rollout）必须共用本模块。

事故背景
--------
训练标签是 7 维 `[轴角旋转 3, 相机系平移 3 / speed, log(FOV 比值) 1]`，
由 `rel_action(T0, T1)` 从相邻帧真值位姿算出；而旧的 `infer/rollout.py`
手工把 DSL 标量塞进 `a[1] / a[3] / a[5]`，维度、符号、量纲全不对：
`forward` 在训练里是 z 方向平移，推理里却被编码成"绕 y 轴转 + x 平移"的混合。
模型学到的动作语义在推理时被整体替换，动作可控性无从谈起。

现在两侧都调用 `dsl_action()` → `rel_action()`，位姿构造复刻
`scripts/render_traj.mjs::makePoses`，因此符号与耦合自动与合成数据一致。
本模块只依赖 numpy，可在无 torch 环境下完整回归。
"""
from __future__ import annotations

import math

import numpy as np

ACT_DIM = 7

# 与 scripts/render_traj.mjs::makePoses 的默认幅度一致（forward 位移 0.06~0.12，
# 中位约 0.09；yaw 每帧 0.035~0.065；pitch 每帧 0.02）。
DEFAULT_SPEED = 0.09
DEFAULT_MOVE = 0.09
DEFAULT_YAW = 0.05
DEFAULT_PITCH = 0.02
DEFAULT_FOV = 1.2

DSL_NAMES = ("forward", "back", "turn_left", "turn_right",
             "look_up", "look_down", "idle")


def mat4_from_rowmajor(v) -> np.ndarray:
    """poses.json 的 camToWorld 是**列主序**（平移在最后 4 个元素，与 WebGL 一致）。

    必须先 reshape 再转置，否则 T[:3,3] 取到齐次列 [0,0,0]，平移恒为零。
    这个名字是历史遗留，保留以免打断既有调用方。
    """
    return np.array(v, dtype=np.float64).reshape(4, 4).T


def cam_to_world(x: float, y: float, z: float,
                 yaw: float, pitch: float) -> np.ndarray:
    """复刻 render_traj.mjs::trsArray（列主序 16 元素 → 行主序 4×4 矩阵）。"""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    colmajor = np.array([
        cy, 0.0, sy, 0.0,
        sy * sp, cp, -cy * sp, 0.0,
        -sy * cp, sp, cy * cp, 0.0,
        x, y, z, 1.0,
    ], dtype=np.float64)
    return colmajor.reshape(4, 4).T


def rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 轴角（Rodrigues），数值稳健版。"""
    tr = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    ang = math.acos(tr)
    if ang < 1e-6:
        return np.zeros(3, dtype=np.float32)
    if abs(math.pi - ang) < 1e-4:
        A = (R + np.eye(3)) * 0.5
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        k = int(np.argmax(axis))
        if axis[k] < 1e-8:
            return np.zeros(3, dtype=np.float32)
        axis = A[:, k] / axis[k]
        return (axis / np.linalg.norm(axis) * ang).astype(np.float32)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
                    dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (axis / n * ang).astype(np.float32)


def rel_action(T0: np.ndarray, T1: np.ndarray, fov0: float, fov1: float,
               speed: float) -> np.ndarray:
    """相邻帧的相对动作：7 维 = 轴角 3 + 相机系平移 3（按速度归一化）+ log FOV 1。"""
    R0, R1 = T0[:3, :3], T1[:3, :3]
    t0, t1 = T0[:3, 3], T1[:3, 3]
    Rrel = R0.T @ R1
    trel = R0.T @ (t1 - t0)
    aa = rot_to_axis_angle(Rrel)
    scale = max(speed, 1e-6)
    return np.concatenate([
        aa,
        (trel / scale).astype(np.float32),
        np.array([math.log(max(fov1, 1e-3) / max(fov0, 1e-3))], dtype=np.float32),
    ]).astype(np.float32)


def apply_action_scale(vec, action_scale):
    """T6（v1.22）：把 dataset 侧的 action_scale 施加到 7 维动作上（逐维同序）。

    抽成纯 numpy 函数的目的：让"训推幅度一致"这条能在**没有 torch** 的环境里被
    回归测试钉住（旧实现这段逻辑只存在于 rollout 里，测试一缺 torch 就整段跳过）。
    """
    import numpy as np
    v = np.asarray(vec, dtype=np.float32)
    if action_scale is None:
        return v
    sc = np.asarray(action_scale, dtype=np.float32).reshape(-1)
    if sc.size == 1:
        return v * float(sc[0])
    if sc.size != v.size:
        raise ValueError("action_scale 必须是标量或长度 %d 的向量" % v.size)
    return v * sc


def dsl_action(name: str, *, speed: float = DEFAULT_SPEED,
               move_step: float = DEFAULT_MOVE, yaw_step: float = DEFAULT_YAW,
               pitch_step: float = DEFAULT_PITCH,
               fov: float = DEFAULT_FOV) -> np.ndarray:
    """一个 DSL 动作 → 与训练标签同构的 7 维向量。

    做法不是手填标量，而是构造与 render_traj.mjs 相同的相邻位姿，再走 rel_action。
    这样旋转/平移的符号、轴序、归一化量纲与训练数据严格一致。
    """
    name = str(name).strip().lower()
    if name not in DSL_NAMES:
        raise ValueError("未知动作 %r，可用：%s" % (name, ", ".join(DSL_NAMES)))
    x, y, z, yaw, pitch = 0.0, 1.7, 0.0, 0.0, 0.0
    nx, ny, nz, nyaw, npitch = x, y, z, yaw, pitch
    if name == "forward":
        nx -= math.sin(yaw) * move_step
        nz -= math.cos(yaw) * move_step
    elif name == "back":
        nx += math.sin(yaw) * move_step
        nz += math.cos(yaw) * move_step
    elif name == "turn_left":
        nyaw += yaw_step
    elif name == "turn_right":
        nyaw -= yaw_step
    elif name == "look_up":
        npitch += pitch_step
    elif name == "look_down":
        npitch -= pitch_step
    T0 = cam_to_world(x, y, z, yaw, pitch)
    T1 = cam_to_world(nx, ny, nz, nyaw, npitch)
    a = rel_action(T0, T1, fov, fov, speed)
    assert a.shape == (ACT_DIM,), a.shape
    return a


def parse_spec(spec) -> list[tuple[str, int]]:
    """['forward 20', ('turn_left', 15)] → [('forward', 20), ('turn_left', 15)]。"""
    out = []
    for item in spec:
        if isinstance(item, (list, tuple)):
            name, n = item[0], int(item[1])
        else:
            parts = str(item).split()
            name, n = parts[0], int(parts[1]) if len(parts) > 1 else 1
        out.append((name, int(n)))
    return out
