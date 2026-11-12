"""Synthetic data: random blocky worlds + one first-person render of each.

Targets are re-sampled into CAMERA-ALIGNED coordinates, so one image -> one
grid that sits in front of the viewer (this is what the browser rebuilds).

    image  (96,96,3)     first-person view, 0..1
    H      (24,24)       height / MAXH          aligned: row0 = far ahead
    C      (24,24,3)     albedo                aligned: same frame
"""
import argparse, os
import numpy as np
from PIL import Image

WORLD = 48
GRID = 24
MAXH = 16.0
IMG = 96
HALF = (GRID - 1) / 2.0


def smooth_noise(size, freq, rng):
    g = rng.random((freq + 2, freq + 2)).astype(np.float32)
    xs = np.linspace(0, freq, size, endpoint=False)
    x0 = np.floor(xs).astype(int)
    fx = xs - x0
    fx = fx * fx * (3 - 2 * fx)
    a = g[np.ix_(x0, x0)]
    b = g[np.ix_(x0, x0 + 1)]
    c = g[np.ix_(x0 + 1, x0)]
    d = g[np.ix_(x0 + 1, x0 + 1)]
    fx2 = fx[None, :]
    fy2 = fx[:, None]
    return (a * (1 - fx2) + b * fx2) * (1 - fy2) + (c * (1 - fx2) + d * fx2) * fy2


PALETTES = [
    ((0.31, 0.50, 0.23), (0.44, 0.34, 0.21)),
    ((0.78, 0.71, 0.48), (0.62, 0.54, 0.34)),
    ((0.42, 0.43, 0.45), (0.30, 0.30, 0.32)),
    ((0.80, 0.83, 0.86), (0.60, 0.62, 0.65)),
    ((0.24, 0.42, 0.40), (0.16, 0.28, 0.30)),
    ((0.52, 0.30, 0.28), (0.36, 0.22, 0.22)),
]


def make_world(rng):
    n1 = smooth_noise(WORLD, 5, rng)
    n2 = smooth_noise(WORLD, 9, rng)
    n3 = smooth_noise(WORLD, 16, rng)
    h = np.floor(2.0 + n1 * 9.0 + n2 * 3.0 + n3 * 1.0).clip(0, MAXH).astype(np.float32)

    pal, acc = PALETTES[rng.integers(0, len(PALETTES))]
    pal = np.array(pal, np.float32)
    acc = np.array(acc, np.float32)
    mix = smooth_noise(WORLD, 10, rng)[..., None]
    col = (pal[None, None, :] * (1 - mix) + acc[None, None, :] * mix).astype(np.float32)
    col *= (0.90 + 0.20 * rng.random((WORLD, WORLD, 1))).astype(np.float32)

    kind = rng.integers(0, 3)
    if kind == 0:
        for _ in range(rng.integers(10, 26)):
            x, y = rng.integers(2, WORLD - 2, 2)
            th = rng.integers(3, 6) * 0.55
            h[max(0, y - 1):y + 2, max(0, x - 1):x + 2] += th
            col[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = np.array([0.20, 0.36, 0.18], np.float32) * (0.85 + 0.3 * rng.random())
    elif kind == 1:
        for _ in range(rng.integers(5, 14)):
            x, y = rng.integers(2, WORLD - 5, 2)
            w = rng.integers(2, 4)
            hh = rng.integers(3, 9)
            h[y:y + w, x:x + w] += hh
            g = 0.36 + 0.2 * rng.random()
            col[y:y + w, x:x + w] = np.array([g, g * 0.97, g * 0.93], np.float32)

    return h.clip(0, MAXH), col.astype(np.float32)


def camera_basis(yaw, pitch):
    fwd = np.array([np.cos(yaw) * np.cos(pitch), np.sin(pitch), np.sin(yaw) * np.cos(pitch)], np.float32)
    right = np.array([-np.sin(yaw), 0, np.cos(yaw)], np.float32)
    up = np.cross(right, fwd).astype(np.float32)
    return fwd, right, up


def render(h, col, camx, camz, camy, yaw, pitch, rng, size=IMG):
    fwd, right, up = camera_basis(yaw, pitch)
    fov = 1.15
    tanf = np.tan(fov / 2)
    u = (np.arange(size) + 0.5) / size * 2 - 1
    uu, vv = np.meshgrid(u, -u)
    d = fwd[None, None, :] + right[None, None, :] * (uu[..., None] * tanf) + up[None, None, :] * (vv[..., None] * tanf)
    d /= np.linalg.norm(d, axis=2, keepdims=True)

    t = np.full((size, size), 0.35, np.float32)
    hit = np.zeros((size, size), bool)
    dist = np.full((size, size), 60.0, np.float32)
    for _ in range(130):
        px = camx + d[..., 0] * t
        pz = camz + d[..., 2] * t
        py = camy + d[..., 1] * t
        ix = np.clip(px, 0, WORLD - 1).astype(np.int32)
        iz = np.clip(pz, 0, WORLD - 1).astype(np.int32)
        newhit = (~hit) & (py <= h[iz, ix])
        hit |= newhit
        dist[newhit] = t[newhit]
        t += 0.5
        if t[0, 0] > 58:
            break

    ix = np.clip(camx + d[..., 0] * dist, 0, WORLD - 1).astype(np.int32)
    iz = np.clip(camz + d[..., 2] * dist, 0, WORLD - 1).astype(np.int32)

    dhx = np.zeros_like(h)
    dhz = np.zeros_like(h)
    dhx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
    dhz[1:-1, :] = (h[2:, :] - h[:-2, :]) * 0.5
    n = np.stack([-dhx * 0.6, np.ones_like(h), -dhz * 0.6], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nrm = n[iz, ix]

    sun = np.array([0.45, 0.78, 0.35], np.float32)
    sun /= np.linalg.norm(sun)
    ndl = np.clip((nrm * sun[None, None, :]).sum(-1), 0, 1)
    shade = 0.52 + 0.55 * ndl
    aoc = np.clip(0.72 + 0.28 * (h[iz, ix] / max(1.0, h.max())), 0, 1)
    lit = col[iz, ix] * (shade * aoc)[..., None]

    sky = np.stack([
        np.clip(0.42 + 1.05 * d[..., 1], 0, 1),
        np.clip(0.52 + 0.95 * d[..., 1], 0, 1),
        np.clip(0.66 + 0.80 * d[..., 1], 0, 1),
    ], axis=-1)
    haze = np.clip(1.0 - np.abs(d[..., 1]) * 5.0, 0, 1)[..., None]
    horizon = np.array([0.86, 0.83, 0.78], np.float32)
    sky = sky * (1 - haze * 0.62) + horizon * (haze * 0.62)
    fogc = horizon[None, None, :] * (0.75 + 0.25 * sky)
    fog = np.clip((dist - 14.0) / 46.0, 0, 1)[..., None] ** 1.6
    out = np.where(hit[..., None], lit * (1 - fog) + fogc * fog, sky)

    out = np.clip(out, 0, 1) ** (0.95 + rng.uniform(-0.08, 0.08))
    out = out * rng.uniform(0.9, 1.1) + rng.normal(0, 0.012, out.shape)
    return np.clip(out, 0, 1).astype(np.float32)


def align(h, col, camx, camz, yaw):
    """Resample world into a camera-aligned GRID x GRID patch: row0 = far ahead."""
    i = np.arange(GRID, dtype=np.float32)
    j = np.arange(GRID, dtype=np.float32)
    di = (18.0 - i)                       # patch runs from 18 ahead to 5 behind

    dj = (j - HALF)                       # left .. right
    wx = camx + di[:, None] * np.cos(yaw) + dj[None, :] * (-np.sin(yaw))
    wz = camz + di[:, None] * np.sin(yaw) + dj[None, :] * np.cos(yaw)
    ix = np.clip(np.round(wx), 0, WORLD - 1).astype(np.int32)
    iz = np.clip(np.round(wz), 0, WORLD - 1).astype(np.int32)
    return h[iz, ix].astype(np.float32), col[iz, ix].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="data.npz")
    ap.add_argument("--preview", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    X = np.zeros((a.n, IMG, IMG, 3), np.float32)
    Yh = np.zeros((a.n, GRID, GRID), np.float32)
    Yc = np.zeros((a.n, GRID, GRID, 3), np.float32)

    for i in range(a.n):
        h, c = make_world(rng)
        camx = rng.uniform(14, WORLD - 14)
        camz = rng.uniform(14, WORLD - 14)
        camy = h[int(camz), int(camx)] + rng.uniform(1.5, 4.0)
        yaw = rng.uniform(0, np.pi * 2)
        pitch = rng.uniform(-0.34, -0.02)
        img = render(h, c, camx, camz, camy, yaw, pitch, rng)
        ah, ac = align(h, c, camx, camz, yaw)
        if rng.random() < 0.5:
            img = img[:, ::-1].copy()
            ah = ah[:, ::-1].copy()
            ac = ac[:, ::-1].copy()
        X[i] = img
        var0 = float(ah.min()); var1 = float(ah.max())
        Yh[i] = (ah - var0) / max(1e-6, (var1 - var0))
        Yc[i] = ac
        if i % 200 == 0:
            print(i, flush=True)

    np.savez_compressed(a.out, X=X, Yh=Yh, Yc=Yc)
    print("saved", a.out, X.shape, flush=True)

    if a.preview:
        os.makedirs("preview", exist_ok=True)
        rows = []
        for i in range(a.preview):
            im = Image.fromarray((X[i] * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST)
            hp = Yh[i]
            hp = (hp - hp.min()) / (float(hp.max() - hp.min()) + 1e-6)
            hm = Image.fromarray((hp * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST).convert("RGB")
            cm = Image.fromarray((Yc[i] * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST)
            row = Image.new("RGB", (192 * 3, 192))
            row.paste(im, (0, 0)); row.paste(hm, (192, 0)); row.paste(cm, (384, 0))
            rows.append(row)
        sheet = Image.new("RGB", (192 * 3, 192 * a.preview))
        for i, r in enumerate(rows):
            sheet.paste(r, (0, 192 * i))
        sheet.save("preview/samples.png")
        print("preview -> preview/samples.png")


if __name__ == "__main__":
    main()
