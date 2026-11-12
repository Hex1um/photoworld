"""Tiny image -> blocky-world regressor, trained with hand written numpy backprop.

in : 96x96x3 first person view
out: 24x24 height map + 24x24x3 albedo map (camera aligned)
"""
import argparse, base64, json, time
import numpy as np


# ---------------- conv helpers ----------------
def im2col(x, k, s, p):
    N, H, W, C = x.shape
    xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
    OH = (H + 2 * p - k) // s + 1
    OW = (W + 2 * p - k) // s + 1
    cols = np.empty((N, OH, OW, k, k, C), np.float32)
    for i in range(k):
        for j in range(k):
            cols[:, :, :, i, j, :] = xp[:, i:i + OH * s:s, j:j + OW * s:s, :]
    return cols.reshape(N * OH * OW, k * k * C), OH, OW


def col2im(dcols, xshape, k, s, p):
    N, H, W, C = xshape
    OH = (H + 2 * p - k) // s + 1
    OW = (W + 2 * p - k) // s + 1
    d = dcols.reshape(N, OH, OW, k, k, C)
    dxp = np.zeros((N, H + 2 * p, W + 2 * p, C), np.float32)
    for i in range(k):
        for j in range(k):
            dxp[:, i:i + OH * s:s, j:j + OW * s:s, :] += d[:, :, :, i, j, :]
    return dxp[:, p:p + H, p:p + W, :]


def conv_fwd(x, W, b, s=1, p=1, k=3):
    N, H, Wd, C = x.shape
    cols, OH, OW = im2col(x, k, s, p)
    out = cols @ W.T + b
    return out.reshape(N, OH, OW, -1), cols


def conv_bwd(dout, cols, W, xshape, s=1, p=1, k=3, need_dx=True):
    N, OH, OW, Co = dout.shape
    d2 = dout.reshape(-1, Co)
    dW = d2.T @ cols
    db = d2.sum(0)
    dx = None
    if need_dx:
        dx = col2im(d2 @ W, xshape, k, s, p)
    return dW, db, dx


def up2(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def up2_bwd(d):
    N, H, W, C = d.shape
    return d.reshape(N, H // 2, 2, W // 2, 2, C).sum(axis=(2, 4))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


# ---------------- model ----------------
SHAPES = [("c1", 3, 16, 2), ("c2", 16, 32, 2), ("c3", 32, 48, 2), ("c4", 48, 64, 2),
          ("d1", 64, 48, 1), ("d2", 48, 32, 1)]
HEAD_H, HEAD_C = 1, 3


def init_params(rng):
    P = {}
    for name, cin, cout, _ in SHAPES:
        sc = np.sqrt(2.0 / (cin * 9))
        P[name + "w"] = (rng.normal(0, sc, (cout, 9 * cin))).astype(np.float32)
        P[name + "b"] = np.zeros(cout, np.float32)
    P["hw"] = (rng.normal(0, np.sqrt(2.0 / (32 * 9)), (HEAD_H, 9 * 32))).astype(np.float32)
    P["hb"] = np.zeros(HEAD_H, np.float32)
    P["cw"] = (rng.normal(0, np.sqrt(2.0 / (32 * 9)), (HEAD_C, 9 * 32))).astype(np.float32)
    P["cb"] = np.zeros(HEAD_C, np.float32)
    return P


def forward(P, x):
    c = {}
    a = x
    a, c["c1"] = conv_fwd(a, P["c1w"], P["c1b"], 2); c["a1"] = a; a = np.maximum(a, 0)
    a, c["c2"] = conv_fwd(a, P["c2w"], P["c2b"], 2); c["a2"] = a; a = np.maximum(a, 0)
    a, c["c3"] = conv_fwd(a, P["c3w"], P["c3b"], 2); c["a3"] = a; a = np.maximum(a, 0)
    a, c["c4"] = conv_fwd(a, P["c4w"], P["c4b"], 2); c["a4"] = a; a = np.maximum(a, 0)
    c["pre1"] = a
    a = up2(a)
    a, c["d1"] = conv_fwd(a, P["d1w"], P["d1b"], 1); c["b1"] = a
    a = np.maximum(a, 0)
    c["pre2"] = a
    a = up2(a)
    a, c["d2"] = conv_fwd(a, P["d2w"], P["d2b"], 1); c["b2"] = a
    a = np.maximum(a, 0)
    c["feat"] = a
    zh, c["zh"] = conv_fwd(a, P["hw"], P["hb"], 1)
    zc, c["zc"] = conv_fwd(a, P["cw"], P["cb"], 1)
    return sigmoid(zh)[..., 0], sigmoid(zc), c


def backward(P, c, ph, pc, gt_h, gt_c, w_h, w_c):
    G = {}
    p3 = ph[..., None]
    gt_h3 = gt_h[..., None]
    dp = (np.sign(p3 - gt_h3) * w_h / p3.size).astype(np.float32)
    dzh = dp * p3 * (1 - p3)
    dw_h, db_h, dfeat = conv_bwd(dzh, c["zh"], P["hw"], c["feat"].shape, 1, 1)
    dpc = (np.sign(pc - gt_c) * w_c / pc.size).astype(np.float32)
    dzc = dpc * pc * (1 - pc)
    dw_c, db_c, dfeat2 = conv_bwd(dzc, c["zc"], P["cw"], c["feat"].shape, 1, 1)
    G["hw"], G["hb"], G["cw"], G["cb"] = dw_h, db_h, dw_c, db_c

    d = (dfeat + dfeat2) * (c["feat"] > 0)

    N = d.shape[0]
    sh2 = (N, c["pre2"].shape[1] * 2, c["pre2"].shape[2] * 2, c["pre2"].shape[3])
    dw, db, dx = conv_bwd(d, c["d2"], P["d2w"], sh2, 1, 1)
    G["d2w"], G["d2b"] = dw, db
    d = up2_bwd(dx) * (c["pre2"] > 0)

    sh1 = (N, c["pre1"].shape[1] * 2, c["pre1"].shape[2] * 2, c["pre1"].shape[3])
    dw, db, dx = conv_bwd(d, c["d1"], P["d1w"], sh1, 1, 1)
    G["d1w"], G["d1b"] = dw, db
    d = up2_bwd(dx) * (c["pre1"] > 0)

    chain = (("c4", "a3"), ("c3", "a2"), ("c2", "a1"))
    for name, src in chain:
        dw, db, dx = conv_bwd(d, c[name], P[name + "w"], c[src].shape, 2, 1)
        G[name + "w"], G[name + "b"] = dw, db
        d = dx * (c[src] > 0)
    dw, db, _ = conv_bwd(d, c["c1"], P["c1w"], (d.shape[0], 96, 96, 3), 2, 1, need_dx=False)
    G["c1w"], G["c1b"] = dw, db
    return G


# ---------------- adam ----------------
class Adam:
    def __init__(self, P, lr=2e-3):
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0
        self.lr = lr

    def step(self, P, G, lr=None):
        lr = lr or self.lr
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1 = 1 - b1 ** self.t
        c2 = 1 - b2 ** self.t
        for k in G:
            g = G[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            P[k] -= (lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + eps)).astype(np.float32)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.npz")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="model.npz")
    a = ap.parse_args()

    d = np.load(a.data)
    X, Yh, Yc = d["X"], d["Yh"], d["Yc"]
    if a.limit:
        X, Yh, Yc = X[:a.limit], Yh[:a.limit], Yc[:a.limit]
    n = len(X)
    ntr = int(n * 0.92)
    rng = np.random.default_rng(0)
    P = init_params(rng)
    opt = Adam(P, a.lr)
    print("train", ntr, "val", n - ntr, flush=True)

    for ep in range(a.epochs):
        idx = rng.permutation(ntr)
        tot, totv, t0 = 0.0, 0.0, time.time()
        for s in range(0, ntr, a.bs):
            b = idx[s:s + a.bs]
            xb, hb, cb = X[b], Yh[b], Yc[b]
            ph, pc, cache = forward(P, xb)
            lh = np.mean(np.abs(ph - hb))
            lc = np.mean(np.abs(pc - cb))
            tot += (0.55 * lh + 0.45 * lc) * len(b)
            G = backward(P, cache, ph, pc, hb, cb, 0.55, 0.45)
            opt.step(P, G, a.lr * (0.5 ** (ep / max(1, a.epochs / 2))))
        # val
        for s in range(ntr, n, 32):
            ph, pc, _ = forward(P, X[s:s + 32])
            totv += (0.55 * np.mean(np.abs(ph - Yh[s:s + 32])) + 0.45 * np.mean(np.abs(pc - Yc[s:s + 32]))) * min(32, n - s)
        print("ep %02d loss %.4f val %.4f  %.0fs" % (ep, tot / ntr, totv / max(1, n - ntr), time.time() - t0), flush=True)

    np.savez(a.out, **P)
    w = {k: b64(v) for k, v in P.items()}
    meta = {"shapes": {k: list(v.shape) for k, v in P.items()}}
    with open("weights.json", "w") as f:
        json.dump({"meta": meta, "w": w}, f)
    print("wrote model.npz + weights.json", flush=True)


if __name__ == "__main__":
    main()
