"""Image -> blocky world regressor (v2).

v2 vs v1:
  * encoder-decoder with a skip connection at 24x24 (keeps spatial detail)
  * height loss = MSE + variance-matching term (fights the flat-output collapse of L1)
  * per-sample normalised height targets (produced by gen.py)
  * cosine LR decay, more channels
"""
import argparse, base64, json, math, time
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
CH = dict(c1=24, c2=48, c3=64, c4=96, d1=64, d2=48)
SKIP_CH = CH["c2"]
FEAT_CH = CH["d2"] + SKIP_CH


def init_params(rng):
    P = {}

    def mk(name, cin, cout, scale=2.0):
        P[name + "w"] = rng.normal(0, math.sqrt(scale / (cin * 9)), (cout, 9 * cin)).astype(np.float32)
        P[name + "b"] = np.zeros(cout, np.float32)

    mk("c1", 3, CH["c1"])
    mk("c2", CH["c1"], CH["c2"])
    mk("c3", CH["c2"], CH["c3"])
    mk("c4", CH["c3"], CH["c4"])
    mk("d1", CH["c4"], CH["d1"])
    mk("d2", CH["d1"], CH["d2"])
    sc = math.sqrt(1.0 / (FEAT_CH * 9))
    P["hw"] = rng.normal(0, sc, (1, 9 * FEAT_CH)).astype(np.float32)
    P["hb"] = np.zeros(1, np.float32)
    P["cw"] = rng.normal(0, sc, (3, 9 * FEAT_CH)).astype(np.float32)
    P["cb"] = np.zeros(3, np.float32)
    return P


def forward(P, x):
    c = {}
    a, c["c1"] = conv_fwd(x, P["c1w"], P["c1b"], 2); c["a1"] = a; a = np.maximum(a, 0)
    a, c["c2"] = conv_fwd(a, P["c2w"], P["c2b"], 2); c["a2"] = a; a = np.maximum(a, 0)
    c["skip"] = a
    a, c["c3"] = conv_fwd(a, P["c3w"], P["c3b"], 2); c["a3"] = a; a = np.maximum(a, 0)
    a, c["c4"] = conv_fwd(a, P["c4w"], P["c4b"], 2); c["a4"] = a; a = np.maximum(a, 0)
    c["pre1"] = a
    a = up2(a)
    a, c["d1"] = conv_fwd(a, P["d1w"], P["d1b"], 1); c["b1"] = a; a = np.maximum(a, 0)
    c["pre2"] = a
    a = up2(a)
    a, c["d2"] = conv_fwd(a, P["d2w"], P["d2b"], 1); c["b2"] = a; a = np.maximum(a, 0)
    a = np.concatenate([a, c["skip"]], axis=-1)
    c["feat"] = a
    zh, c["zh"] = conv_fwd(a, P["hw"], P["hb"], 1)
    zc, c["zc"] = conv_fwd(a, P["cw"], P["cb"], 1)
    return sigmoid(zh)[..., 0], sigmoid(zc), c


def backward(P, c, ph, pc, gt_h, gt_c, w_h, w_c, lam_var=0.55):
    G = {}
    p3 = ph[..., None]
    gt3 = gt_h[..., None]
    n = float(p3.size)

    mp = float(p3.mean())
    mt = float(gt3.mean())
    sp = math.sqrt(float(((p3 - mp) ** 2).mean()) + 1e-8)
    st = math.sqrt(float(((gt3 - mt) ** 2).mean()) + 1e-8)
    sgn = 1.0 if sp >= st else -1.0
    dp = (2.0 * (p3 - gt3) / n) + lam_var * sgn * (p3 - mp) / (n * sp)
    dzh = (dp * p3 * (1 - p3) * w_h).astype(np.float32)
    lh = float(((p3 - gt3) ** 2).mean()) + lam_var * abs(sp - st)
    dw_h, db_h, dfeat = conv_bwd(dzh, c["zh"], P["hw"], c["feat"].shape, 1, 1)
    G["hw"], G["hb"] = dw_h, db_h

    dpc = (np.sign(pc - gt_c) * 0.5 + 2.0 * (pc - gt_c)) / pc.size
    dzc = (dpc * pc * (1 - pc) * w_c).astype(np.float32)
    lc = float(np.mean(np.abs(pc - gt_c))) * 0.5 + float(np.mean((pc - gt_c) ** 2)) * 0.5
    dw_c, db_c, dfeat2 = conv_bwd(dzc, c["zc"], P["cw"], c["feat"].shape, 1, 1)
    G["cw"], G["cb"] = dw_c, db_c

    d = (dfeat + dfeat2) * (c["feat"] > 0)
    dec = d[..., :CH["d2"]]
    dsk = d[..., CH["d2"]:FEAT_CH]

    N = d.shape[0]
    sh2 = (N, c["pre2"].shape[1] * 2, c["pre2"].shape[2] * 2, c["pre2"].shape[3])
    dw, db, dx = conv_bwd(dec, c["d2"], P["d2w"], sh2, 1, 1)
    G["d2w"], G["d2b"] = dw, db
    d = up2_bwd(dx) * (c["pre2"] > 0)

    sh1 = (N, c["pre1"].shape[1] * 2, c["pre1"].shape[2] * 2, c["pre1"].shape[3])
    dw, db, dx = conv_bwd(d, c["d1"], P["d1w"], sh1, 1, 1)
    G["d1w"], G["d1b"] = dw, db
    d = up2_bwd(dx) * (c["pre1"] > 0)

    # c4: 6x6 -> 12x12
    dw, db, dx = conv_bwd(d, c["c4"], P["c4w"], c["a3"].shape, 2, 1)
    G["c4w"], G["c4b"] = dw, db
    d = dx * (c["a3"] > 0)

    # c3: 12x12 -> 24x24 , 跳连在这里并入
    dw, db, dx = conv_bwd(d, c["c3"], P["c3w"], c["a2"].shape, 2, 1)
    G["c3w"], G["c3b"] = dw, db
    d = dx * (c["a2"] > 0) + dsk * (c["skip"] > 0)

    # c2: 24 -> 48
    dw, db, dx = conv_bwd(d, c["c2"], P["c2w"], c["a1"].shape, 2, 1)
    G["c2w"], G["c2b"] = dw, db
    d = dx * (c["a1"] > 0)
    dw, db, _ = conv_bwd(d, c["c1"], P["c1w"], (N, 96, 96, 3), 2, 1, need_dx=False)
    G["c1w"], G["c1b"] = dw, db
    return G, lh, lc


class Adam:
    def __init__(self, P, lr):
        self.m = {k: np.zeros_like(v) for k, v in P.items()}
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
        self.t = 0

    def step(self, P, G, lr):
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="model.npz")
    a = ap.parse_args()

    d = np.load(a.data)
    X, Yh, Yc = d["X"], d["Yh"], d["Yc"]
    n = len(X)
    ntr = int(n * 0.93)
    rng = np.random.default_rng(0)
    P = init_params(rng)
    opt = Adam(P, a.lr)
    print("train", ntr, "val", n - ntr, flush=True)

    for ep in range(a.epochs):
        idx = rng.permutation(ntr)
        tot, t0 = 0.0, time.time()
        lr = a.lr * 0.5 * (1 + math.cos(math.pi * ep / a.epochs))
        for s in range(0, ntr, a.bs):
            b = idx[s:s + a.bs]
            ph, pc, cache = forward(P, X[b])
            G, lh, lc = backward(P, cache, ph, pc, Yh[b], Yc[b], 0.6, 0.4)
            opt.step(P, G, max(lr, 2e-4))
            tot += (0.6 * lh + 0.4 * lc) * len(b)
        vh = vc = vsp = 0.0
        for s in range(ntr, n, 32):
            ph, pc, _ = forward(P, X[s:s + 32])
            m = min(32, n - s)
            vh += float(np.mean((ph - Yh[s:s + 32]) ** 2)) * m
            vc += float(np.mean(np.abs(pc - Yc[s:s + 32]))) * m
            vsp += float(np.std(ph)) * m
        vn = max(1, n - ntr)
        print("ep %02d train %.5f val_mse %.5f val_l1c %.4f val_std %.3f lr %.5f %.0fs"
              % (ep, tot / ntr, vh / vn, vc / vn, vsp / vn, lr, time.time() - t0), flush=True)

    np.savez(a.out, **P)
    with open("weights.json", "w") as f:
        json.dump({"meta": {"shapes": {k: list(v.shape) for k, v in P.items()}},
                   "w": {k: b64(v) for k, v in P.items()}}, f)
    print("wrote model.npz + weights.json", flush=True)


if __name__ == "__main__":
    main()
