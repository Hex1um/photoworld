import numpy as np, train, sys
i = int(sys.argv[1]) if len(sys.argv) > 1 else -1
d = np.load("data.npz")
X = d["X"]
P = dict(np.load("model.npz"))
ph, pc, _ = train.forward(P, X[i:i+1])
X[i:i+1].astype(np.float32).tofile("in.f32")
ph[0].astype(np.float32).tofile("out_h.f32")
pc[0].astype(np.float32).tofile("out_c.f32")
print("dumped sample", i, "h", ph[0].shape, "c", pc[0].shape, "hstd", round(float(ph.std()),4))
