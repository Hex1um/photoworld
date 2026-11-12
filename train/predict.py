"""Visual check: input | GT height | predicted height | GT colour | predicted colour"""
import numpy as np
from PIL import Image
import train

d = np.load("data.npz")
X, Yh, Yc = d["X"], d["Yh"], d["Yc"]
P = dict(np.load("model.npz"))
n = len(X)
sel = range(n - 6, n)

rows = []
for i in sel:
    ph, pc, _ = train.forward(P, X[i:i + 1])
    im = Image.fromarray((X[i] * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST)
    gh = Image.fromarray((np.clip(Yh[i], 0, 1) * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST).convert("RGB")
    phh = ph[0]
    phh = (phh - phh.min()) / (float(phh.max() - phh.min()) + 1e-6)
    pr = Image.fromarray((phh * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST).convert("RGB")
    gc = Image.fromarray((np.clip(Yc[i], 0, 1) * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST)
    pcimg = Image.fromarray((np.clip(pc[0], 0, 1) * 255).astype(np.uint8)).resize((192, 192), Image.NEAREST)
    row = Image.new("RGB", (192 * 5, 192))
    for k, im2 in enumerate([im, gh, pr, gc, pcimg]):
        row.paste(im2, (192 * k, 0))
    rows.append(row)

sheet = Image.new("RGB", (192 * 5, 192 * len(rows)))
for i, r in enumerate(rows):
    sheet.paste(r, (0, 192 * i))
sheet.save("preview/predict.png")

err_h = float(np.mean(np.abs(train.forward(P, X[-60:])[0] - Yh[-60:])))
err_c = float(np.mean(np.abs(train.forward(P, X[-60:])[1] - Yc[-60:])))
print("val L1  height %.4f  colour %.4f" % (err_h, err_c))
print("preview -> preview/predict.png")
