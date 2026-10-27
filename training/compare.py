"""Head-to-head: compare trained models on the same held-out synthetic set.   python compare.py a.keras b.keras"""
import os, sys; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
from train import loss_fn, norm, peaks, match, dataset, P

X, Y, C = dataset(600, 77_000_000)                      # seeds far from both training sets
Xn = np.stack([norm(x.astype(np.float32) / 255) for x in X])
ok = lambda q: (q[:, 0] >= 6) & (q[:, 1] >= 6) & (q[:, 0] < P - 6) & (q[:, 1] < P - 6)
for name in sys.argv[1:]:
    m = tf.keras.models.load_model(name, compile=False)
    pv = m.predict(Xn, batch_size=32, verbose=0)[..., 0]
    print(name)
    for thr in (.6, .7, .78, .85):
        TP = FP = FN = 0
        for hm, c in zip(pv, C):
            pr = peaks(hm, thr); pr = pr[ok(pr)] if len(pr) else pr
            cc = c[ok(c)] if len(c) else c
            a, b, d = match(pr, cc[:, :2] if len(cc) else np.zeros((0, 2)), max(3, np.median(c[:, 2]) if len(c) else 6))
            TP += a; FP += b; FN += d
        print(f"  thr {thr}: recall {TP/(TP+FN):.3f}  precision {TP/max(1,TP+FP):.3f}  F1 {2*TP/(2*TP+FP+FN):.3f}  count err {(FP-FN)/(TP+FN)*100:+.1f}%")
