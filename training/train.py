"""CellNet v0: tiny U-Net predicting a cell-centre heatmap (one class: cell)."""
import os, sys, time, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
from gen import make_patch, heatmap

P = 128
HERE = os.path.dirname(os.path.abspath(__file__))

def norm(img):
    # per-image, per-channel standardisation: makes dark and bright photos look alike to the net
    m = img.reshape(-1, 3).mean(0); s = img.reshape(-1, 3).std(0) + 1e-3
    return (img - m) / s

def _gen(seed):
    rng = np.random.default_rng(seed)
    im, c = make_patch(rng, P, P, d_range=(8, 19))
    return (im * 255).astype(np.uint8), heatmap(c, P, P)[0].astype(np.float16), np.array([(x, y, r) for x, y, _, r in c], np.float32).reshape(-1, 3)

def dataset(n, seed0):
    with Pool(2) as p:
        out = p.map(_gen, range(seed0, seed0 + n), chunksize=64)
    X = np.stack([o[0] for o in out]); Y = np.stack([o[1] for o in out]); C = [o[2] for o in out]
    return X, Y, C

def build():
    inp = tf.keras.Input((None, None, 3))
    def blk(x, f):
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        return tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    e1 = blk(inp, 16)
    e2 = blk(tf.keras.layers.MaxPool2D()(e1), 32)
    e3 = blk(tf.keras.layers.MaxPool2D()(e2), 64)
    d2 = blk(tf.keras.layers.Concatenate()([tf.keras.layers.UpSampling2D()(e3), e2]), 32)
    d1 = blk(tf.keras.layers.Concatenate()([tf.keras.layers.UpSampling2D()(d2), e1]), 16)
    out = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return tf.keras.Model(inp, out)

def loss_fn(y, p):
    w = 1 + 25 * y
    return tf.reduce_mean(w * tf.square(y - p))

def peaks(hm, thr=0.35, win=5):
    import scipy.ndimage as nd
    mx = nd.maximum_filter(hm, size=win)
    ys, xs = np.where((hm == mx) & (hm > thr))
    return np.stack([xs, ys], 1).astype(np.float32) if len(xs) else np.zeros((0, 2), np.float32)

def match(pred, gt, rad):
    used = set(); tp = 0
    for p in pred:
        if len(gt) == 0: break
        d = np.hypot(gt[:, 0] - p[0], gt[:, 1] - p[1])
        for j in np.argsort(d):
            if d[j] > rad: break
            if j not in used: used.add(j); tp += 1; break
    return tp, len(pred) - tp, len(gt) - tp

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t = time.time()
    X, Y, _ = dataset(N, 1000)
    Xv, Yv, Cv = dataset(400, 9_000_000)
    print("data", X.shape, f"{time.time()-t:.0f}s", flush=True)
    Xvn = np.stack([norm(x.astype(np.float32) / 255) for x in Xv]).astype(np.float32)
    Yv = Yv.astype(np.float32)[..., None]
    def tfnorm(x):
        x = tf.cast(x, tf.float32) / 255.
        m = tf.reduce_mean(x, [0, 1], keepdims=True)
        s = tf.math.reduce_std(x, [0, 1], keepdims=True) + 1e-3
        return (x - m) / s
    # augmentation: flips / 90° rotations, normalisation done per patch here (keeps RAM low)
    def aug(x, y):
        y = tf.cast(y, tf.float32)[..., None]
        k = tf.random.uniform([], 0, 4, tf.int32)
        x = tf.image.rot90(x, k); y = tf.image.rot90(y, k)
        if tf.random.uniform([]) < .5: x = tf.image.flip_left_right(x); y = tf.image.flip_left_right(y)
        return tfnorm(x), y
    def gen_xy():
        idx = np.random.permutation(len(X))
        for i in idx: yield X[i], Y[i]
    ds = tf.data.Dataset.from_generator(gen_xy, output_signature=(tf.TensorSpec((P, P, 3), tf.uint8), tf.TensorSpec((P, P), tf.float16))) \
        .map(aug, num_parallel_calls=2).batch(16).prefetch(2)
    m = build()
    if os.path.exists(os.path.join(HERE, "cellnet.keras")) and os.environ.get("WARM"):
        m.load_weights(os.path.join(HERE, "cellnet.keras")); print("warm start", flush=True)
    print("params", m.count_params(), flush=True)
    lr = tf.keras.optimizers.schedules.CosineDecay(2e-3, EPOCHS * (N // 16))
    m.compile(tf.keras.optimizers.Adam(lr), loss_fn)
    m.fit(ds, epochs=EPOCHS, validation_data=(Xvn, Yv), verbose=2)
    m.save(os.path.join(HERE, "cellnet.keras"))
    # eval detection
    pv = m.predict(Xvn, batch_size=32, verbose=0)[..., 0]
    best = None
    for thr in (.5, .6, .7, .78, .85):
        TP = FP = FN = 0
        for hm, c in zip(pv, Cv):
            pr = peaks(hm, thr)
            # ignore a 6 px border: cells cut by the patch edge are ambiguous for both sides
            ok = lambda q: (q[:, 0] >= 6) & (q[:, 1] >= 6) & (q[:, 0] < P - 6) & (q[:, 1] < P - 6)
            pr = pr[ok(pr)] if len(pr) else pr
            c = c[ok(c)] if len(c) else c
            rad = np.median(c[:, 2]) if len(c) else 6
            a, b, d = match(pr, c[:, :2], max(3, rad)); TP += a; FP += b; FN += d
        f1 = 2 * TP / max(1, 2 * TP + FP + FN)
        print(f"thr {thr}: TP {TP} FP {FP} FN {FN} F1 {f1:.3f} count err {(TP+FP-(TP+FN))/(TP+FN)*100:+.1f}%")
        if best is None or f1 > best[1]: best = (thr, f1)
    json.dump({"thr": best[0], "f1": best[1]}, open(os.path.join(HERE, "best.json"), "w"))
    print("best", best)
