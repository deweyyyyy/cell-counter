"""Export a trained CellNet (cellnet.keras) to the compact weights file the server loads.

    python export.py                      # -> cellnet_w.json next to this script
    MODEL=other.keras VERSION="CellNet v0.3" THR=0.85 python export.py

Copy the resulting cellnet_w.json into ../server/ and restart the server.
"""
import os, json, base64
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf


HERE = os.path.dirname(os.path.abspath(__file__))
m = tf.keras.models.load_model(os.environ.get("MODEL", os.path.join(HERE, "cellnet.keras")), compile=False)
convs = [l for l in m.layers if isinstance(l, tf.keras.layers.Conv2D)]
assert len(convs) == 11, "unexpected architecture"
buf, layers = [], []
for l in convs:
    k, b = l.get_weights()                       # Keras kernels are HWIO, as the server expects
    buf += [k.astype("<f4").ravel(), b.astype("<f4").ravel()]
    layers.append({"kshape": list(k.shape)})
blob = np.concatenate(buf).astype("<f4").tobytes()
out = {"version": os.environ.get("VERSION", "CellNet"), "thr": float(os.environ.get("THR", "0.85")), "target": 13,
       "layers": layers, "data": base64.b64encode(blob).decode()}
path = os.path.join(HERE, "cellnet_w.json")
json.dump(out, open(path, "w"), separators=(",", ":"))
print(f"wrote {path} ({len(blob)} bytes of weights, threshold {out['thr']})")
