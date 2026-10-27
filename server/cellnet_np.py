"""CellNet inference in pure NumPy (no TensorFlow needed on the Mac).

Each 3x3 conv is 9 shifted matrix multiplies, so memory stays at a few copies of the
feature map. Images are processed in tiles with overlap, like the browser version.
"""
import base64, json
import numpy as np


class CellNet:
    def __init__(self, weights_path):
        J = json.load(open(weights_path))
        f = np.frombuffer(base64.b64decode(J["data"]), dtype="<f4")
        self.layers, o = [], 0
        for l in J["layers"]:
            ks = l["kshape"]; kn = int(np.prod(ks))
            k = f[o:o + kn].reshape(ks); o += kn
            b = f[o:o + ks[3]]; o += ks[3]
            self.layers.append((k.astype(np.float32), b.astype(np.float32)))
        self.thr, self.target, self.version = J["thr"], J["target"], J.get("version", "cellnet")

    def _conv(self, x, i, act="relu"):
        k, b = self.layers[i]
        kh, kw, cin, cout = k.shape
        H, W, _ = x.shape
        if kh == 1:
            y = x.reshape(-1, cin) @ k[0, 0] + b
            y = y.reshape(H, W, cout)
        else:
            xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
            y = np.empty((H, W, cout), np.float32); y[:] = b
            for dy in range(3):
                for dx in range(3):
                    y += (xp[dy:dy + H, dx:dx + W].reshape(-1, cin) @ k[dy, dx]).reshape(H, W, cout)
        if act == "relu":
            np.maximum(y, 0, out=y)
        else:
            y = 1 / (1 + np.exp(-y))
        return y

    @staticmethod
    def _pool(x):
        H, W, C = x.shape
        return x.reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))

    @staticmethod
    def _up(x):
        return x.repeat(2, 0).repeat(2, 1)

    def _net(self, x):
        c = self._conv
        e1 = c(c(x, 0), 1)
        e2 = c(c(self._pool(e1), 2), 3)
        e3 = c(c(self._pool(e2), 4), 5)
        d2 = c(c(np.concatenate([self._up(e3), e2], -1), 6), 7)
        d1 = c(c(np.concatenate([self._up(d2), e1], -1), 8), 9)
        return c(d1, 10, "sigmoid")[..., 0]

    # ---------------- Apple GPU (MLX) path ----------------
    def use_mlx(self):
        """Switch to Apple's MLX (Metal GPU on Apple-silicon Macs). Returns the device name, or None."""
        try:
            import mlx.core as mx
        except Exception:
            return None
        self.mx = mx
        # Keras HWIO -> MLX (out, kh, kw, in)
        self.mlayers = [(mx.array(np.ascontiguousarray(k.transpose(3, 0, 1, 2))), mx.array(b)) for k, b in self.layers]
        self.backend = f"MLX ({mx.default_device()})"
        return self.backend

    def _mlx_net(self, x):
        mx = self.mx
        def c(t, i, act="relu"):
            k, b = self.mlayers[i]
            y = mx.conv2d(t, k, padding=k.shape[1] // 2) + b
            return mx.maximum(y, 0) if act == "relu" else mx.sigmoid(y)
        def pool(t):
            n, h, w, ch = t.shape
            return t.reshape(n, h // 2, 2, w // 2, 2, ch).max(axis=(2, 4))
        def up(t):
            return mx.repeat(mx.repeat(t, 2, axis=1), 2, axis=2)
        e1 = c(c(x, 0), 1)
        e2 = c(c(pool(e1), 2), 3)
        e3 = c(c(pool(e2), 4), 5)
        d2 = c(c(mx.concatenate([up(e3), e2], -1), 6), 7)
        d1 = c(c(mx.concatenate([up(d2), e1], -1), 8), 9)
        return c(d1, 10, "sigmoid")[0, ..., 0]

    def heatmap(self, rgb, tile=256, pad=24):
        """rgb: float32 HxWx3 in 0..1 -> heatmap HxW"""
        H, W, _ = rgb.shape
        m = rgb.reshape(-1, 3).mean(0); s = rgb.reshape(-1, 3).std(0) + 1e-3
        x = ((rgb - m) / s).astype(np.float32)
        if getattr(self, "mx", None) is not None:
            # GPU has room for a whole square at once: no tiles, no overlap work
            ph, pw = (4 - H % 4) % 4, (4 - W % 4) % 4
            xp = np.pad(x, ((0, ph), (0, pw), (0, 0)))
            hm = self._mlx_net(self.mx.array(xp[None]))
            return np.array(hm, copy=False)[:H, :W].astype(np.float32)
        out = np.zeros((H, W), np.float32)
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                ty, tx = min(tile, H - y0), min(tile, W - x0)
                ay, ax = max(0, y0 - pad), max(0, x0 - pad)
                by, bx = min(H, y0 + ty + pad), min(W, x0 + tx + pad)
                t = x[ay:by, ax:bx]
                h, w = t.shape[:2]
                t = np.pad(t, ((0, (4 - h % 4) % 4), (0, (4 - w % 4) % 4), (0, 0)))
                hm = self._net(t)
                out[y0:y0 + ty, x0:x0 + tx] = hm[y0 - ay:y0 - ay + ty, x0 - ax:x0 - ax + tx]
        return out

    @staticmethod
    def peaks(hm, thr, r=2):
        """local maxima above thr in a (2r+1)^2 window, refined by 3x3 weighted centroid"""
        H, W = hm.shape
        p = np.pad(hm, r, constant_values=-1)
        mx = np.full_like(hm, -1)
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                mx = np.maximum(mx, p[dy:dy + H, dx:dx + W])
        ys, xs = np.where((hm >= mx) & (hm > thr))
        pts, taken = [], set()
        for y, x in zip(ys, xs):                       # break plateaus: keep first in scan order
            if any((y + dy, x + dx) in taken for dy in range(-r, r + 1) for dx in range(-r, r + 1)):
                continue
            taken.add((y, x))
            y0, y1, x0, x1 = max(0, y - 1), min(H, y + 2), max(0, x - 1), min(W, x + 2)
            w = hm[y0:y1, x0:x1]; gy, gx = np.mgrid[y0:y1, x0:x1]
            pts.append((float((w * gx).sum() / w.sum()), float((w * gy).sum() / w.sum()), float(hm[y, x])))
        return pts
