"""Extra analysis on the image the phone sends for counting.

classify_dead(rgb, pts, target)  trypan-blue rule: a detected cell is dead when its centre is
                                 bluer than the background around it and not brighter.
detect_frame(rgb)                finds the 4x4 counting square of a hemocytometer from its grid lines.
Both work on the image at model scale (cells about `target` px across), values 0..1.
"""
import numpy as np
from PIL import Image, ImageFilter

LUM = np.array([.299, .587, .114], np.float32)
DEAD_BLUE = 0.06      # blue excess (B-R, relative to background) above which a cell counts as stained
DEAD_LUM = 0.0        # ...and its centre is no brighter than the background


def classify_dead(rgb, pts, target):
    """Return a list of 0/1 (1 = dead) for each (x, y, ...) in pts."""
    rc, r1, r2 = 0.32 * target, 0.9 * target, 1.5 * target
    R = int(np.ceil(r2)) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.hypot(xx, yy)
    cm, am = d <= rc, (d >= r1) & (d <= r2)
    P = np.pad(rgb, ((R, R), (R, R), (0, 0)), mode="edge")
    out = []
    for p in pts:
        xi, yi = int(round(p[0])) + R, int(round(p[1])) + R
        win = P[yi - R:yi + R + 1, xi - R:xi + R + 1]
        if win.shape[:2] != cm.shape:
            out.append(0); continue
        c, b = win[cm].mean(0), np.median(win[am], 0)
        lc, lb = float(c @ LUM), float(b @ LUM)
        lum_rel = (lc - lb) / (lb + .05)
        blue_rel = ((c[2] - c[0]) - (b[2] - b[0])) / (lb + .05)
        out.append(1 if blue_rel > DEAD_BLUE and lum_rel < DEAD_LUM else 0)
    return out


def _peaks(p, mind):
    med = np.median(p); mad = np.median(np.abs(p - med)) + 1e-6
    cand = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] >= p[i + 1] and p[i] > med + 4 * mad]
    cand.sort(key=lambda i: -p[i]); out = []
    for i in cand:
        if all(abs(i - j) >= mind for j in out): out.append(i)
    return sorted(out), med, mad


def _runs(pk, strength, n, smin, smax):
    """Best set of n equally spaced lines with no other line between them."""
    best = None
    for a in range(len(pk)):
        for b in range(a + n - 1, len(pk)):
            s = (pk[b] - pk[a]) / (n - 1)
            if s < smin: continue
            if s > smax: break
            tol = max(3, 0.07 * s); idx = [a]; ok = True
            for i in range(1, n - 1):
                tgt = pk[a] + i * s
                j = min(range(len(pk)), key=lambda k: abs(pk[k] - tgt))
                if abs(pk[j] - tgt) > tol: ok = False; break
                idx.append(j)
            if not ok: continue
            idx.append(b)
            if any(pk[a] + tol < pk[k] < pk[b] - tol and k not in idx for k in range(len(pk))): continue
            pos = [pk[j] for j in idx]; sp = np.diff(pos)
            sc = sum(strength[j] for j in idx) * (1 - np.std(sp) / np.mean(sp))
            if best is None or sc > best[0]: best = (sc, pos[0], pos[-1], float(np.mean(sp)))
    return best


def detect_frame(rgb):
    """Bounding box {x, y, w, h, angle} of the 4x4 large square, or None when no whole square is visible."""
    H, W, _ = rgb.shape
    g = Image.fromarray((np.clip(rgb @ LUM, 0, 1) * 255).astype(np.uint8))
    sc = min(1.0, 900 / max(H, W))
    gs = g.resize((max(8, round(W * sc)), max(8, round(H * sc))), Image.BILINEAR)
    best = None
    for ang in np.arange(-4, 4.01, 0.5):
        r = gs.rotate(float(ang), resample=Image.BILINEAR, fillcolor=0)
        a = np.asarray(r, np.float32)
        th = np.abs(a - np.asarray(r.filter(ImageFilter.GaussianBlur(4)), np.float32))   # thin lines, either polarity
        rows, cols = np.percentile(th, 75, axis=1), np.percentile(th, 75, axis=0)
        h, w = th.shape; smin, smax = 0.12 * min(h, w), 0.32 * max(h, w)
        pr, _, madr = _peaks(rows, max(3, int(0.012 * h)))
        pc, _, madc = _peaks(cols, max(3, int(0.012 * w)))
        ry, rx = _runs(pr, rows, 5, smin, smax), _runs(pc, cols, 5, smin, smax)
        if not ry or not rx or abs(ry[3] - rx[3]) / max(ry[3], rx[3]) > 0.15: continue
        score = ry[0] / (np.median(rows) + madr * 5) + rx[0] / (np.median(cols) + madc * 5)
        if best is None or score > best[0]: best = (score, ang, rx, ry, (w, h))
    if not best: return None
    _, ang, rx, ry, (w, h) = best
    cx, cy, t = w / 2, h / 2, np.deg2rad(ang)
    xs, ys = [], []
    for x, y in [(rx[1], ry[1]), (rx[2], ry[1]), (rx[1], ry[2]), (rx[2], ry[2])]:
        dx, dy = x - cx, y - cy
        xs.append((dx * np.cos(t) - dy * np.sin(t) + cx) / sc)
        ys.append((dx * np.sin(t) + dy * np.cos(t) + cy) / sc)
    return {"x": float(min(xs)), "y": float(min(ys)), "w": float(max(xs) - min(xs)), "h": float(max(ys) - min(ys)),
            "angle": float(ang)}
