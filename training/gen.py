"""Synthetic hemocytometer patches resembling dim brightfield Hep-55.1C + trypan blue.

Every patch comes with ground-truth cell centres (x, y, cls) where cls 0 = live, 1 = dead.
Rendering is deliberately varied (exposure, tint, cell size, rim contrast, blur, noise,
grid lines, debris) so the model learns the cell, not one particular look.
"""
import numpy as np

def _fast_blur(img, s):
    """separable gaussian via cumulative box approximation (3 box passes)"""
    if s < 0.3:
        return img
    import numpy as np
    w = max(1, int(round(np.sqrt(12 * s * s / 3 + 1))))
    r = w // 2
    out = img.astype(np.float32)
    for _ in range(3):
        for ax in (0, 1):
            pad = [(0, 0)] * out.ndim; pad[ax] = (r + 1, r)
            p = np.pad(out, pad, mode="edge")
            c = np.cumsum(p, axis=ax)
            sl_hi = [slice(None)] * out.ndim; sl_lo = [slice(None)] * out.ndim
            n = out.shape[ax]
            sl_hi[ax] = slice(2 * r + 1, 2 * r + 1 + n); sl_lo[ax] = slice(0, n)
            out = (c[tuple(sl_hi)] - c[tuple(sl_lo)]) / (2 * r + 1)
    return out

def make_patch(rng, H=128, W=128, d_range=(7, 22), max_cells=None, margin=12):
    # ---- background
    base = rng.uniform(0.12, 0.6)                           # dark to moderate exposure
    tint = np.array([rng.uniform(.9, 1.1), rng.uniform(.95, 1.1), rng.uniform(.8, 1.05)])
    yy, xx = np.mgrid[0:H + 2 * margin, 0:W + 2 * margin].astype(np.float32)
    Hm, Wm = H + 2 * margin, W + 2 * margin
    gx, gy = rng.uniform(-1, 1, 2) * rng.uniform(0, 0.35)
    illum = 1 + gx * (xx / Wm - .5) + gy * (yy / Hm - .5)
    lowf = _fast_blur(rng.normal(0, 1, (Hm // 8 + 2, Wm // 8 + 2)).astype(np.float32), 1.2)
    lowf = np.kron(lowf, np.ones((8, 8)))[:Hm, :Wm] * rng.uniform(0, .06)
    lum = base * (illum + lowf)
    img = lum[..., None] * tint[None, None, :]

    # ---- grid lines (dark, sometimes triple)
    for axis in (0, 1):
        n = rng.choice([0, 0, 1, 1, 2])
        for _ in range(n):
            p = rng.uniform(0, (Hm if axis == 0 else Wm))
            th = rng.uniform(.8, 2.6); dark = rng.uniform(.2, .6)
            offs = [0] if rng.random() < .75 else [-rng.uniform(4, 7), 0, rng.uniform(4, 7)]
            for o in offs:
                coord = (yy if axis == 0 else xx)
                prof = np.exp(-0.5 * ((coord - p - o) / (th / 2)) ** 2)
                img *= (1 - dark * prof)[..., None]

    # ---- debris (labelled negative)
    for _ in range(rng.integers(0, 14)):
        cx, cy = rng.uniform(0, Wm), rng.uniform(0, Hm)
        r = rng.uniform(.6, 3.5) if rng.random() < .8 else rng.uniform(3, 6)
        rr = np.hypot((xx - cx) / r, (yy - cy) / (r * rng.uniform(.5, 1)))
        m = np.clip(1.3 - rr, 0, 1)
        img *= (1 - rng.uniform(.1, .5) * m)[..., None]

    d0 = rng.uniform(*d_range)
    # ---- hard negatives (not cells): cell-sized grey debris without a rim, and air bubbles
    for _ in range(rng.integers(0, 5)):
        cx, cy = rng.uniform(0, Wm), rng.uniform(0, Hm); r = d0 / 2 * rng.uniform(.5, 1.2)
        ang = rng.uniform(0, 3.14); e = rng.uniform(.5, 1)
        sx, sy = xx - cx, yy - cy
        rr = np.hypot((sx * np.cos(ang) + sy * np.sin(ang)) / r, (-sx * np.sin(ang) + sy * np.cos(ang)) / (r * e))
        rr = rr * (1 + .25 * np.sin(5 * np.arctan2(sy, sx) + rng.uniform(0, 6)))   # irregular outline
        img *= (1 - rng.uniform(.15, .55) * np.clip((1.1 - rr) / .3, 0, 1))[..., None]
    if rng.random() < .08:
        cx, cy = rng.uniform(0, Wm), rng.uniform(0, Hm); R = d0 / 2 * rng.uniform(3, 7)
        rr = np.hypot(xx - cx, yy - cy) / R
        img *= (1 - .45 * np.exp(-0.5 * ((rr - 1) / .04) ** 2) + .12 * np.exp(-0.5 * ((rr - .93) / .04) ** 2))[..., None]

    # ---- cells
    density = 0 if rng.random() < .12 else rng.uniform(0, 1.8)   # some empty, noisy patches as negatives
    n_cells = int(density * (Hm * Wm) / (3 * d0) ** 2)
    if max_cells is not None:
        n_cells = min(n_cells, max_cells)
    dead_frac = rng.uniform(0, .45)
    rim_c = .03 + .34 * rng.random() ** 1.6                  # global rim contrast, biased to very transparent
    inv = rng.random() < .25                                # slightly defocused: bright rim, darker core
    clump_p = rng.uniform(.05, .4)
    pts = []
    for _ in range(n_cells * 3):
        if len(pts) >= n_cells:
            break
        r = d0 / 2 * rng.uniform(.85, 1.18)
        if pts and rng.random() < clump_p:                  # clump
            p = pts[rng.integers(len(pts))]; a = rng.uniform(0, 6.283)
            cx, cy = p[0] + np.cos(a) * (p[2] + r) * 1.0, p[1] + np.sin(a) * (p[2] + r) * 1.0
            minf = .95
        else:
            cx, cy = rng.uniform(-r, Wm + r), rng.uniform(-r, Hm + r); minf = 1.25
        if any(np.hypot(cx - q[0], cy - q[1]) < minf * (q[2] + r) * .9 for q in pts):
            continue
        pts.append((cx, cy, r, 1 if rng.random() < dead_frac else 0))
    for cx, cy, r, cls in pts:
        x0, x1 = int(max(0, cx - 2 * r - 2)), int(min(Wm, cx + 2 * r + 3))
        y0, y1 = int(max(0, cy - 2 * r - 2)), int(min(Hm, cy + 2 * r + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        sx, sy = xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy
        ang = rng.uniform(0, 3.14); e = rng.uniform(.88, 1.12)
        u = (sx * np.cos(ang) + sy * np.sin(ang)) / (r * e)
        v = (-sx * np.sin(ang) + sy * np.cos(ang)) / (r / e)
        rr = np.sqrt(u * u + v * v)
        patch = img[y0:y1, x0:x1]
        rimw = rng.uniform(.1, .22)
        rim = np.exp(-0.5 * ((rr - 1) / rimw) ** 2)
        inside = np.clip((1 - rr) / .15, 0, 1)
        if cls == 0:   # live: transparent, slightly bright core, dark rim, faint bright halo
            c = rim_c * rng.uniform(.6, 1.4)
            if not inv:
                core = rng.uniform(-.03, .18) * inside
                halo = rng.uniform(0, .15) * np.exp(-0.5 * ((rr - 1.35) / .18) ** 2)
                f = (1 + core + halo - c * rim)
            else:
                core = -rng.uniform(.02, .14) * inside
                halo = -rng.uniform(0, .08) * np.exp(-0.5 * ((rr - 1.4) / .2) ** 2)
                f = (1 + core + halo + c * .9 * rim)
            patch *= f[..., None]
        else:          # dead: blue stain, darker, weaker refractile rim
            s = rng.uniform(.3, .95)
            blue = np.array([1 - .55 * s, 1 - .42 * s, 1 - .08 * s])
            dark = 1 - rng.uniform(.05, .35) * s
            m = inside[..., None]
            patch *= (1 - m) + m * blue[None, None, :] * dark
            patch *= (1 - rim_c * .6 * rim)[..., None]
            # granular texture
            for _ in range(3):
                gx_, gy_ = rng.uniform(-.6, .6, 2)
                gr = np.hypot(u - gx_, v - gy_)
                patch *= (1 - .15 * s * np.clip(1 - gr / .25, 0, 1))[..., None]
        img[y0:y1, x0:x1] = patch

    # ---- optics + sensor (phone-camera-like: mostly luminance noise, some chroma, demosaic blur, JPEG)
    img = _fast_blur(img, rng.uniform(0, 1.3))
    amp = rng.uniform(.004, .045) * (1.6 if base < .3 and rng.random() < .5 else 1)   # dark photo = high ISO
    lum_n = rng.normal(0, amp, img.shape[:2]) if rng.random() < .7 else rng.uniform(-1.7, 1.7, img.shape[:2]) * amp
    if rng.random() < .5:
        lum_n = _fast_blur(lum_n, rng.uniform(.4, 1.0)) * 1.6     # grainy / blotchy noise
    img = img + lum_n[..., None] + rng.normal(0, rng.uniform(0, .012), img.shape)
    img = np.clip(img, 0, 1)
    img = img[margin:margin + H, margin:margin + W]
    if rng.random() < .6:                                        # the page's auto-levels step
        lum = img.mean(-1); lo, hi = np.quantile(lum, [rng.uniform(0, .02), rng.uniform(.99, 1)])
        if hi - lo > .03: img = np.clip((img - lo) / (hi - lo), 0, 1)
    if rng.random() < .7:
        import io
        from PIL import Image
        b = io.BytesIO(); Image.fromarray((img * 255).astype(np.uint8)).save(b, "JPEG", quality=int(rng.integers(55, 96)))
        img = np.asarray(Image.open(b), np.float32) / 255
    centres = [(cx - margin, cy - margin, cls, r) for cx, cy, r, cls in pts
               if 0 <= cx - margin < W and 0 <= cy - margin < H]
    return img.astype(np.float32), centres

def heatmap(centres, H, W, sigma_scale=0.22):
    """single channel: every cell (live or stained) is a 'cell'"""
    hm = np.zeros((1, H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for cx, cy, cls, r in centres:
        s = max(1.2, 2 * r * sigma_scale)
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
        hm[0] = np.maximum(hm[0], g)
    return hm

if __name__ == "__main__":
    import time
    from PIL import Image
    rng = np.random.default_rng(0)
    t = time.time()
    for i in range(50):
        im, c = make_patch(rng)
    print("50 patches", time.time() - t)
    tiles = []
    for i in range(16):
        im, c = make_patch(rng, d_range=(8, 14))
        tiles.append((im * 255).astype(np.uint8))
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
    Image.fromarray(grid).resize((1024, 1024), Image.NEAREST).save("preview.png")
