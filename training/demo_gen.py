"""NumPy port of the public page's simulated sample (makeSample in public.html)."""
import numpy as np

def make_sample(seed):
    S, A, B = 1200, 50, 1150; L = B - A
    sd = [seed]
    def rnd():
        sd[0] = (sd[0] * 16807) % 2147483647; return sd[0] / 2147483647
    base = .26 + rnd() * .1; tint = np.array([1, 1.03, .9], np.float32)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    il = 1 + .25 * (1 - np.hypot((xx - S * .45) / S, (yy - S * .42) / S) * 1.8)
    px = (base * il)[..., None] * tint
    def line(p, vert, th, dark):
        for o in range(-4, 5):
            f = 1 - dark * np.exp(-.5 * (o / (th / 2)) ** 2); q = int(round(p + o))
            if 0 <= q < S:
                if vert: px[:, q] *= f
                else: px[q, :] *= f
    for i in range(1, 4): line(A + i * L / 4, 1, 1.4, .45); line(A + i * L / 4, 0, 1.4, .45)
    for p in (A, B):
        for o in (-6, 0, 6): line(p + o, 1, 1.3, .5); line(p + o, 0, 1.3, .5)
    for _ in range(160):
        cx, cy = rnd() * S, rnd() * S; r = .6 + rnd() * (2 if rnd() < .85 else 4); k = .15 + rnd() * .4
        x0, x1, y0, y1 = int(cx - r - 1), int(cx + r + 2), int(cy - r - 1), int(cy + r + 2)
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(S, x1), min(S, y1)
        if x1 <= x0 or y1 <= y0: continue
        m = np.clip(1.3 - np.hypot(xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy) / r, 0, 1)
        px[y0:y1, x0:x1] *= (1 - k * m)[..., None]
    rimC = .16; dead_frac = .04 + rnd() * .16
    pts = []
    R = lambda: 4.6 + rnd() * 1.4
    free = lambda x, y, r, k: all(np.hypot(p[0] - x, p[1] - y) > k * (p[2] + r) for p in pts)
    i = 0
    while i < 16:
        r = R(); q = A + 60 + rnd() * (L - 120); on = (-1 if i % 2 else 1) * rnd() * r * .6
        x, y, c = [(q, A + on, 1), (A + on, q, 1), (q, B + on, 0), (B + on, q, 0)][i % 4]
        if free(x, y, r, 1.3): pts.append([x, y, r, c]); i += 1
    need = 110 + int(rnd() * 150) - sum(1 for p in pts if p[3]); placed = tries = 0
    while placed < need and tries < 20000:
        tries += 1; r = R()
        if len(pts) > 20 and rnd() < .15:
            p = pts[int(rnd() * len(pts))]; a = rnd() * 6.28
            x = p[0] + np.cos(a) * (p[2] + r) * 1.02; y = p[1] + np.sin(a) * (p[2] + r) * 1.02
            if not free(x, y, r, .97): continue
        else:
            x = A + rnd() * L; y = A + rnd() * L
            if not free(x, y, r, 1.4): continue
        if x < A + r + 3 or y < A + r + 3 or x > B - r - 3 or y > B - r - 3: continue
        pts.append([x, y, r, 1]); placed += 1
    for i in range(14):
        r = R(); side = i % 4; q = 60 + rnd() * (S - 120); o = A - r - 14
        x, y = [(q, o), (q, S - o), (o, q), (S - o, q)][side]; pts.append([x, y, r, 0])
    out = []
    for p in pts:
        cx, cy, r, c = p; dead = rnd() < dead_frac
        a = rnd() * 3.14; e = .9 + rnd() * .2; rimw = .13 + rnd() * .06; cc = rimC * (.6 + rnd() * .8); core = rnd() * .1; s = .45 + rnd() * .45
        x0, x1, y0, y1 = max(0, int(np.floor(cx - 2 * r))), min(S, int(cx + 2 * r) + 1), max(0, int(np.floor(cy - 2 * r))), min(S, int(cy + 2 * r) + 1)
        if x1 > x0 and y1 > y0:
            dx, dy = xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy
            u = (dx * np.cos(a) + dy * np.sin(a)) / (r * e); v = (-dx * np.sin(a) + dy * np.cos(a)) / (r / e); rr = np.hypot(u, v)
            msk = rr <= 2
            rim = np.exp(-.5 * ((rr - 1) / rimw) ** 2); ins = np.clip((1 - rr) / .15, 0, 1)
            P = px[y0:y1, x0:x1]
            if not dead:
                f = 1 + core * ins + .08 * np.exp(-.5 * ((rr - 1.35) / .18) ** 2) - cc * rim
                P[msk] *= f[msk][:, None]
            else:
                bl = np.array([1 - .5 * s, 1 - .38 * s, 1 - .06 * s]) * (1 - .2 * s)
                f3 = (1 - ins)[..., None] + ins[..., None] * bl
                P[msk] *= f3[msk]; f = 1 - cc * .6 * rim; P[msk] *= f[msk][:, None]
        out.append((cx, cy, r, c, dead))
    k = np.array([[.5, 1, .5], [1, 2, 1], [.5, 1, .5]], np.float32); k /= k.sum()
    pad = np.pad(px, ((1, 1), (1, 1), (0, 0)), mode="edge"); tmp = np.zeros_like(px)
    for dy in range(3):
        for dx in range(3): tmp += k[dy, dx] * pad[dy:dy + S, dx:dx + S]
    rs = np.random.default_rng(seed); nz = (rs.random((S, S)) - .5) * .05
    img = np.clip(tmp + nz[..., None], 0, 1).astype(np.float32)
    return img, out   # out: (x, y, r, counted, dead)
