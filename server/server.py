#!/usr/bin/env python3
"""Cell Counter server: serves the phone app and runs CellNet.

  python3 server.py            # this computer only: http://localhost:8000
  python3 server.py --lan      # also phones on the same network: http://<this-computer's-IP>:8000 (no login!)

Endpoints
  GET  /              the phone page
  GET  /api/info      model version, default threshold
  POST /api/count     body = JPEG/PNG of the frame (+margin), ?diam=<cell diameter px in that image>
  POST /api/training  body = JSON export from the page; saved under training_data/
  POST /api/runs      a named counting round: report PNG + each square's image; saved under reports/
  GET  /api/runs      saved rounds, newest first
  GET  /reports/...   saved report files, and reports/runs.csv (all rounds, opens in Excel)
  GET  /api/lab       the lab log: every count, passage, treatment, harvest, culture and compound
  POST /api/lab/photo?id=... store a photo for a note (GET /lab/photos/<id>.jpg)
  POST /api/lab/event add one lab-log event (idempotent by its "id", so offline phones can resend safely)
  GET  /sw.js, /manifest.webmanifest   let the page open and record entries without internet
"""
import argparse, base64, csv, io, json, os, platform, re, sys, threading, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from PIL import Image
from cellnet_np import CellNet

HERE = os.path.dirname(os.path.abspath(__file__))
NET = CellNet(os.path.join(HERE, "cellnet_w.json"))
LOCK = threading.Lock()                     # one inference at a time; others wait their turn
DATA = os.path.join(HERE, "training_data"); os.makedirs(DATA, exist_ok=True)
REPORTS = os.path.join(HERE, "reports"); os.makedirs(REPORTS, exist_ok=True)
CSV_PATH = os.path.join(REPORTS, "runs.csv")
FILE_LOCK = threading.Lock()
LAB = os.path.join(HERE, "lab"); os.makedirs(LAB, exist_ok=True)
LAB_LOG = os.path.join(LAB, "events.jsonl")
LAB_LOCK = threading.Lock()
EVENT_TYPES = {"count", "passage", "treat", "harvest", "culture", "compound", "note", "discard", "delete", "protein", "bca", "checklist", "growth"}
PHOTOS = os.path.join(LAB, "photos")
PHOTO_ID = re.compile(r"[A-Za-z0-9_-]{6,64}")

def photo_add(pid, raw):
    """Store one photo as lab/photos/<id>.jpg. Re-sending the same id is harmless."""
    if not PHOTO_ID.fullmatch(pid or ""): raise ValueError("bad photo id")
    path = os.path.join(PHOTOS, pid + ".jpg")
    try:
        im = Image.open(io.BytesIO(raw)); fmt = im.format; im.load()
    except Exception: raise ValueError("not an image")
    with LAB_LOCK:
        if os.path.exists(path): return False
        os.makedirs(PHOTOS, exist_ok=True)
        if fmt == "JPEG": open(path, "wb").write(raw)
        else: im.convert("RGB").save(path, "JPEG", quality=88)
    return True

def lab_events():
    if not os.path.exists(LAB_LOG): return []
    out = []
    for line in open(LAB_LOG, encoding="utf-8"):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def lab_add(ev, who):
    """Append one event; an event whose id is already stored is ignored (safe to resend)."""
    if not isinstance(ev, dict) or ev.get("type") not in EVENT_TYPES: raise ValueError("unknown event type")
    eid = str(ev.get("id", ""))
    if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", eid): raise ValueError("bad event id")
    raw = json.dumps(ev, ensure_ascii=False)
    if len(raw) > 200_000: raise ValueError("event too large")
    with LAB_LOCK:
        if any(e.get("id") == eid for e in lab_events()): return False
        ev = dict(ev, by=who, received=datetime.now().isoformat(timespec="seconds"))
        os.makedirs(LAB, exist_ok=True)
        with open(LAB_LOG, "a", encoding="utf-8") as f: f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    return True
def _page(name):
    """Pages are read on each request, so replacing a file needs no restart."""
    return open(os.path.join(HERE, name), "rb").read()

def _strip_old(p):
    # the old page lives at /old; its relative API calls may arrive as /old/api/... or /old/reports/...
    return p[4:] if p.startswith("/old/api/") or p.startswith("/old/reports/") else p
MAX_BODY = 30 * 1024 * 1024
MAX_PIXELS = 9_000_000                      # after rescaling to the model's cell size

def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)

def pick_backend():
    """Time the CPU (NumPy) and Apple-GPU (MLX) paths on one test square and keep the faster."""
    probe = np.random.default_rng(0).random((640, 640, 3), dtype=np.float32)
    def bench():
        NET.heatmap(probe); t = time.time(); NET.heatmap(probe); return time.time() - t
    t_cpu = bench(); NET.backend = "NumPy (CPU)"
    if os.environ.get("CELLNET_CPU") or not NET.use_mlx():
        NET.mx = None; NET.backend = "NumPy (CPU)"
        return f"NumPy (CPU) {t_cpu*1000:.0f} ms/test"
    try:
        t_gpu = bench()
    except Exception as e:
        NET.mx = None; NET.backend = "NumPy (CPU)"
        return f"NumPy (CPU) {t_cpu*1000:.0f} ms/test — MLX failed: {e!r}"
    if t_gpu < t_cpu:
        return f"{NET.backend} {t_gpu*1000:.0f} ms/test (CPU {t_cpu*1000:.0f} ms)"
    NET.mx = None; NET.backend = "NumPy (CPU)"
    return f"NumPy (CPU) {t_cpu*1000:.0f} ms/test (MLX {t_gpu*1000:.0f} ms, slower)"

class H(BaseHTTPRequestHandler):
    server_version = "CellCounter/1"

    def who(self):  # set by Cloudflare Access; absent on direct LAN use
        return self.headers.get("Cf-Access-Authenticated-User-Email", "local")

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str): body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError): pass   # the phone cancelled or left; nothing to do

    def log_message(self, *a):  # keep the console readable
        pass

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY: raise ValueError("body size")
        return self.rfile.read(n)

    def do_GET(self):
        p = _strip_old(self.path.split("?")[0])
        if p in ("/", "/index.html"): return self.send(200, _page("app.html"), "text/html; charset=utf-8")
        if p == "/sw.js": return self.send(200, _page("sw.js"), "text/javascript; charset=utf-8")
        if p == "/manifest.webmanifest": return self.send(200, _page("manifest.webmanifest"), "application/manifest+json")
        if p == "/api/lab": return self.send(200, {"events": lab_events(), "now": datetime.now().astimezone().isoformat(timespec="seconds")})
        if p.startswith("/lab/photos/"):
            name = p[len("/lab/photos/"):]
            path = os.path.join(PHOTOS, name)
            if name.endswith(".jpg") and PHOTO_ID.fullmatch(name[:-4]) and os.path.isfile(path):
                return self.send(200, open(path, "rb").read(), "image/jpeg")
            return self.send(404, {"error": "not found"})
        if p in ("/old", "/old/", "/old/index.html"):
            classic = "classic.html" if os.path.exists(os.path.join(HERE, "classic.html")) else "index.html"
            return self.send(200, _page(classic), "text/html; charset=utf-8")
        if p == "/api/info": return self.send(200, {"version": NET.version, "thr": NET.thr, "target": NET.target, "engine": NET.backend, "host": {"Darwin": "Mac", "Windows": "PC"}.get(platform.system(), "Server")})
        if p == "/api/runs": return self.send(200, {"runs": list_runs()})
        if p == "/reports/runs.csv":
            if not os.path.exists(CSV_PATH): return self.send(404, {"error": "no rounds saved yet"})
            body = open(CSV_PATH, "rb").read()
            self.send_response(200); self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="cellcounts.csv"')
            self.send_header("Content-Length", str(len(body))); self.end_headers(); return self.wfile.write(body)
        if p.startswith("/reports/"): return self.report_file(p[len("/reports/"):])
        self.send(404, {"error": "not found"})

    def do_POST(self):
        p, _, q = self.path.partition("?")
        p = _strip_old(p)
        args = dict(kv.split("=", 1) for kv in q.split("&") if "=" in kv)
        try:
            if p == "/api/count": return self.count(args)
            if p == "/api/training": return self.training(args)
            if p == "/api/runs": return self.save_run()
            if p == "/api/lab/photo":
                new = photo_add(args.get("id", ""), self.body())
                if new: log(f"photo  {self.who():30s} {args.get('id')}.jpg")
                return self.send(200, {"ok": True, "new": new})
            if p == "/api/lab/event":
                ev = json.loads(self.body()); new = lab_add(ev, self.who())
                if new: log(f"lab    {self.who():30s} {ev.get('type'):8s} {str(ev.get('title') or ev.get('name') or '')[:50]}")
                return self.send(200, {"ok": True, "new": new})
            self.send(404, {"error": "not found"})
        except ValueError as e:
            self.send(400, {"error": str(e)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            log("ERROR", repr(e)); self.send(500, {"error": "server error"})

    def count(self, args):
        diam = float(args.get("diam", "0"))
        if not 2 <= diam <= 200: raise ValueError("diam must be 2..200")
        im = Image.open(io.BytesIO(self.body())).convert("RGB")
        k = NET.target / diam
        w, h = max(8, round(im.width * k)), max(8, round(im.height * k))
        if w * h > MAX_PIXELS: raise ValueError("image too large for this cell size")
        if (w, h) != im.size: im = im.resize((w, h), Image.BILINEAR)
        rgb = np.asarray(im, np.float32) / 255
        t = time.time()
        with LOCK:
            hm = NET.heatmap(rgb)
        pts = NET.peaks(hm, 0.3)                   # low cut; the page applies its own threshold
        secs = time.time() - t
        # back to the uploaded image's pixel grid
        out = [[round((x + .5) / k - .5, 1), round((y + .5) / k - .5, 1), round(p, 3)] for x, y, p in pts]
        log(f"count  {self.who():30s} {w}x{h}px  {sum(1 for o in out if o[2] > NET.thr)} cells  {secs:.2f}s  [{NET.backend}]")
        self.send(200, {"points": out, "secs": secs, "version": NET.version})

    def save_run(self):
        folder, meta = _save_run(self)
        log(f"round  {self.who():30s} {meta.get('n')}/{meta.get('layout')} squares  -> reports/{folder}")
        self.send(200, {"ok": True, "folder": folder})

    def report_file(self, rel):
        full = os.path.realpath(os.path.join(REPORTS, rel))
        if not full.startswith(os.path.realpath(REPORTS) + os.sep) or not os.path.isfile(full):
            return self.send(404, {"error": "not found"})
        ext = os.path.splitext(full)[1].lower()
        ctype = {".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json"}.get(ext)
        if not ctype: return self.send(404, {"error": "not found"})
        self.send(200, open(full, "rb").read(), ctype)

    def training(self, args):
        d = json.loads(self.body())
        if d.get("format") != "cellcount-v1" or not str(d.get("image", "")).startswith("data:image/"):
            raise ValueError("not a cellcount-v1 export")
        d["submitted_by"] = self.who(); d["received"] = datetime.now().isoformat(timespec="seconds")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", args.get("name", "cells.json"))[:80]
        who = re.sub(r"[^A-Za-z0-9]", "_", self.who())[:40]
        path = os.path.join(DATA, f"{who}__{safe}")
        json.dump(d, open(path, "w"))
        total = len([f for f in os.listdir(DATA) if f.endswith(".json")])
        log(f"train  {self.who():30s} {len(d.get('cells', []))} cells  -> {os.path.basename(path)}  (total {total})")
        self.send(200, {"ok": True, "total": total})

def list_runs(limit=100):
    out = []
    for d in sorted(os.listdir(REPORTS), reverse=True):
        f = os.path.join(REPORTS, d, "run.json")
        if os.path.isfile(f):
            try:
                m = json.load(open(f)); m["report_url"] = f"reports/{d}/report.png"; out.append(m)
            except Exception: pass
        if len(out) >= limit: break
    return out

def _decode(data_url, kind):
    head, _, b64 = str(data_url).partition(",")
    if not head.startswith("data:image/" + kind): raise ValueError("bad image")
    return base64.b64decode(b64)

def _save_run(handler):
    d = json.loads(handler.body()); meta = d.get("meta") or {}
    if not isinstance(meta, dict) or not str(meta.get("name", "")).strip(): raise ValueError("round needs a name")
    meta["operator"] = handler.who()
    png = _decode(d.get("report_png"), "png")
    slug = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "-", meta["name"]).strip(" .-")[:60] or "round"
    with FILE_LOCK:
        folder = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + slug
        path = os.path.join(REPORTS, folder); n = 2
        while os.path.exists(path): path = os.path.join(REPORTS, f"{folder}-{n}"); n += 1
        folder = os.path.basename(path); os.makedirs(path)
        open(os.path.join(path, "report.png"), "wb").write(png)
        for k, v in (d.get("images") or {}).items():
            if v and re.fullmatch(r"(b-)?(tl|tr|bl|br)", k):
                open(os.path.join(path, f"{k}.jpg"), "wb").write(_decode(v, "jpeg"))
        json.dump(meta, open(os.path.join(path, "run.json"), "w"), ensure_ascii=False, indent=1)
        row = next(csv.reader([d.get("csv_row", "")]), None); head = d.get("csv_head")
        if row and isinstance(head, list) and len(row) == len(head):
            if "operator" in head: row[head.index("operator")] = meta["operator"]
            new = not os.path.exists(CSV_PATH)
            with open(CSV_PATH, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
                w = csv.writer(f)
                if new: w.writerow(head)
                w.writerow(row)
    return folder, meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lan", action="store_true", help="listen on all network interfaces so phones on the same network can connect (no login)")
    a = ap.parse_args()
    host = "0.0.0.0" if a.lan else "127.0.0.1"
    srv = ThreadingHTTPServer((host, a.port), H)
    log("checking the fastest way to run CellNet on this machine…")
    try: log("engine:", pick_backend())
    except Exception as e: log("engine check failed, using CPU:", repr(e)); NET.mx = None; NET.backend = "NumPy (CPU)"
    log(f"CellNet {NET.version} ready on http://{host}:{a.port}  (Ctrl+C to stop)")
    if a.lan:
        log("LAN mode: anyone on this network can open it without logging in")
        try:
            import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("10.255.255.255", 1)); ip = s.getsockname()[0]; s.close()
            log(f"on your phone, open http://{ip}:{a.port}")
        except Exception: pass
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
