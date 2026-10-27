#!/bin/bash
# Start the Cell Counter server. The first run creates a private Python environment (numpy + pillow).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run: installing numpy and pillow…"
  python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt || { echo "Install failed"; exit 1; }
fi
# Apple-silicon Macs: MLX lets CellNet run on the GPU (optional; the server falls back to the CPU without it)
if [ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] && ! .venv/bin/python -c "import mlx.core" 2>/dev/null; then
  echo "Installing MLX so CellNet can use the Mac's GPU…"
  .venv/bin/pip install -q mlx || echo "MLX install failed; using the CPU instead"
fi
if command -v caffeinate >/dev/null; then
  exec caffeinate -i .venv/bin/python server.py "$@"     # keep a Mac awake while serving
fi
exec .venv/bin/python server.py "$@"
