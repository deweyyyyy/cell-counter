# Training CellNet

CellNet is a small U-Net (≈118k parameters) that predicts a heatmap with a peak at every cell centre. The server finds the peaks, so the model never has to output a count directly.

## Setup

Training uses TensorFlow, which the server does not need:

```bash
cd training
python3 -m venv .venv && source .venv/bin/activate
pip install tensorflow numpy pillow scipy
```

## Files

| File | What it does |
|---|---|
| `gen.py` | Draws simulated 128×128 hemocytometer patches with known cell centres: dim and uneven light, faint or defocused cells, trypan-blue-stained cells, clumps, grid lines, debris, bubbles, phone-camera noise and JPEG artefacts. `python gen.py` writes `preview.png`. |
| `train.py` | Generates a dataset and trains the model. `python train.py 14000 10` means 14,000 patches for 10 epochs. Set `WARM=1` to continue from an existing `cellnet.keras`. |
| `compare.py` | Scores models on the same held-out simulated set: `python compare.py old.keras new.keras` |
| `export.py` | Converts `cellnet.keras` into `cellnet_w.json` for the server: `VERSION="CellNet v0.3" THR=0.85 python export.py` |

When `export.py` finishes, copy `cellnet_w.json` into `../server/` and restart the server.

## Using your own corrected squares

Each file the app saves in `server/training_data/` is one square: the auto-levelled photo (`image`, a data URL), the cell centres you confirmed (`cells`, in image pixels), the square border (`roi`) and the cell diameter. To use them, cut 128×128 patches from these images at the model's scale (cells about 13 px across) and mix them with simulated patches. Hold a few squares back, so you can check the new model against your own hand counts before switching to it.

Files marked `"source": "synthetic"` came from the classic page's simulated sample, not from a microscope. Leave them out of training.
