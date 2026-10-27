# Changelog

## v0.3.0 — 2026-10-27

- **CellNet v0.3:** far fewer false marks on grainy empty background. v0.2 was fine-tuned on simulated patches, simulated full squares with known answers, and hand-corrected squares (only the area inside the drawn square border counts in the loss).
  - On 3 held-out simulated squares (636 cells) at the default strictness of 0.85: false marks dropped from 41 to 8, and every cell was found (v0.2 missed 1).
  - On 3 held-out hand-corrected squares (20 cells): false marks dropped from 8 to 0, and every cell was found.
  - No change on the general simulated test set (F1 0.797 → 0.794, within noise).
- `export.py` and `compare.py` load models without their training loss, so models trained with a custom loss export cleanly.


## v0.2.0 — 2026-10-27

First public release.

- Phone-first app in Thai and English: start a round, count square by square, review, round summary.
- CellNet v0.2 runs on your own computer: the GPU through MLX on Apple-silicon Macs, NumPy on the CPU elsewhere. The server picks the faster one at startup.
- Edge rule from a square border drawn once and reused for the following squares.
- Review tools: add or remove cells, undo, pinch-zoom, and a strictness slider that recounts instantly.
- 4 or 8 squares per round, with mean ± SD, a CV quality tag, per-chamber means and cells/mL.
- PNG report with every square and a per-square table, plus `runs.csv` covering every round.
- Corrected squares can be saved as training data. Training scripts are in `training/`.

Known limitation: CellNet v0.2 was trained on simulated images only and can mark grainy empty background as cells.
