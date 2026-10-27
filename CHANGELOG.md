# Changelog

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
