# Changelog

## v0.4.0 — 2026-10-29

- **Lab log** on a new home screen: hand-entered counts, sub-passage with pipetting volumes per well/dish/flask, treatments (compound library, multi-compound groups, stock µL per group, matched vehicle %, plate map, harvest timer), culture list, full log with filters and CSV export.
- **Works offline:** the app is cached on the phone, and log entries made without internet are sent when it's back.
- Camera counts are added to the log automatically, with a link to their report; the round summary has a "use this count for a passage" button.
- An unfinished camera round shows on the home screen with a Continue button.
- `install-service.sh` runs the server as a macOS background service with a log file.
- **Protein harvest:** tubes per group or per well, lysis buffer, volume, storage and box; BCA values can be added later.
- **Notes with photos:** attach photos from the camera or gallery, link a note to an experiment or culture; works offline.
- Date and time fields on passage, hand count and culture forms, for logging past work.
- Detail page for every log entry, with a labelled plate map and links between related entries.
- "In the incubator" on the home screen, with plate maps for plates under treatment.
- Editable checklists and steps for counting, passage, treatment and protein harvest.
- Plate maps fill column by column (1A, 1B, 2A, …) and well positions are saved with each group.
- People: entries record who did them; passage vessels can belong to a person; log filter and badges by person.
- Easier-to-read log cards and detail pages (one group, tube or note per line).
- Growth prediction in the incubator view, with editable doubling time, lag and passage threshold; passage reminders under To do.
- A passage can take several source vessels.
- PDF / print for checklists, entry details and the log; count entries show their report image.
- Protocols screen and illustrated steps (icons, highlighted numbers, cautions, tips, timers), with a full-screen step-by-step mode.
- PDF export waits for every photo to load before printing.

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
