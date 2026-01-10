# `out/wave_atlas/` layout

This folder is the canonical home for Wave Atlas milestone artifacts used by the PDF/TeX and for reproducible runs.

## Conventions

- Milestone outputs live under `out/wave_atlas/mXX/` (or a labeled subfolder inside it).
- A few baseline figures for early sections are kept directly under `out/wave_atlas/` and referenced by `docs/wave_atlas.tex`.
- Large tabular artifacts should be stored as `*.csv.gz` (gzip) to keep the repo compact.
  - Reader scripts should accept both `*.csv` and `*.csv.gz` (fallback to `.gz` if the plain file is missing).
  - Writer scripts should prefer emitting `*.csv.gz` by default.

