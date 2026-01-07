# Wave Atlas (M1)

This is a minimal, reproducible atlas of wave/front visualizations in the
Geometric Table. Generate artifacts and build the PDF:

```bash
PYTHONPATH=code python code/scripts/wave_atlas_generate.py \
  --N 3000 --K 120 --H 220 --step 60 \
  --diag-N 60 --diag-N 420 --diag-N 2520 --diag-N 27720 \
  --out-dir out/wave_atlas

cd docs
lualatex wave_atlas.tex
lualatex wave_atlas.tex
```

Artifacts are written under `out/wave_atlas/`.
