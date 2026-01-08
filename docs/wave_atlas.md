# Wave Atlas (M1)

This is a minimal, reproducible atlas of wave/front visualizations in the
Geometric Table. Build the atlas with a single command:

```bash
./scripts/make_wave_atlas.sh
```

On Windows:

```powershell
.\scripts\make_wave_atlas.ps1
```

PDF-only (use existing artifacts):

```bash
./scripts/make_wave_atlas.sh --pdf-only
```
```powershell
.\scripts\make_wave_atlas.ps1 -PdfOnly
```

Manual steps (if needed):

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
