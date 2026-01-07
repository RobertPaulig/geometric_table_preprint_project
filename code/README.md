# Geometric Table Snowflakes – Experiments (v2)

This code builds a row-projection graph from the Geometric Table and computes
the spectrum of the normalized Laplacian as a "fingerprint" of a center window.

## Core idea
- Rows are integers `n` in a window `[center-h, center+h]`.
- A row `n` connects to a quotient `q = n/k` if `k | n` and `k <= K`.
- Primitive cell filter: keep only factorizations `n = k*q` with `gcd(k,q)=1`.
- Row-projection: connect two rows if they share a quotient `q`, with weight `w(q)`.

## Install
Create a venv and install:
```bash
pip install -r requirements.txt
```

## Run

Examples (Windows PowerShell):

```bash
python scripts/run_experiment.py --center 1000 --h 200 --K 200 --primitive --weight ones --out out/center_1000_rowproj_ones
python scripts/run_experiment.py --center 1000 --h 200 --K 200 --primitive --weight atan --out out/center_1000_rowproj_atan

python scripts/run_experiment.py --center 840 --h 200 --K 200 --primitive --weight ones --out out/center_840_rowproj_ones
python scripts/run_experiment.py --center 840 --h 200 --K 200 --primitive --weight atan --out out/center_840_rowproj_atan

python scripts/run_experiment.py --center 600 --h 200 --K 200 --primitive --weight ones --out out/center_600_rowproj_ones
python scripts/run_experiment.py --center 600 --h 200 --K 200 --primitive --weight atan --out out/center_600_rowproj_atan
```

## Outputs per run

* params.json
* nodes.json
* edges.csv
* eigenvalues.json
* metrics.json
* checksums.sha256

## Build PDF
Without `latexmk`:
```bash
./scripts/build_pdf.sh
```
On Windows:
```powershell
.\scripts\build_pdf.ps1
```
See `BUILD.md` for details.

## Wave Atlas
Generate Wave Atlas artifacts (from repo root):
```bash
PYTHONPATH=code python code/scripts/wave_atlas_generate.py \
  --N 3000 --K 120 --H 220 --step 60 \
  --diag-N 60 --diag-N 420 --diag-N 2520 --diag-N 27720 \
  --out-dir out/wave_atlas
```
Build the PDF:
```bash
cd docs
lualatex wave_atlas.tex
lualatex wave_atlas.tex
```
