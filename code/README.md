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
