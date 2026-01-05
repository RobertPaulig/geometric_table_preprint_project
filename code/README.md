# Geometric Table Snowflakes — Reference Code

Minimal reproducible implementation:
- build Primitive Geometric Table Graph from a window,
- compute normalized Laplacian eigenvalues,
- export an Evidence Pack with SHA256 checksums.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python scripts/run_experiment.py --center 600 --h 200 --K 200 --weight ones --out out/center_600_ones
python scripts/run_experiment.py --center 600 --h 200 --K 200 --weight atan --out out/center_600_atan
```
