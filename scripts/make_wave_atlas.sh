#!/usr/bin/env bash
set -euo pipefail

PDF_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --pdf-only)
      PDF_ONLY=1
      shift
      ;;
    *)
      ;;
  esac
done

if [ "$PDF_ONLY" -eq 0 ]; then
  PYTHONPATH=code python code/scripts/wave_atlas_generate.py
  PYTHONPATH=code python code/scripts/wave_metrics.py
fi

pushd docs >/dev/null
lualatex wave_atlas.tex
lualatex wave_atlas.tex
popd >/dev/null

PDF_PATH="docs/wave_atlas.pdf"
echo "PDF: ${PDF_PATH}"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${PDF_PATH}"
else
  python - <<'PY'
import hashlib
path = "docs/wave_atlas.pdf"
h = hashlib.sha256()
with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(f"{h.hexdigest()}  {path}")
PY
fi
