#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Iterable

from wave_atlas_io import open_binary


def sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_decompressed_gz(path_gz: Path) -> str:
    h = hashlib.sha256()
    with open_binary(path_gz, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_paths(args: argparse.Namespace) -> Iterable[Path]:
    for raw in args.paths:
        p = Path(raw)
        if any(ch in raw for ch in ["*", "?", "["]):
            yield from (Path(".").glob(raw))
        else:
            yield p


def repack_csv_to_gz(path_csv: Path, *, compresslevel: int) -> tuple[Path, int]:
    if path_csv.suffix != ".csv":
        raise ValueError(f"expected .csv: {path_csv}")
    out_gz = Path(str(path_csv) + ".gz")
    out_gz.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("rb") as src, open_binary(out_gz, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return out_gz, out_gz.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministically gzip .csv files to .csv.gz (mtime=0).")
    ap.add_argument("paths", nargs="+", help="File paths or globs (relative to repo root).")
    ap.add_argument("--min-bytes", type=int, default=0, help="Skip files smaller than this size.")
    ap.add_argument("--delete-original", action="store_true", help="Delete the original .csv after creating .csv.gz.")
    args = ap.parse_args()

    n = 0
    for p in iter_paths(args):
        if not p.exists() or not p.is_file():
            continue
        if p.suffix != ".csv":
            continue
        if p.stat().st_size < args.min_bytes:
            continue

        gz = Path(str(p) + ".gz")
        if gz.exists():
            print(f"SKIP (exists): {gz}")
            continue

        orig_sha = sha256_bytes(p)
        out_gz, out_sz = repack_csv_to_gz(p, compresslevel=9)
        dec_sha = sha256_decompressed_gz(out_gz)
        if orig_sha != dec_sha:
            raise RuntimeError(f"sha mismatch after gzip: {p} -> {out_gz}")

        if args.delete_original:
            p.unlink()
        n += 1
        print(f"OK: {p} -> {out_gz} ({out_sz} bytes)")

    print(f"Done: {n} file(s)")


if __name__ == "__main__":
    main()

