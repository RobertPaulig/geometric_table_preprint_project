from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import IO, AnyStr, Union


PathLike = Union[str, Path]


def resolve_existing_or_gz(path: PathLike) -> Path:
    p = Path(path)
    if p.exists():
        return p
    gz = Path(str(p) + ".gz")
    if gz.exists():
        return gz
    if p.suffix == ".gz":
        ungz = Path(str(p)[: -len(".gz")])
        if ungz.exists():
            return ungz
    return p


def open_text(path: PathLike, mode: str = "rt", encoding: str = "utf-8", newline: str = "") -> IO[str]:
    p = Path(path)
    if p.suffix == ".gz":
        if "r" in mode:
            return gzip.open(p, mode, encoding=encoding, newline=newline)
        raw = p.open("wb")
        gz = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
        return io.TextIOWrapper(gz, encoding=encoding, newline=newline)
    return p.open(mode, encoding=encoding, newline=newline)


def open_binary(path: PathLike, mode: str = "rb") -> IO[bytes]:
    p = Path(path)
    if p.suffix == ".gz":
        if "r" in mode:
            return gzip.open(p, mode)
        raw = p.open("wb")
        return gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
    return p.open(mode)
