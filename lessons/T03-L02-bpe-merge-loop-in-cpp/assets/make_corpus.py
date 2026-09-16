#!/usr/bin/env python3
"""Regenerate assets/corpus.txt — the single input both trainers in T03-L02 read.

The corpus is CPython's own standard-library source. It is on every machine that can run
this lesson, so nothing is downloaded, and it is licensed under the PSF License Agreement
(see assets/SOURCE.md). Code is a deliberate choice of text: identifiers, punctuation runs
and indentation give byte-pair merges something with real structure to find.

    python assets/make_corpus.py --bytes 240000 --out assets/corpus.txt

The defaults below are the ones the shipped corpus.txt was built with, so a bare run
reproduces it byte for byte and the sha256 it prints matches the one in SOURCE.md.

The shipped corpus.txt was produced by this script and is byte-identical for every student,
which is what makes the merge list a testable object rather than a machine-dependent one.
"""
from __future__ import annotations

import argparse
import hashlib
import sysconfig
from pathlib import Path

# A fixed, alphabetical slice of the standard library. Every name here has shipped with
# CPython since well before 3.12, so the script reproduces the same corpus on any 3.12
# interpreter. Order is fixed because concatenation order changes the merge list.
MODULES = (
    "argparse.py",
    "ast.py",
    "dataclasses.py",
    "difflib.py",
    "enum.py",
    "functools.py",
    "inspect.py",
    "pathlib.py",
    "random.py",
    "statistics.py",
    "textwrap.py",
    "typing.py",
)


def build(target_bytes: int) -> bytes:
    """Concatenate MODULES until `target_bytes` is reached, cutting on a whitespace byte."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    chunks: list[bytes] = []
    total = 0
    for name in MODULES:
        path = stdlib / name
        if not path.exists():
            continue
        blob = path.read_bytes()
        chunks.append(blob)
        total += len(blob) + 1
        if total >= target_bytes:
            break
    text = b"\n".join(chunks)
    if len(text) <= target_bytes:
        return text
    cut = target_bytes
    while cut > 0 and not text[cut - 1 : cut].isspace():
        cut -= 1
    return text[:cut]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bytes", type=int, default=240_000)
    ap.add_argument("--out", default="corpus.txt")
    a = ap.parse_args()
    data = build(a.bytes)
    out = Path(a.out)
    out.write_bytes(data)
    print(f"wrote {out} — {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
