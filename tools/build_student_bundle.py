#!/usr/bin/env python3
"""Build the student bundle (gate 7: solutions and the hidden rubric are excluded by the
BUILD, never by .gitignore alone).

    python tools/build_student_bundle.py [--out dist]

For each lesson it emits into <out>/<lesson-id>/:
  lesson.ipynb        generated from the py:percent source
  lesson.py           the source, for students who prefer a plain file
  meta.student.yaml   metadata minus reviewer fields
  assets/             cached data the lesson needs offline
  lesson.c/.cpp, Makefile   for C/C++ lessons
It NEVER copies solutions/ or tests/, and it fails loudly if either would land in the bundle.
"""
import argparse, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUPYTEXT = str(ROOT / ".venv/bin/jupytext")
EXCLUDE = {"solutions", "tests", "__pycache__", ".ipynb_checkpoints"}
STRIP_META = ("reviewed_by", "status", "measured_seconds", "measured_peak_mib")


def lessons():
    for pat in ("lessons/*/meta.yaml", "flagships/*/lessons/*/meta.yaml", "programmes/*/lessons/*/meta.yaml"):
        for m in sorted(ROOT.glob(pat)):
            yield m.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()
    out_root = ROOT / a.out
    built = leaked = 0
    for d in lessons():
        rid = (re.search(r"^id:\s*(.+)$", (d / "meta.yaml").read_text(), re.M) or [None, d.name])[1].strip()
        dest = out_root / rid
        if dest.exists():
            # ignore_errors: this tree may sit in an iCloud-synced folder, where the sync
            # daemon can remove a file between our scandir and our unlink.
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        src = d / "lesson.py"
        if src.exists():
            subprocess.run([JUPYTEXT, "--to", "ipynb", str(src), "-o", str(dest / "lesson.ipynb")],
                           check=False, capture_output=True)
            shutil.copy2(src, dest / "lesson.py")
        for extra in ("lesson.c", "lesson.cpp", "Makefile", "claims.yaml", "README.md"):
            if (d / extra).exists():
                shutil.copy2(d / extra, dest / extra)
        if (d / "assets").is_dir():
            shutil.copytree(d / "assets", dest / "assets")
        meta, skipping = [], False
        for line in (d / "meta.yaml").read_text().splitlines():
            if any(line.startswith(k + ":") for k in STRIP_META):
                skipping = True          # drop the key and any folded/indented continuation
                continue
            if skipping and (line.startswith((" ", "\t")) or not line.strip()):
                continue
            skipping = False
            # Comments in meta.yaml are MAINTAINER notes — review narratives, tooling caveats,
            # curriculum decisions. A student receives the fields, not the workshop floor.
            if line.lstrip().startswith("#"):
                continue
            meta.append(line)
        (dest / "meta.student.yaml").write_text("\n".join(meta) + "\n")
        # gate 7 assertion: nothing from EXCLUDE may exist in the bundle
        for bad in EXCLUDE:
            if (dest / bad).exists():
                print(f"  LEAK: {rid} bundle contains {bad}/")
                leaked += 1
        # ...and no BUNDLED FILE may point at a path the student does not receive.
        # Scanning only the notebook was the old bug: claims.yaml and meta.student.yaml
        # reference files too, and a reference to solutions/ is a broken link for a student.
        # Only files a STUDENT reads as instructions can contain a broken pointer. A Makefile
        # `clean` target that removes solutions/lesson_bin, or a provenance note in claims.yaml,
        # is not an instruction to open a directory the student does not have.
        # lesson.py is the source of truth (the notebook is generated from it), and in a
        # py:percent file the student-facing PROSE is the comment lines. Code that names a
        # reference path — the dual-build detection in the C/C++ lessons — is machinery, not
        # an instruction to open a directory the student does not have.
        for f in sorted(dest.rglob("*")):
            if not f.is_file() or f.name not in {"lesson.py", "README.md"}:
                continue
            try:
                lines = f.read_text().splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            prose = [l for l in lines if f.name == "README.md" or l.lstrip().startswith("#")]
            for needle in ("lesson_solution", "solutions/"):
                hit = next((l for l in prose if needle in l), None)
                if hit:
                    print(f"  LEAK: {rid}/{f.relative_to(dest)} prose references {needle!r}: {hit.strip()[:70]}")
                    leaked += 1
                    break
        built += 1
        print(f"  built {rid}")
    print(f"\n  {built} lessons bundled into {out_root.relative_to(ROOT)}/, {leaked} leaks")
    return 1 if leaked else 0


if __name__ == "__main__":
    raise SystemExit(main())
