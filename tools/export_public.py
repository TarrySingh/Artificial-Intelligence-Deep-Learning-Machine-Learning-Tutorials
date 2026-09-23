#!/usr/bin/env python3
"""Write the student version of Synapsa Commons into a clone of the public repository.

    python tools/export_public.py <path-to-public-clone>

This repository is the source: it holds every lesson's worked solution, rubric tests and
authoring masters, and it is never pushed anywhere. The public repository receives what a student
may have -- the notebooks, lesson files, assets, course maps, tools and licence -- and nothing
that answers an exercise. This tool is the only way lessons reach it:

  * exports the COMMITTED tree (HEAD), so nothing half-edited leaves;
  * drops every lesson's solutions/, tests/ and authoring/ directories;
  * appends to .gitignore the block that stops answers being added in the clone by hand;
  * deletes from the clone anything the source no longer ships (never touching its .git);
  * refuses to write at all if an answer path would leave, and prints what changed.

It never commits or pushes. Review the diff in the clone, commit, push.
"""
import re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANSWERS = re.compile(r"^(lessons/[^/]+|flagships/[^/]+/lessons/[^/]+|programmes/[^/]+/lessons/[^/]+)"
                     r"/(solutions|tests|authoring)/")
GITIGNORE_BLOCK = """
# Answers never live in this repository: worked solutions, rubric tests and authoring masters
# stay in the private source and come with enrolment on Synapsa.
/lessons/*/solutions/
/lessons/*/tests/
/lessons/*/authoring/
/flagships/*/lessons/*/solutions/
/flagships/*/lessons/*/tests/
/flagships/*/lessons/*/authoring/
/programmes/*/lessons/*/solutions/
/programmes/*/lessons/*/tests/
/programmes/*/lessons/*/authoring/
"""


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    dest = Path(sys.argv[1]).resolve()
    if not (dest / ".git").exists():
        raise SystemExit(f"{dest} is not a git clone of the public repository")
    if dest == ROOT or ROOT in dest.parents:
        raise SystemExit("refusing to export into the source itself")
    if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                      capture_output=True, text=True).stdout.strip():
        print("  note: uncommitted changes in the source are NOT exported (HEAD is)")

    with tempfile.TemporaryDirectory(prefix="commons-export-") as tmp:
        archive = Path(tmp) / "head.tar"
        subprocess.run(["git", "-C", str(ROOT), "archive", "-o", str(archive), "HEAD"], check=True)
        out = Path(tmp) / "tree"
        with tarfile.open(archive) as t:
            names = [m.name for m in t.getmembers() if m.isfile()]
            keep = [n for n in names if not ANSWERS.match(n)]
            t.extractall(out, members=[m for m in t.getmembers() if m.name in keep or m.isdir()],
                         filter="data")
        # anything answer-shaped that survived is a bug in this tool: stop before writing
        leaked = [p.relative_to(out).as_posix() for p in out.rglob("*")
                  if p.is_file() and ANSWERS.match(p.relative_to(out).as_posix())]
        if leaked:
            raise SystemExit("REFUSING: answers would leave the source:\n  " + "\n  ".join(leaked))
        for d in [p for p in out.rglob("*") if p.is_dir()]:
            if not any(d.iterdir()):
                d.rmdir()
        gi = out / ".gitignore"
        gi.write_text(gi.read_text().rstrip("\n") + "\n" + GITIGNORE_BLOCK)

        shipped = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
        tracked = set(subprocess.run(["git", "-C", str(dest), "ls-files"], capture_output=True,
                                     text=True).stdout.split("\n")) - {""}
        removed = sorted(tracked - shipped)
        for rel in removed:
            (dest / rel).unlink(missing_ok=True)
        added = changed = 0
        for rel in sorted(shipped):
            src, dst = out / rel, dest / rel
            if not dst.exists():
                added += 1
            elif dst.read_bytes() != src.read_bytes():
                changed += 1
            else:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        print(f"  exported HEAD {subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()}"
              f" -> {dest}\n  {len(shipped)} files shipped, {len(names) - len(keep)} answer files held back;"
              f" {added} added, {changed} changed, {len(removed)} removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
