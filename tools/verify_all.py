#!/usr/bin/env python3
"""Repo-wide gate runner. Trusts nothing an author or reviewer reported.

    python tools/verify_all.py [--write-back]

For every directory holding a meta.yaml under lessons/ or flagships/*/lessons/:
  gate 9+10  tools/execute.py            -> runs clean inside the declared tier and budget
  gate 6a    tools/grade.py --solution   -> the reference implementation scores 100%
  gate 4     tools/grade.py              -> the student file does NOT score 100% (stubs are real)
  gate 7     jupytext --to ipynb         -> the source converts to a student notebook
Exits non-zero if any lesson fails any gate.
"""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")
JUPYTEXT = str(ROOT / ".venv/bin/jupytext")


def lessons():
    seen = set()
    for pat in ("lessons/*/meta.yaml", "flagships/*/lessons/*/meta.yaml", "programmes/*/lessons/*/meta.yaml"):
        for m in sorted(ROOT.glob(pat)):
            if m.parent not in seen:
                seen.add(m.parent)
                yield m.parent


def run(cmd, timeout=1800):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def score(out):
    m = re.search(r"score:\s*(\d+)/(\d+)\s*\((\d+)%\)", out)
    return int(m.group(3)) if m else None


def meta_field(d, key):
    f = d / "meta.yaml"
    if f.exists():
        m = re.search(rf"^{key}:\s*(.+?)\s*$", f.read_text(), re.M)
        if m:
            return re.sub(r"\s+#.*$", "", m.group(1)).strip().strip("\"'")
    return "?"


def main() -> int:
    wb = ["--write-back"] if "--write-back" in sys.argv else []
    rows, failed = [], 0
    for d in lessons():
        rid = meta_field(d, "id")
        lang = meta_field(d, "language")
        tier = meta_field(d, "tier")
        rc_exec, out_exec = run([PY, "tools/execute.py", str(d), *wb])
        secs = (re.search(r"wall=([\d.]+)s", out_exec) or [None, "?"])[1]
        peak = (re.search(r"peak=(\d+)MiB", out_exec) or [None, "?"])[1]
        _, out_sol = run([PY, "tools/grade.py", str(d), "--solution"])
        _, out_stu = run([PY, "tools/grade.py", str(d)])
        sol, stu = score(out_sol), score(out_stu)
        nb = "-"
        rc_nb, _ = run([JUPYTEXT, "--to", "ipynb", str(d / "lesson.py"), "-o", f"/tmp/{rid}.ipynb"])
        if rc_nb == 0:
            try:
                import json
                nb = str(len(json.load(open(f"/tmp/{rid}.ipynb"))["cells"]))
            except Exception:
                nb = "?"
        ok = (rc_exec == 0) and (sol == 100) and (stu is not None and stu < 100) and (rc_nb == 0)
        if not ok:
            failed += 1
        rows.append((("PASS" if ok else "FAIL"), rid, lang, tier, secs, peak,
                     f"{sol}%" if sol is not None else "-", f"{stu}%" if stu is not None else "-", nb,
                     "" if ok else f"exec_rc={rc_exec} nb_rc={rc_nb}"))

    w = max((len(r[1]) for r in rows), default=8) + 1
    print(f"\n  {'':5} {'lesson':<{w}} {'lang':<7} {'tier':<9} {'secs':>6} {'MiB':>6} {'ref':>5} {'stub':>5} {'cells':>5}")
    for r in rows:
        print(f"  [{r[0]}] {r[1]:<{w}} {r[2]:<7} {r[3]:<9} {r[4]:>6} {r[5]:>6} {r[6]:>5} {r[7]:>5} {r[8]:>5}  {r[9]}")
    print(f"\n  {len(rows) - failed}/{len(rows)} lessons pass every gate\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
