#!/usr/bin/env python3
"""Portability gate (gate 13). Stdlib only in this process; the notebooks run elsewhere.

    python tools/verify_portable.py [lesson ...] [--env py311,py312] [--reset] [--live]

Every other gate runs a lesson in THIS repository's environment, from THIS repository's
directory. A student does neither. They click a badge, and an online service opens one
notebook in an empty directory, in a Python it chose, with packages it chose.

So this gate does exactly that. For each lesson and each environment it:
  * copies ONLY lesson.ipynb into an empty temporary directory -- no Makefile, no lesson.c,
    no assets, no repository around it;
  * executes it in a real Jupyter kernel (so there is no __file__, as in Colab), in a
    deliberately minimal environment: Python, numpy, matplotlib, a kernel, and nothing else;
  * requires every cell to run.

The launcher cell at the top of the notebook must install what is missing and fetch the
sibling files. Until the repository is published there is nothing at the GitHub raw URL, so
by default the fetch is pointed at the lesson's own directory through ATLAS_RAW_OVERRIDE --
the identical urllib code path, reading file:// instead of https://. Pass --live once the
repository is public to test the real URLs.

Environments live in ~/.cache/atlas-portable, outside the repo and outside iCloud:
  py311  ~ Kaggle (Python 3.11)        py312  ~ Colab (Python 3.12)
--reset uninstalls the optional packages first, so the first lesson that needs each one
exercises the real install path instead of finding it left over from a previous run.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "atlas-portable"
OPTIONAL = ["mujoco", "tokenizers"]

RUNNER = r'''
import json, sys, time, nbformat
from nbclient import NotebookClient
path, cwd, timeout, expect = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
nb = nbformat.read(path, as_version=4)
# Prove which interpreter the kernel really is. A stray user-level "python3" kernelspec
# would otherwise run every notebook against the wrong Python and report a pass it never earned.
nb.cells.insert(0, nbformat.v4.new_code_cell("import sys; print('__ATLAS_EXE__' + sys.executable)"))
# allow_errors: keep going past a failing cell, so ONE run reports EVERY failing cell, the way
# a student pressing "Run all" would experience it cell by cell.
client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True,
                        resources={"metadata": {"path": cwd}})
t0, crash = time.time(), ""
try:
    client.execute()
except Exception as e:
    crash = (str(e).strip().splitlines() or ["?"])[-1][:300]
exe, launcher, errors = "", "", []
for i, cell in enumerate(nb.cells):
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            errors.append({"cell": i - 1, "ename": out["ename"], "evalue": out["evalue"][:160]})
        text = out.get("text", "") if out.get("output_type") == "stream" else ""
        for line in text.splitlines():
            if line.startswith("__ATLAS_EXE__"):
                exe = line[len("__ATLAS_EXE__"):]
            elif line.startswith(("ready on", "installing")):
                launcher += line + " "
wrong_env = bool(exe) and not exe.startswith(expect)
print(json.dumps({"secs": round(time.time() - t0, 1), "crash": crash, "errors": errors,
                  "launcher": launcher.strip(), "exe": exe, "wrong_env": wrong_env}))
'''


def classify(res) -> str:
    """Sort a failure by what would fix it, so a real portability break is never buried under
    a pile of unfilled-stub noise.

      ENV       the kernel was not the environment under test; the run proves nothing
      CRASH     the notebook could not be executed at all
      PORTABLE  a genuine environment failure: import, syntax, missing file, version drift
      STUB      an unfilled exercise raised instead of reporting "not implemented yet"
      EXIT      the lesson calls sys.exit() at the end -- right for a script, ugly in a notebook
    """
    if res.get("wrong_env"):
        return "ENV"
    if res.get("crash"):
        return "CRASH"
    names = [e["ename"] for e in res["errors"]]
    if not names:
        return "PASS"
    if any(n not in ("NotImplementedError", "SystemExit") for n in names):
        return "PORTABLE"
    return "STUB" if "NotImplementedError" in names else "EXIT"


def lessons(args):
    if args:
        return [Path(a).resolve() for a in args]
    out = []
    for pat in ("lessons/*/", "flagships/*/lessons/*/", "programmes/*/lessons/*/"):
        out += [p for p in ROOT.glob(pat) if (p / "lesson.ipynb").exists()]
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lesson", nargs="*")
    # ONE comma-separated value, not nargs="+": a greedy --env swallows every lesson path that
    # follows it on the command line and treats each as an environment name.
    ap.add_argument("--env", default="py311,py312")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    a.env = [e.strip() for e in a.env.split(",") if e.strip()]
    for env in a.env:
        if not re.fullmatch(r"py3\d{1,2}", env):
            ap.error(f"--env takes names like py311 or py312, got {env!r}")

    for env in a.env:
        py = CACHE / env / "bin" / "python"
        if not py.exists():
            print(f"missing environment {py}: create it with\n  uv venv --seed --python "
                  f"3.{env[3:]} {CACHE / env}\n  uv pip install --python {py} numpy "
                  "matplotlib ipykernel nbclient nbformat")
            return 2
        if a.reset:
            subprocess.run([str(py), "-m", "pip", "uninstall", "-y", "-q", *OPTIONAL],
                           capture_output=True)

    fails, counts = 0, {}
    for d in lessons(a.lesson):
        for env in a.env:
            py = CACHE / env / "bin" / "python"
            with tempfile.TemporaryDirectory(prefix="atlas-alone-") as tmp:
                shutil.copy2(d / "lesson.ipynb", Path(tmp) / "lesson.ipynb")
                environ = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
                environ["MPLBACKEND"] = "Agg"
                if not a.live:
                    environ["ATLAS_RAW_OVERRIDE"] = d.as_uri() + "/"
                try:
                    r = subprocess.run([str(py), "-c", RUNNER, str(Path(tmp) / "lesson.ipynb"),
                                        tmp, str(a.timeout), str(CACHE / env)],
                                       capture_output=True, text=True, env=environ,
                                       timeout=a.timeout + 60)
                    res = json.loads(r.stdout.strip().splitlines()[-1])
                except Exception as e:
                    res = {"secs": 0, "crash": f"runner crashed: {e}", "errors": [],
                           "launcher": "", "wrong_env": False}
            kind = classify(res)
            fails += kind != "PASS"
            counts[kind] = counts.get(kind, 0) + 1
            first = res["errors"][0] if res["errors"] else None
            detail = (res["launcher"] if kind == "PASS" else res["crash"] or
                      (f"cell {first['cell']}: {first['ename']}: {first['evalue']}" if first else ""))
            print(f"  [{kind:<9}] {env}  {d.name:<38} {res['secs']:>6}s  {detail}", flush=True)
    total = len(lessons(a.lesson)) * len(a.env)
    print("\n  " + "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"  {total - fails}/{total} notebook runs pass"
          f"{'' if a.live else '  (siblings served via ATLAS_RAW_OVERRIDE, not GitHub)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
