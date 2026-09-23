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
by default the fetch is pointed at the lesson's own directory through COMMONS_RAW_OVERRIDE --
the identical urllib code path, reading file:// instead of https://. Pass --live once the
repository is public to test the real URLs.

Environments live in ~/.cache/synapsa-commons/portable, outside the repo and outside iCloud:
  py311  ~ Kaggle (Python 3.11)        py312  ~ Colab (Python 3.12)
--reset uninstalls the optional packages first, so the first lesson that needs each one
exercises the real install path instead of finding it left over from a previous run.

--completed answers a different question. A student notebook full of unfilled stubs cannot
tell an environment failure from an unfinished exercise -- one stub's missing result cascades
into later cells. So --completed runs the FINISHED lesson (solutions/lesson_solution.py,
exactly as tools/execute.py does) in each minimal environment, where any error is real. It
then diffs what the lesson printed across environments, ignoring timing lines: if a finished
lesson prints the same numbers on Python 3.11 + numpy 2.4 as on 3.12 + numpy 2.5, it is
reproducible in the sense a student cares about. If it does not, that is reported, not hidden.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "synapsa-commons" / "portable"
OPTIONAL = ["mujoco", "tokenizers"]

RUNNER = r'''
import json, sys, time, nbformat
from nbclient import NotebookClient
path, cwd, timeout, expect = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
nb = nbformat.read(path, as_version=4)
# Prove which interpreter the kernel really is. A stray user-level "python3" kernelspec
# would otherwise run every notebook against the wrong Python and report a pass it never earned.
nb.cells.insert(0, nbformat.v4.new_code_cell("import sys; print('__COMMONS_EXE__' + sys.executable)"))
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
            if line.startswith("__COMMONS_EXE__"):
                exe = line[len("__COMMONS_EXE__"):]
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


# A lesson that prints its own environment ("python 3.11.15 · numpy 2.4.6") is SUPPOSED to
# differ across environments; that line is a banner, not a result.
BANNER = re.compile(r"\b(python|numpy|matplotlib|mujoco|tokenizers|torch|pyyaml|clang|gcc)"
                    r"\s+v?\d+\.\d+", re.I)
# A measurement of THIS machine -- a number carrying a unit of time, memory or rate -- is not a
# result, and is replaced by a placeholder before any comparison. Only the measurement is
# masked, not its line: "tuned on the nominal model alone (1.1s)" still has its words compared.
# Masking whole lines by pattern would hide results that share a line with a timing; relying
# only on two same-environment runs misses coarse readings like "0.9 s" that happen to repeat.
MEASURE = re.compile(
    # "≈N" is a number DERIVED from this machine's speed (how many samples fit a tick): the
    # lesson marks it so a student knows theirs will differ, and it is masked like a timing.
    r"≈\s*\d[\d,]*(?:\.\d+)?|"
    r"(?<![\w.])\d[\d,]*(?:\.\d+)?\s*"
    r"(?:s|ms|us|µs|ns|secs?|seconds?|milliseconds?|microseconds?|nanoseconds?"
    r"|KiB|MiB|GiB|kB|MB|GB|Hz|kHz|MHz|x|×"
    # rates of THIS machine. Not m/s: a simulated walking speed is a deterministic result.
    r"|(?:B|KB|kB|MB|GB|KiB|MiB|GiB|bytes|steps|it|samples|calls|ticks|rows|records|tokens"
    r"|docs|documents|captures|readings|lines|fields|events|merges|rollouts|symbols)/s"
    # time per unit of work on THIS machine: "0.673 us/tick", "12 ns/record"
    r"|(?:s|ms|us|µs|ns)/(?:tick|step|call|iter|iteration|item|row|record|doc|document|sample"
    r"|token|merge|byte|op|frame)s?)(?![\w/])")
BUILD_JUNK = shutil.ignore_patterns("lesson_bin", "lesson_bin *", "*.o", "*.dSYM", "__pycache__",
                                    "* [0-9]", "* [0-9].*", "lesson.ipynb", "build", ".ipynb_checkpoints")


def pins() -> dict:
    """name -> version from requirements.txt, so a completed run installs what was measured."""
    out = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.-]+)==([^\s;#]+)\s*(;|#|$)", line)
        if m and ";" not in line:
            out[m.group(1).lower()] = m.group(2)
    return out


def required(d: Path) -> list:
    """The (import, pip) pairs the lesson's own launcher declares."""
    m = re.search(r"^COMMONS_PIP = \[(.*?)\]", (d / "lesson.py").read_text(), re.M)
    return re.findall(r'\("([^"]+)", "([^"]+)"\)', m.group(1)) if m else []


def completed(d: Path, env: str, timeout: int) -> dict:
    py = CACHE / env / "bin" / "python"
    pin = pins()
    missing = [pip for imp, pip in required(d)
               if subprocess.run([str(py), "-c", f"import {imp}"], capture_output=True).returncode]
    if missing:
        subprocess.run([str(py), "-m", "pip", "install", "-q",
                        *[f"{m}=={pin[m.lower()]}" if m.lower() in pin else m for m in missing]],
                       capture_output=True, check=False)
    idents = env_identity(py)
    with tempfile.TemporaryDirectory(prefix="commons-done-") as tmp:
        work = Path(tmp) / d.name
        shutil.copytree(d, work, ignore=BUILD_JUNK)
        environ = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        environ["MPLBACKEND"] = "Agg"
        t0 = time.time()
        try:
            r = subprocess.run([str(py), "-c",
                                "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')",
                                "lesson_solution.py"], cwd=work / "solutions", env=environ,
                               capture_output=True, text=True, timeout=timeout)
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = -1, "", f"timed out after {timeout}s"
        # Strip what identifies THIS run and THIS environment -- the temp directory, the
        # interpreter path, the environment's own version strings -- before anything is compared.
        for real, label in [(str(work.resolve()), "<lesson>"), (str(work), "<lesson>"),
                            (str(Path(tmp).resolve()), "<tmp>"), (str(Path(tmp)), "<tmp>"),
                            *idents]:
            out = out.replace(real, label)
        out = MEASURE.sub("<measured>", out)
        # Column padding follows the width of whatever was printed in the column; two timings of
        # different widths shift the spaces around them. Alignment is not a result.
        out = "\n".join(re.sub(r"[ \t]+", " ", l).strip() for l in out.splitlines())
    last = (err.strip().splitlines() or [""])[-1][:200]
    return {"rc": rc, "secs": round(time.time() - t0, 1), "err": last,
            "installed": missing, "out": out.splitlines()}


def env_identity(py: Path) -> list:
    """(string, label) pairs that name an environment rather than describe a result."""
    probe = ("import sys, importlib.metadata as m\n"
             "print(sys.executable); print(sys.prefix); print(sys.version.split()[0])\n"
             "for p in ('numpy', 'matplotlib', 'mujoco', 'tokenizers', 'pyyaml'):\n"
             "    try: print(m.version(p))\n"
             "    except m.PackageNotFoundError: print('')")
    lines = subprocess.run([str(py), "-c", probe], capture_output=True, text=True).stdout.splitlines()
    pairs = [(lines[0], "<python>"), (str(Path(lines[1]).resolve()), "<env>"), (lines[1], "<env>")]
    pairs += [(v, "<ver>") for v in lines[2:] if v]
    # longest first, so /path/env/bin/python is replaced before /path/env
    return sorted(pairs, key=lambda p: -len(p[0]))


def run_completed(targets, envs, timeout) -> int:
    fails, drift = 0, 0
    for d in targets:
        res = {env: completed(d, env, timeout) for env in envs}
        # A line that differs between two runs in the SAME environment is non-deterministic by
        # definition -- a timing, a speed, a temp path. No list of regexes is needed to find
        # them: run the reference environment twice and mask whatever moved. Anything that is
        # stable within an environment but differs ACROSS environments is real drift.
        again = completed(d, envs[0], timeout)
        ok = all(r["rc"] == 0 for r in res.values()) and again["rc"] == 0
        fails += not ok
        base = res[envs[0]]["out"]
        noisy = {i for i, (a, b) in enumerate(zip(base, again["out"])) if a != b}
        noisy |= {i for i, l in enumerate(base) if BANNER.search(l)}
        diffs = {env: [(i, a, b) for i, (a, b) in enumerate(zip(base, r["out"]))
                       if a != b and i not in noisy]
                 + ([("len", len(base), len(r["out"]))] if len(base) != len(r["out"]) else [])
                 for env, r in res.items() if env != envs[0]}
        same = all(not v for v in diffs.values())
        drift += ok and not same
        status = "PASS" if ok and same else "DRIFT" if ok else "FAIL"
        secs = "  ".join(f"{e} {r['secs']}s" for e, r in res.items())
        inst = sorted({p for r in res.values() for p in r["installed"]})
        print(f"  [{status:<5}] {d.name:<38} {secs}"
              f"{'  installed ' + ','.join(inst) if inst else ''}"
              f"{'  output identical across ' + ' / '.join(envs) + f' ({len(base) - len(noisy)} lines; {len(noisy)} vary run to run)' if ok and same else ''}",
              flush=True)
        for env, r in res.items():
            if r["rc"] != 0:
                print(f"            {env}: exit {r['rc']}: {r['err']}")
        for env, v in diffs.items():
            for item in v[:3]:
                if item[0] == "len":
                    print(f"            {env}: {item[1]} vs {item[2]} output lines")
                else:
                    print(f"            {env} line {item[0]}:\n              {envs[0]}: {item[1][:110]}"
                          f"\n              {env}: {item[2][:110]}")
    n = len(targets)
    print(f"\n  completed runs: {n - fails}/{n} finish cleanly in every environment; "
          f"{drift} print different numbers across environments")
    return 1 if (fails or drift) else 0


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
    ap.add_argument("--completed", action="store_true",
                    help="run the finished lesson in each env and diff its output across envs")
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

    if a.completed:
        return run_completed(lessons(a.lesson), a.env, a.timeout)

    fails, counts = 0, {}
    for d in lessons(a.lesson):
        for env in a.env:
            py = CACHE / env / "bin" / "python"
            with tempfile.TemporaryDirectory(prefix="commons-alone-") as tmp:
                shutil.copy2(d / "lesson.ipynb", Path(tmp) / "lesson.ipynb")
                environ = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
                environ["MPLBACKEND"] = "Agg"
                if not a.live:
                    environ["COMMONS_RAW_OVERRIDE"] = d.as_uri() + "/"
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
          f"{'' if a.live else '  (siblings served via COMMONS_RAW_OVERRIDE, not GitHub)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
