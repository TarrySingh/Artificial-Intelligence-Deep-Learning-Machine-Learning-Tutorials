#!/usr/bin/env python3
"""Launch surface (gate 13). Stdlib only, apart from jupytext for the build step.

    python tools/notebooks.py --inject [lesson ...]   # put/refresh the launcher in lesson.py
    python tools/notebooks.py --build  [lesson ...]   # regenerate lesson.ipynb from lesson.py
    python tools/notebooks.py --check  [lesson ...]   # fail if any .ipynb is stale or missing

Every lesson must open and run in the environments a student actually has: Google Colab,
Kaggle, Binder, GitHub Codespaces, or a local Jupyter. That is not something an author should
hand-write per lesson -- hand-written badges rot the moment a file moves. So the launcher is
GENERATED from the lesson itself: this tool reads which third-party packages the lesson
imports and which sibling files it needs beside it, and writes one canonical block into
lesson.py between sentinels. Re-run it and every lesson updates at once.

The block is a no-op wherever everything is already present, which is why the execution gate
does not slow down and why `no network on a required path` still holds: nothing is installed
and nothing is fetched unless it is genuinely absent.
"""
import argparse, ast, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUPYTEXT = str(ROOT / ".venv/bin/jupytext")

# Where the published copy will live. Every badge URL is derived from these three, so a repo
# rename is a one-line change here followed by --inject.
REPO, BRANCH, PREFIX = "TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials", "main", "atlas"

VERSION = "v1"
OPEN, CLOSE = f"# --- ATLAS LAUNCHER {VERSION}", "# --- END ATLAS LAUNCHER ---"
MD_OPEN = f"# <!-- ATLAS LAUNCHER {VERSION}"

# import name -> pip name, for the few that differ. Anything not listed installs under its own
# import name, which is right for numpy, matplotlib, mujoco, tokenizers and torch.
PIP_NAME = {"sklearn": "scikit-learn", "cv2": "opencv-python-headless", "PIL": "pillow",
            "yaml": "pyyaml", "skimage": "scikit-image"}
# Present in every environment worth supporting; never worth an install line.
ALWAYS_THERE = {"numpy", "matplotlib"}
SIBLINGS = ("Makefile", "lesson.c", "lesson.cpp", "lesson.h")


def stdlib() -> set:
    return set(sys.stdlib_module_names)


OPTIONAL_GUARDS = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _guards_import(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True
    names = [t] if not isinstance(t, ast.Tuple) else list(t.elts)
    return any(isinstance(n, ast.Name) and n.id in OPTIONAL_GUARDS for n in names)


def third_party(src: str) -> list:
    """REQUIRED top-level third-party imports of a lesson, in first-seen order.

    An import inside `try: ... except ImportError:` is optional by the author's design -- one
    lesson imports torch that way precisely to show what happens when it is absent. Forcing an
    install there would download hundreds of megabytes on every run AND destroy the lesson's
    point, so such imports are skipped.
    """
    found, seen = [], set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return found

    optional = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(_guards_import(h) for h in node.handlers):
            for inner in node.body:
                for sub in ast.walk(inner):
                    optional.add(id(sub))

    for node in ast.walk(tree):
        if id(node) in optional:
            continue
        names = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module.split(".")[0]]
        for n in names:
            if n in seen or n in stdlib() or n.startswith("_"):
                continue
            seen.add(n)
            found.append(n)
    return found


def lesson_rel(d: Path) -> str:
    return d.relative_to(ROOT).as_posix()


def launcher(d: Path, src: str) -> str:
    rel = lesson_rel(d)
    nb = f"{PREFIX}/{rel}/lesson.ipynb"
    raw = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{PREFIX}/{rel}/"
    pkgs = [p for p in third_party(src) if p not in ALWAYS_THERE]
    pairs = ", ".join(f'("{p}", "{PIP_NAME.get(p, p)}")' for p in pkgs) or ""
    sibs = [s for s in SIBLINGS if (d / s).exists()]
    if (d / "assets").is_dir():
        sibs += [f"assets/{f.name}" for f in sorted((d / "assets").iterdir()) if f.is_file()]
    sib_list = ", ".join(f'"{s}"' for s in sibs)

    badges = (
        f"# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{nb})\n"
        f"# [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]"
        f"(https://kaggle.com/kernels/welcome?src=https://github.com/{REPO}/blob/{BRANCH}/{nb})\n"
        f"# [![Open in Binder](https://mybinder.org/badge_logo.svg)]"
        f"(https://mybinder.org/v2/gh/{REPO}/{BRANCH}?labpath={nb})\n"
        f"# [![Open in Codespaces](https://github.com/codespaces/badge.svg)]"
        f"(https://github.com/codespaces/new?repo={REPO})\n"
    )
    needs = "a compiler (`clang` or `gcc`)" if any(s.startswith("lesson.") and s != "lesson.h" for s in sibs) else "nothing beyond Python"
    return f'''# %% [markdown]
{MD_OPEN} · generated by tools/notebooks.py · do not edit by hand -->
{badges}#
# **Runs in:** Google Colab · Kaggle · Binder · GitHub Codespaces · local Jupyter.
# This lesson needs {needs}. The cell below installs anything missing and fetches the files
# this lesson needs beside it; on a machine that already has them it does nothing at all.

# %%
{OPEN} · generated by tools/notebooks.py · do not edit by hand ---
# Makes this notebook run anywhere. Every line is a no-op when the thing is already present,
# so a local clone pays nothing and an online notebook repairs itself.
import importlib.util, os, subprocess, sys, urllib.request
from pathlib import Path

ATLAS_PIP = [{pairs}]            # (import name, pip name) for what this lesson imports
ATLAS_SIBLINGS = [{sib_list}]    # files that must sit beside the notebook
# A fork, a classroom mirror or an offline copy can serve the files from elsewhere by setting
# ATLAS_RAW_OVERRIDE before running this cell.
ATLAS_RAW = os.environ.get("ATLAS_RAW_OVERRIDE") or "{raw}"

# Resolve siblings against the LESSON's own directory, not the working directory. A notebook
# has no __file__ and runs with cwd alongside itself; a grader imports this file from the repo
# root. Checking cwd blindly makes the grader think every sibling is missing and reach for the
# network -- which would put a download on a graded path.
try:
    ATLAS_DIR = Path(__file__).resolve().parent
except NameError:
    ATLAS_DIR = Path.cwd()


def atlas_host() -> str:
    """Name the notebook service we are on. Used for the message, and for honest errors."""
    try:
        if importlib.util.find_spec("google.colab") is not None:
            return "Google Colab"
    except (ImportError, ValueError):
        pass
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "Kaggle"
    if os.environ.get("BINDER_SERVICE_HOST"):
        return "Binder"
    if os.environ.get("CODESPACES"):
        return "GitHub Codespaces"
    return "a local Python environment"


_missing = [pip for imp, pip in ATLAS_PIP if importlib.util.find_spec(imp) is None]
if _missing:
    print("installing " + ", ".join(_missing) + " ...")
    # pip everywhere a student is likely to be; uv-managed local venvs ship without pip.
    if importlib.util.find_spec("pip") is not None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_missing], check=True)
    else:
        subprocess.run(["uv", "pip", "install", "-q", "--python", sys.executable, *_missing],
                       check=True)
    importlib.invalidate_caches()

_fetched = []
for _name in ATLAS_SIBLINGS:
    if not (ATLAS_DIR / _name).exists():
        (ATLAS_DIR / _name).parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(ATLAS_RAW + _name, ATLAS_DIR / _name)
            _fetched.append(_name)
        except Exception as _e:  # Kaggle disables the internet by default; say so plainly
            raise RuntimeError(
                f"this lesson needs {{_name}} beside the notebook and could not fetch it "
                f"({{_e}}). On Kaggle, switch Internet on in the notebook settings panel; "
                f"otherwise download it from {{ATLAS_RAW + _name}} and upload it."
            ) from None

print("ready on " + atlas_host() + ("; fetched " + ", ".join(_fetched) if _fetched else ""))
{CLOSE}

'''


def inject(d: Path) -> str:
    p = d / "lesson.py"
    src = p.read_text()
    body = strip(src)
    p.write_text(launcher(d, body) + body)
    return "injected"


def strip(src: str) -> str:
    """Remove any existing launcher block, so --inject is idempotent and updates in place."""
    src = re.sub(r"^# %% \[markdown\]\n# <!-- ATLAS LAUNCHER .*?(?=^# %%)", "", src,
                 flags=re.S | re.M)
    src = re.sub(r"^# %%\n# --- ATLAS LAUNCHER .*?^# --- END ATLAS LAUNCHER ---\n\n?", "", src,
                 flags=re.S | re.M)
    return src


def build(d: Path) -> str:
    r = subprocess.run([JUPYTEXT, "--to", "ipynb", str(d / "lesson.py"),
                        "-o", str(d / "lesson.ipynb")], capture_output=True, text=True)
    return "built" if r.returncode == 0 else f"FAILED {r.stderr.strip()[:120]}"


def check(d: Path) -> str:
    """A committed .ipynb that no longer matches its source is worse than none at all."""
    nb = d / "lesson.ipynb"
    if not nb.exists():
        return "MISSING lesson.ipynb"
    before = nb.read_bytes()
    build(d)
    return "ok" if nb.read_bytes() == before else "STALE (regenerated; commit it)"


def lessons(args) -> list:
    if args:
        return [Path(a).resolve() for a in args]
    out = []
    for pat in ("lessons/*/", "flagships/*/lessons/*/", "programmes/*/lessons/*/"):
        out += [p for p in ROOT.glob(pat) if (p / "lesson.py").exists()]
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("lesson", nargs="*")
    a = ap.parse_args()
    fn = inject if a.inject else build if a.build else check if a.check else None
    if fn is None:
        ap.error("pick one of --inject, --build, --check")
    bad = 0
    for d in lessons(a.lesson):
        r = fn(d)
        if r not in ("ok", "injected", "built"):
            bad += 1
        print(f"  {r:<12} {d.name}")
    print(f"\n  {len(lessons(a.lesson)) - bad}/{len(lessons(a.lesson))} lessons ok")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
