#!/usr/bin/env python3
"""Launch surface (gate 13). Stdlib only, apart from jupytext for the build step.

    python tools/notebooks.py --inject [lesson ...]   # put/refresh the launcher in lesson.py
    python tools/notebooks.py --build  [lesson ...]   # regenerate lesson.ipynb from lesson.py
    python tools/notebooks.py --check  [lesson ...]   # fail if a launcher or .ipynb is stale or missing

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
import argparse, ast, hashlib, json, os, re, subprocess, sys, tempfile, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JUPYTEXT = str(ROOT / ".venv/bin/jupytext")

# Where the published copy will live. Every badge URL is derived from these three, so a repo
# rename is a one-line change here followed by --inject. PREFIX is the folder the lessons sit
# under inside the repository; "" means they sit at its root.
REPO, BRANCH, PREFIX = "TarrySingh/Artificial-Intelligence-Deep-Learning-Machine-Learning-Tutorials", "master", ""
# Binder and Codespaces clone the WHOLE repository they are pointed at, history included, and
# mybinder.org's build pod has 2 GB of disk. Colab and Kaggle fetch one notebook. So the two
# cloning services may point at a slim copy of the lessons; until one exists this is REPO.
LAUNCH_REPO = REPO

# The header every notebook opens with. SYNAPSA_URL is None until the platform is public; set
# it and re-run --inject, and every notebook links the name at once.
BRAND = "Synapsa Commons"
BRAND_BADGE = "brand/synapsa-commons-badge.png"
SYNAPSA_URL = "https://synapsa.realai.eu"

# Ownership and licence, stated in every lesson: a line in the header, the full notice at the
# foot, and a header in every source file a student receives. Generated here so that no lesson
# can drift from the wording RealAI publishes in NOTICE.
OWNER, YEAR = "RealAI", 2026
LICENCE, LICENCE_URL = "CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/"
CONTACT_URL = "https://www.realai.eu/contact"
SPDX = "SPDX-License-Identifier: CC-BY-NC-SA-4.0"
STAMP = f"Synapsa Commons - Copyright {YEAR} {OWNER} - free to learn from and share, not to sell; see NOTICE."
STAMPED = ("lesson.c", "lesson.cpp", "lesson.h", "Makefile", "solutions/lesson.c",
           "solutions/lesson.cpp", "solutions/lesson.h", "authoring/lesson.master.c")
NOTICE_OPEN = "# <!-- COMMONS NOTICE v1"

VERSION = "v4"
OPEN, CLOSE = f"# --- COMMONS LAUNCHER {VERSION}", "# --- END COMMONS LAUNCHER ---"
MD_OPEN = f"# <!-- COMMONS LAUNCHER {VERSION}"

# import name -> pip name, for the few that differ. Anything not listed installs under its own
# import name, which is right for numpy, matplotlib, mujoco, tokenizers and torch.
PIP_NAME = {"sklearn": "scikit-learn", "cv2": "opencv-python-headless", "PIL": "pillow",
            "yaml": "pyyaml", "skimage": "scikit-image"}
# Present in every environment worth supporting; never worth an install line.
ALWAYS_THERE = {"numpy", "matplotlib"}
SIBLINGS = ("Makefile", "lesson.c", "lesson.cpp", "lesson.h")


def stdlib() -> set:
    return set(sys.stdlib_module_names)


def pinned() -> dict:
    """pip name -> "name==version" from requirements.txt, so a notebook installs what every
    number in the lessons was measured with. Lines carrying an environment marker (numpy's
    per-Python pins) are skipped: those packages are never installed by the launcher."""
    out = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.-]+)==([^\s;#]+)\s*(#.*)?$", line)
        if m:
            out[m.group(1).lower()] = f"{m.group(1)}=={m.group(2)}"
    return out


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
    rel = "/".join(p for p in (PREFIX, lesson_rel(d)) if p)
    nb = f"{rel}/lesson.ipynb"
    raw = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{rel}/"
    badge = "/".join(p for p in (PREFIX, BRAND_BADGE) if p)
    synapsa = f"[Synapsa]({SYNAPSA_URL})" if SYNAPSA_URL else "Synapsa"
    pkgs = [p for p in third_party(src) if p not in ALWAYS_THERE]
    pins = pinned()
    specs = [pins.get(PIP_NAME.get(p, p).lower(), PIP_NAME.get(p, p)) for p in pkgs]
    pairs = ", ".join(f'("{p}", "{s}")' for p, s in zip(pkgs, specs)) or ""
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
        f"(https://mybinder.org/v2/gh/{LAUNCH_REPO}/{BRANCH}?labpath={nb})\n"
        f"# [![Open in Codespaces](https://github.com/codespaces/badge.svg)]"
        f"(https://codespaces.new/{LAUNCH_REPO})\n"
    )
    compiled = any(s.startswith("lesson.") and s != "lesson.h" for s in sibs)
    base = [p for p in ("numpy", "matplotlib") if p in third_party(src)]
    needs = ("Python 3.11 or newer" + (f" with {' and '.join(base)}, which Colab, Kaggle, Binder and "
             "Codespaces already have" if base else "")
             + (", and a C or C++ compiler (`clang` or `gcc`)" if compiled else ""))
    about = f"This lesson needs {needs}."
    if pkgs or sibs:
        what = " and ".join(x for x in (
            ("installs " + ", ".join(f"`{s}`" for s in specs)) if pkgs else "",
            "fetches the files it needs beside it" if sibs else "") if x)
        about += (f" The cell below {what}, and does nothing where they are already present. On "
                  "Kaggle, switch Internet on in the notebook's settings first; Kaggle allows that "
                  "only for phone-verified accounts.")
    about = textwrap.fill(about, width=94, initial_indent="# ", subsequent_indent="# ",
                          break_on_hyphens=False, break_long_words=False)
    return f'''# %% [markdown]
{MD_OPEN} · generated by tools/notebooks.py · do not edit by hand -->
# <a href="https://github.com/{REPO}"><img src="https://raw.githubusercontent.com/{REPO}/{BRANCH}/{badge}" alt="{BRAND}" height="36"></a>
#
# Free, hands-on AI courses that run anywhere, from the team building {synapsa}, an AI-native
# learning platform.
#
# © {YEAR} {OWNER} · free to learn from, share and adapt, not to sell ([{LICENCE}]({LICENCE_URL})).
# The notice at the end of this notebook says what you may and may not do.
#
{badges}#
{about}

# %%
{OPEN} · generated by tools/notebooks.py · do not edit by hand ---
# Makes this notebook run anywhere. Every line is a no-op when the thing is already present,
# so a local clone pays nothing and an online notebook repairs itself.
import importlib.util, os, subprocess, sys, urllib.request
from pathlib import Path

COMMONS_PIP = [{pairs}]            # (import name, pinned pip spec) for what this lesson imports
COMMONS_SIBLINGS = [{sib_list}]    # files that must sit beside the notebook
# A fork, a classroom mirror or an offline copy can serve the files from elsewhere by setting
# COMMONS_RAW_OVERRIDE before running this cell.
COMMONS_RAW = os.environ.get("COMMONS_RAW_OVERRIDE") or "{raw}"

# Resolve siblings against the LESSON's own directory, not the working directory. A notebook
# has no __file__ and runs with cwd alongside itself; a grader imports this file from the repo
# root. Checking cwd blindly makes the grader think every sibling is missing and reach for the
# network -- which would put a download on a graded path.
try:
    COMMONS_DIR = Path(__file__).resolve().parent
except NameError:
    COMMONS_DIR = Path.cwd()


def commons_host() -> str:
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


_missing = [pip for imp, pip in COMMONS_PIP if importlib.util.find_spec(imp) is None]
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
for _name in COMMONS_SIBLINGS:
    if not (COMMONS_DIR / _name).exists():
        (COMMONS_DIR / _name).parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(COMMONS_RAW + _name, COMMONS_DIR / _name)
            _fetched.append(_name)
        except Exception as _e:  # Kaggle disables the internet by default; say so plainly
            raise RuntimeError(
                f"this lesson needs {{_name}} beside the notebook and could not fetch it "
                f"({{_e}}). On Kaggle, switch Internet on in the notebook settings panel "
                f"(Kaggle allows that only for phone-verified accounts); otherwise download it "
                f"from {{COMMONS_RAW + _name}} and upload it beside the notebook."
            ) from None

print("ready on " + commons_host() + ("; fetched " + ", ".join(_fetched) if _fetched else ""))
{CLOSE}

'''


def wrap(text: str) -> str:
    return textwrap.fill(text, width=94, initial_indent="# ", subsequent_indent="# ",
                         break_on_hyphens=False, break_long_words=False)


def notice(d: Path) -> str:
    """The ownership and licence notice every lesson ends with."""
    repo = f"https://github.com/{REPO}"
    paras = [
        f"**Synapsa Commons** · © {YEAR} {OWNER} · licensed under [{LICENCE}]({LICENCE_URL})",
        "**You may** use this lesson to learn and to teach, and copy, fork, share and adapt it.",
        f"**You must** credit \"Synapsa Commons by {OWNER}\" with a link to {repo}, say what you "
        "changed, and share anything you adapt under this same licence.",
        "**You may not** use it, or anything adapted from it, in a way primarily intended for "
        "commercial advantage or payment: for example selling it, charging for a course, "
        "bootcamp or training built on it, or packaging it into a paid product or service. "
        f"For a commercial licence, contact [{OWNER}]({CONTACT_URL}).",
        "Third-party material in this lesson keeps its own licence, named in `assets/SOURCE.md` "
        f"or `claims.yaml`. The Synapsa name and logo belong to {OWNER} and are not licensed. "
        f"This summary is not the licence: the [legal code]({LICENCE_URL}legalcode) governs.",
    ]
    return ("# %% [markdown]\n"
            f"{NOTICE_OPEN} · generated by tools/notebooks.py · do not edit by hand -->\n"
            "# ---\n" + "\n#\n".join(wrap(t) for t in paras) + "\n")


def compose(d: Path, src: str) -> str:
    """lesson.py as it must be: generated launcher, the lesson itself, generated notice."""
    body = strip(src).rstrip("\n") + "\n"
    return launcher(d, body) + body + "\n" + notice(d)


def stamp_text(name: str, text: str) -> str:
    lead = "# " if name.endswith("Makefile") else "// "
    head = f"{lead}{SPDX}\n{lead}{STAMP}\n"
    lines = text.split("\n")
    if lines and lines[0].endswith(SPDX):
        return head + "\n".join(lines[2:])
    return head + text


def stamps(d: Path) -> list:
    """(path, stamped text) for every source file a lesson ships that should carry the header."""
    return [(d / n, stamp_text(n, (d / n).read_text())) for n in STAMPED if (d / n).exists()]


def inject(d: Path) -> str:
    p = d / "lesson.py"
    p.write_text(compose(d, p.read_text()))
    for f, text in stamps(d):
        if f.read_text() != text:
            f.write_text(text)
    return "injected"


def strip(src: str) -> str:
    """Remove any existing launcher block, so --inject is idempotent and updates in place."""
    src = re.sub(r"^# %% \[markdown\]\n# <!-- COMMONS LAUNCHER .*?(?=^# %%)", "", src,
                 flags=re.S | re.M)
    src = re.sub(r"^# %%\n# --- COMMONS LAUNCHER .*?^# --- END COMMONS LAUNCHER ---\n\n?", "", src,
                 flags=re.S | re.M)
    src = re.sub(r"\n*^# %% \[markdown\]\n# <!-- COMMONS NOTICE .*\Z", "\n", src, flags=re.S | re.M)
    return src


def render(d: Path):
    """lesson.py -> notebook text, rendered in a temporary directory. Touches nothing in the repo.

    Returns (text, "") or (None, error). Cell ids are derived, not random: nbformat gives every
    cell a RANDOM id on every conversion, which made identical notebooks compare unequal and
    would churn every one of them in git on every rebuild. Each id is a hash of the cell's
    index, type and source -- stable while the cell is unchanged, unique within the notebook.
    """
    with tempfile.TemporaryDirectory(prefix="commons-nb-") as tmp:
        out = Path(tmp) / "lesson.ipynb"
        r = subprocess.run([JUPYTEXT, "--to", "ipynb", str(d / "lesson.py"), "-o", str(out)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr.strip()[:160]
        nb = json.loads(out.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        cell["id"] = hashlib.sha1(f"{i}\x00{cell.get('cell_type')}\x00{src}".encode()).hexdigest()[:12]
    # nbformat's own on-disk style, so a file written here is byte-identical to one it writes.
    return json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n", ""


def build(d: Path) -> str:
    text, err = render(d)
    if text is None:
        return f"FAILED {err}"
    tmp = d / ".lesson.ipynb.tmp"
    tmp.write_text(text)
    os.replace(tmp, d / "lesson.ipynb")   # atomic: a reader never sees half a notebook
    return "built"


def check(d: Path) -> str:
    """Compare the committed notebook with a fresh render. Writes NOTHING.

    An earlier version regenerated the notebook in place and then compared it, so a check run
    with no arguments silently rewrote every stale notebook in the repository -- one reviewer
    did exactly that and changed a lesson it had not been asked to touch. A check must not
    mutate what it checks.
    """
    nb = d / "lesson.ipynb"
    if not nb.exists():
        return "MISSING lesson.ipynb"
    # The launcher is generated too. Comparing only notebook to lesson.py would pass a lesson
    # whose launcher still points at an old branch, an old path or an unpinned package.
    src = (d / "lesson.py").read_text()
    if compose(d, src) != src:
        return "STALE LAUNCHER OR NOTICE (run --inject, then --build, then commit)"
    if any(f.read_text() != text for f, text in stamps(d)):
        return "UNSTAMPED SOURCE (run --inject)"
    text, err = render(d)
    if text is None:
        return f"FAILED {err}"
    return "ok" if nb.read_text() == text else "STALE (run --build, then commit)"


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
