#!/usr/bin/env python3
"""Repo-wide gate runner. Trusts nothing an author or reviewer reported.

    python tools/verify_all.py [--write-back]

For every lesson directory under lessons/, flagships/*/lessons/ and programmes/*/lessons/:
  gate 1     3-5 objectives in meta.yaml, none opening with "understand", "learn" or "know"
  gate 2     every prerequisite names a lesson directory that exists
  gate 3     no run of more than 40 non-blank markdown lines before the next code cell
  gate 5     lesson.py defines at least one public _check_ helper
  gate 8     lesson.py has a self-check section
  gate 11    every dataset in meta.yaml names its licence and a URL
  gate 12    claims.yaml exists and every claim carries a source URL and an access date
  gate 9+10  tools/execute.py            -> runs clean inside the declared tier and budget
  gate 6     tools/grade.py --solution   -> the reference implementation scores 100%
  gate 4     tools/grade.py              -> the student file does NOT score 100% (stubs are real)
  gate 7     jupytext --to ipynb         -> the source converts to a student notebook
  gate T     (checked inside structure()) timer guards: in lesson.py, solutions/lesson_solution.py and tests/test_lesson.py, any
             code that arms SIGALRM must save the timer already running (signal.getitimer) and, in a
             `finally`, re-arm it with its time left AND its interval; it must never switch the timer
             off with a literal `setitimer(ITIMER_REAL, 0...)` (except inside a `finally` that re-arms
             the saved timer after it, as the canonical guard does to stop its own one-shot alarm before
             it restores the old handler) nor use signal.alarm (which cannot
             restore an interval), and what its SIGALRM handler raises must derive from BaseException,
             not Exception. A guard that cancels an outer timer silently disables tools/grade.py's
             per-row time limit and any harness alarm. The canonical guard is in tools/lesson_template.
Gates 13-14 are tools/notebooks.py --check and tools/verify_portable.py. What no tool can judge --
whether an objective is measurable, whether a hint is specific, whether a claim's quote is on the
live page, whether a number in the prose was typed -- is the independent reviewer's, recorded in
each lesson's meta.yaml. Exits non-zero if any lesson fails any gate.
"""
import ast, builtins, re, subprocess, sys
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


PROSE_LIMIT = 40
VAGUE = ("understand", "learn", "know", "appreciate", "be aware")


def before_you_start(meta: dict, src: str) -> list:
    """Gate 2, second half: a "Before you start" box, so a learner knows what to bring and where to
    learn it. The data lives in meta.yaml; tools/notebooks.py --inject writes the box. Whether each
    link is live and whether `free` is true is checked by tools/check_resources.py (needs the
    network) and by the reviewer; this check is what a machine can decide offline."""
    box = meta.get("before_you_start")
    if not isinstance(box, dict):
        return ["g2: no before_you_start in meta.yaml"]
    bad = []
    assumed, res = box.get("assumed") or [], box.get("resources") or []
    bad += [f"g2: an assumed skill is not plain text (quote it in meta.yaml): {str(a)[:40]}"
            for a in assumed if not isinstance(a, str)]
    if not 2 <= len(assumed) <= 5:
        bad.append(f"g2: before_you_start lists {len(assumed)} assumed skills (2-5)")
    if not 2 <= len(res) <= 5:
        bad.append(f"g2: before_you_start lists {len(res)} resources (2-5)")
    seen = set()
    for r in res:
        if not isinstance(r, dict) or not all(isinstance(r.get(k), str) and r.get(k) for k in ("title", "by", "url", "kind", "why")):
            bad.append("g2: a resource needs title, by, url, kind and why")
            continue
        url = str(r["url"])
        if not url.startswith("https://") or any(x in url for x in ("bit.ly", "t.co/", "tinyurl")):
            bad.append(f"g2: resource url must be a direct https link: {url[:50]}")
        if url in seen:
            bad.append(f"g2: resource listed twice: {url[:50]}")
        seen.add(url)
        if r["kind"] not in ("free", "paid"):
            bad.append(f"g2: resource kind must be free or paid: {r['kind']}")
        if len(str(r["why"])) > 220:
            bad.append("g2: a resource's `why` runs past 220 characters")
    if res and not any(isinstance(r, dict) and r.get("kind") == "free" for r in res):
        bad.append("g2: at least one resource must be free")
    if "# <!-- COMMONS BEFORE" not in src:
        bad.append("g2: the Before-you-start box is not injected (run tools/notebooks.py --inject)")
    return bad


def structure(d: Path, known: set) -> list:
    """The machine-checkable half of gates 1, 2, 3, 5, 8, 11 and 12. Returns what failed."""
    import yaml
    bad = []
    meta = yaml.safe_load((d / "meta.yaml").read_text()) or {}
    objectives = meta.get("objectives") or []
    if not 3 <= len(objectives) <= 5:
        bad.append(f"g1: {len(objectives)} objectives")
    bad += [f"g1: vague objective '{o[:30]}'" for o in objectives if str(o).lower().startswith(VAGUE)]
    bad += [f"g2: unknown prerequisite {p}" for p in (meta.get("prerequisites") or []) if p not in known]
    bad += before_you_start(meta, (d / "lesson.py").read_text())
    run = worst = 0
    in_md = False
    in_box = False
    for line in (d / "lesson.py").read_text().splitlines():
        if line.startswith("# %%"):
            if "[markdown]" not in line:
                worst, run = max(worst, run), 0
            in_md = "[markdown]" in line
            in_box = False
        elif in_md and line.startswith("# <!-- COMMONS BEFORE"):
            in_box = True          # generated reference box: not teaching prose, not counted by g3
        elif in_md and not in_box and line.lstrip("#").strip():
            run += 1
    worst = max(worst, run)
    if worst > PROSE_LIMIT:
        bad.append(f"g3: {worst} lines of prose before a code cell")
    src = (d / "lesson.py").read_text()
    if not re.search(r"^def _check_", src, re.M):
        bad.append("g5: no _check_ helper")
    if not re.search(r"self-check", src, re.I):
        bad.append("g8: no self-check")
    for ds in meta.get("datasets") or []:
        # A download names its URL; data generated by the lesson (kind: synthetic) or authored for
        # it and committed beside it (cached_at) has none to name, but every entry names a licence.
        where = isinstance(ds, dict) and (ds.get("url") or ds.get("kind") == "synthetic" or ds.get("cached_at"))
        if not (isinstance(ds, dict) and ds.get("licence") and where):
            bad.append(f"g11: dataset without a licence and a source: {str(ds)[:40]}")
    claims = d / "claims.yaml"
    if not claims.exists():
        bad.append("g12: no claims.yaml")
    else:
        for c in (yaml.safe_load(claims.read_text()) or {}).get("claims") or []:
            if str(c.get("claim", "")).startswith("NOT A CLAIM"):   # a recorded omission
                continue
            if not (str(c.get("source_url", "")).startswith("http") and c.get("accessed")):
                bad.append(f"g12: claim {c.get('id', '?')} lacks a source URL or access date")
    bad += timer_guards(d)
    return bad


TIMER_FILES = ("lesson.py", "solutions/lesson_solution.py", "tests/test_lesson.py")


def _call_name(node) -> str:
    f = node.func if isinstance(node, ast.Call) else None
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""


def _mentions_sigalrm(node) -> bool:
    return any(isinstance(n, (ast.Attribute, ast.Name)) and "SIGALRM" in (getattr(n, "attr", "") or getattr(n, "id", ""))
               for n in ast.walk(node))


def _derives_from_base_only(name: str, classes: dict, seen=()) -> bool:
    """True if the exception class `name` derives from BaseException but not from Exception."""
    if name in classes and name not in seen:
        return all(_derives_from_base_only(b.id if isinstance(b, ast.Name) else "?", classes, seen + (name,))
                   for b in classes[name].bases) and bool(classes[name].bases)
    obj = getattr(builtins, name, None)
    return isinstance(obj, type) and issubclass(obj, BaseException) and not issubclass(obj, Exception)


def timer_guards(d: Path) -> list:
    """Gate T: a SIGALRM guard must restore the timer it found, and raise a BaseException."""
    bad = []
    for rel in TIMER_FILES:
        f = d / rel
        if not f.exists():
            continue
        src = f.read_text()
        if not re.search(r"setitimer|\balarm\s*\(|SIGALRM", src):
            continue
        tree = ast.parse(src)
        classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        # units: every top-level function or class, and the module's own statements
        units = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        loose = ast.Module(body=[n for n in tree.body if n not in units], type_ignores=[])
        for unit in units + [loose]:
            where = f"{rel}:{getattr(unit, 'name', '<module>')}"
            calls = [n for n in ast.walk(unit) if isinstance(n, ast.Call)]
            arms = [c for c in calls if _call_name(c) == "setitimer"]
            if any(_call_name(c) == "alarm" and isinstance(c.func, ast.Attribute) for c in calls):
                bad.append(f"gT: {where} uses signal.alarm, which cannot restore an outer interval timer")
            def _rearms(c):     # setitimer(which, <time left>, <interval>) or setitimer(which, *saved)
                return (_call_name(c) == "setitimer"
                        and (len(c.args) >= 3 or any(isinstance(a, ast.Starred) for a in c.args))
                        and not isinstance(c.args[1] if len(c.args) > 1 else None, ast.Constant))

            def _finally_calls(n):
                return [c for stmt in n.finalbody for c in ast.walk(stmt) if isinstance(c, ast.Call)]
            for c in arms:
                iv = c.args[2] if len(c.args) > 2 else None
                # A guard must arm a ONE-SHOT alarm and re-arm it by hand. A repeating interval of its own
                # (a numeric literal) can deliver a late alarm after the old handler is back: under GC
                # pressure that killed the process (-14). Re-arming a SAVED interval is not a literal.
                if isinstance(iv, ast.Constant) and isinstance(iv.value, (int, float)) and iv.value:
                    bad.append(f"gT: {where} line {c.lineno} arms a repeating alarm of its own "
                               "(setitimer(..., interval)); use the one-shot canonical guard")
                t = c.args[1] if len(c.args) > 1 else None
                # A literal stop is allowed only inside a `finally` that re-arms the saved timer AFTER it
                # (the canonical guard stops its own alarm, takes any of its own that is pending, puts the
                # old handler back, and only then re-arms the outer one). Anywhere else it cancels the outer.
                if isinstance(t, ast.Constant) and not t.value and not any(
                        any(x is c for x in _finally_calls(n)) and any(
                            _rearms(r) and (r.lineno, r.col_offset) > (c.lineno, c.col_offset) for r in _finally_calls(n))
                        for n in ast.walk(unit) if isinstance(n, ast.Try)):
                    bad.append(f"gT: {where} line {c.lineno} switches the timer off (setitimer(..., 0)) "
                               "instead of re-arming the timer that was running before")
            if arms:
                saved = any(_call_name(c) == "getitimer" for c in calls)
                restored = any(isinstance(n, ast.Try) and any(_rearms(c) for c in _finally_calls(n))
                               for n in ast.walk(unit))
                if not saved:
                    bad.append(f"gT: {where} arms a timer without saving the one already running (signal.getitimer)")
                if not restored:
                    bad.append(f"gT: {where} never re-arms the previous timer, with its time left and its "
                               "interval, in a finally")
            # what each SIGALRM handler raises must be a BaseException, never an Exception
            for c in calls:
                if _call_name(c) == "signal" and len(c.args) == 2 and _mentions_sigalrm(c.args[0]) \
                        and isinstance(c.args[1], ast.Name) and c.args[1].id in funcs:
                    handler = funcs[c.args[1].id]
                    for r in ast.walk(handler):
                        if isinstance(r, ast.Raise) and r.exc is not None:
                            e = r.exc.func if isinstance(r.exc, ast.Call) else r.exc
                            name = e.id if isinstance(e, ast.Name) else ast.unparse(e)
                            if not _derives_from_base_only(name, classes):
                                bad.append(f"gT: {where} SIGALRM handler {handler.name} raises {name}, which "
                                           "an `except Exception:` can swallow; derive it from BaseException")
    return sorted(set(bad))


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    wb = ["--write-back"] if "--write-back" in sys.argv else []
    rows, failed = [], 0
    every = list(lessons())
    known = {d.name for d in every}
    for d in every:
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
        struct = structure(d, known)
        ok = (rc_exec == 0) and (sol == 100) and (stu is not None and stu < 100) and (rc_nb == 0) and not struct
        if not ok:
            failed += 1
        rows.append((("PASS" if ok else "FAIL"), rid, lang, tier, secs, peak,
                     f"{sol}%" if sol is not None else "-", f"{stu}%" if stu is not None else "-", nb,
                     "" if ok else f"exec_rc={rc_exec} nb_rc={rc_nb} " + "; ".join(struct)))

    w = max((len(r[1]) for r in rows), default=8) + 1
    print(f"\n  {'':5} {'lesson':<{w}} {'lang':<7} {'tier':<9} {'secs':>6} {'MiB':>6} {'ref':>5} {'stub':>5} {'cells':>5}")
    for r in rows:
        print(f"  [{r[0]}] {r[1]:<{w}} {r[2]:<7} {r[3]:<9} {r[4]:>6} {r[5]:>6} {r[6]:>5} {r[7]:>5} {r[8]:>5}  {r[9]}")
    print(f"\n  {len(rows) - failed}/{len(rows)} lessons pass every gate\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
