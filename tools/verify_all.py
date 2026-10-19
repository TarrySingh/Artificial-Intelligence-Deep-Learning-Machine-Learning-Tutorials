#!/usr/bin/env python3
"""Repo-wide gate runner. Trusts nothing an author or reviewer reported.

    python tools/verify_all.py [--write-back] [lesson-dir ...]     # no directory: every lesson

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
             And every notebook check must run under such a guard: in lesson.py, the worked solution
             and the authoring master (authoring/lesson.master.py, where there is one), the `_try`
             runner calls its check inside `with <guard>(...)`, or hands it to a helper that does,
             and no cell calls a `_check_` function outside `_try`. Otherwise an answer that loops
             for ever, or appends until memory runs out, freezes "Run all" (and the grader, which
             imports the notebook) instead of failing its own row. Since the GUARD-2026-10 review it also fails a lesson that
             runs student code outside `_try` in any shape (a direct call, an alias, a loop variable over a table of checks,
             `partial`, `map`, a thread, `exec`; meta.yaml `guard_allow` names the given calls a human has checked), whose
             COMMONS TIME LIMIT block (and, in a C or C++ lesson, COMMONS CHILD LIMIT block) differs from the one in
             tools/lesson_template, that does not declare `_CHECK_SECONDS`/`_DEMO_SECONDS` and arm `_try` with them, or whose
             limits are out of bounds (check 10 s, demo within `budget_seconds`, memory 2048 MiB, exercise cells x limit within
             the grader's 135 s import limit; `guard_limits_reason` in meta.yaml explains an exception).
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
    unguarded = checks_guarded(d)
    if d.name not in guard_backlog():
        bad += unguarded
    elif not unguarded:      # the backlog only shrinks: a lesson that is done must leave it
        bad.append(f"gT: every check is time-limited now; remove {d.name} from {_backlog_name()}")
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


# Gate T, second half: every notebook check runs under a time limit. The files a check runs in: the
# student notebook, the worked solution, and the authoring master the two are generated from.
CHECK_FILES = ("lesson.py", "solutions/lesson_solution.py", "authoring/lesson.master.py")
# Lessons built before this rule that still await the retrofit (internal/retrofit_guard.py), one
# directory name per line. PRIVATE, and absent from the public copy, where the rule is simply strict.
# It may only shrink: a listed lesson whose checks are all guarded fails until it is taken off.
GUARD_BACKLOG = ROOT / "internal" / "guard_backlog.txt"


def _backlog_name() -> str:
    return str(GUARD_BACKLOG.relative_to(ROOT) if GUARD_BACKLOG.is_relative_to(ROOT) else GUARD_BACKLOG)


def guard_backlog() -> frozenset:
    if not GUARD_BACKLOG.exists():
        return frozenset()
    return frozenset(line.split("#")[0].strip() for line in GUARD_BACKLOG.read_text().splitlines()
                     if line.split("#")[0].strip())


def _guard_functions(tree) -> set:
    """Module-level functions that arm SIGALRM: the file's time-limit guards. timer_guards() checks
    that each one restores the timer it found and raises a BaseException."""
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(c, ast.Call) and _call_name(c) == "setitimer" for c in ast.walk(n))}


def _is_guard(item, guards) -> bool:
    return isinstance(item.context_expr, ast.Call) and _call_name(item.context_expr) in guards


def _param_calls(fn, param: str, guards: set) -> list:
    """For every call of the parameter `param` inside fn: is it inside `with <guard>(...)`?"""
    out = []

    def visit(node, guarded):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                visit(item.context_expr, guarded)
            inside = guarded or any(_is_guard(item, guards) for item in node.items)
            for stmt in node.body:
                visit(stmt, inside)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == param:
            out.append(guarded)
        for child in ast.iter_child_nodes(node):
            visit(child, guarded)
    for stmt in fn.body:
        visit(stmt, False)
    return out


def _loose_checks(tree, guards: set) -> list:
    """`_check_` calls a cell makes itself (module level, outside `_try(...)` and any guard)."""
    out = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return                                   # defined here, run later (through _try)
        if isinstance(node, (ast.With, ast.AsyncWith)) and any(_is_guard(i, guards) for i in node.items):
            return
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "_try":
                return                               # its arguments run under _try's limit
            if isinstance(node.func, ast.Name) and name.startswith("_check_"):
                out.append(f"{name}() on line {node.lineno}")
        for child in ast.iter_child_nodes(node):
            visit(child)
    for stmt in tree.body:
        visit(stmt)
    return out


# --- Gate T, third part: student code run outside `_try`, the canonical block, and the limits ----------------

TOOLCHAIN = ("make", "cc", "clang", "gcc", "c++", "g++", "clang++", "uv")


def _body(fn) -> list:
    return fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)) else fn.body


def _is_stub(fn) -> bool:
    return any(isinstance(s, ast.Raise) and s.exc is not None and "NotImplementedError" in ast.unparse(s.exc)
               for s in _body(fn))


def _runs_a_program(fn) -> bool:
    for c in ast.walk(fn):
        if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name)
                and c.func.value.id == "subprocess" and c.func.attr in ("run", "Popen", "call", "check_output", "check_call")):
            first = c.args[0] if c.args else None
            words = [e.value for e in first.elts if isinstance(e, ast.Constant)] if isinstance(first, ast.List) else []
            if words and words[0] in TOOLCHAIN and not {"test", "run", "check"} & set(words):
                continue                                 # a compile or a path query
            return True
    return False


INDIRECT = ("partial", "map", "filter", "starmap", "submit", "apply_async", "call_soon", "Thread", "Process")


def outside_try(d: Path, allow=()) -> list:
    """Where a cell of lesson.py runs the student's code WITHOUT going through `_try`, where no limit can reach it.

    The student's code is every function or method that is a stub in lesson.py (its body raises
    NotImplementedError) and every helper that runs a program other than the compiler, plus everything
    that calls either (a call-graph closure), plus an alias of any of them (`f = risky`, or a loop variable
    over a list that holds one). A module-level call to any of these outside `_try(...)`, a hand-off to
    `partial`/`map`/a thread, or `exec`/`eval`, is a hit. A hit that has been looked at and runs no student
    code is accepted by the lesson's meta.yaml `guard_allow` (a list of function names)."""
    student = ast.parse((d / "lesson.py").read_text())
    funcs = {n.name: n for n in student.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    methods = {m.name for n in student.body if isinstance(n, ast.ClassDef) for m in n.body
               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_stub(m)}
    risky = {name for name, n in funcs.items() if not isinstance(n, ast.ClassDef) and (_is_stub(n) or _runs_a_program(n))}
    risky |= {name for name, n in funcs.items() if isinstance(n, ast.ClassDef) and any(
        isinstance(m, ast.FunctionDef) and m.name in ("__init__", "__call__", "__post_init__") and _is_stub(m) for m in n.body)}
    risky |= {"." + m for m in methods}

    def refs(node) -> set:
        out = set()
        for c in ast.walk(node):
            if isinstance(c, ast.Name) and isinstance(c.ctx, ast.Load):
                out.add(c.id)
            elif isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr in methods:
                out.add("." + c.func.attr)
        return out
    graph = {name: refs(n) for name, n in funcs.items() if name != "_try"}
    for name in allow:                       # accepted: it, and what it calls, runs no student code
        risky.discard(name)
        graph[name] = set()
    # aliases made by module-level code: `f = risky`, `for name, f in [("x", risky), ...]`. A table of checks
    # (`_BOARD = [("exercise 1", _check_one, ...)]`) is data, not an alias: only a loop over it is.
    holders = {}
    for stmt in student.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, (ast.List, ast.Tuple, ast.Dict, ast.Set)) \
                and isinstance(stmt.target, ast.Name):
            holders[stmt.target.id] = refs(stmt.value)
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    holders[tgt.id] = refs(stmt.value)
        elif isinstance(stmt, ast.Assign) and refs(stmt.value) & (set(graph) | risky):
            for tgt in stmt.targets:
                for n in ast.walk(tgt):
                    if isinstance(n, ast.Name):
                        graph.setdefault(n.id, set()).update(refs(stmt.value))
        elif isinstance(stmt, ast.For):
            reached = refs(stmt.iter)
            for r in list(reached):
                reached |= holders.get(r, set())
            if reached & (set(graph) | risky):
                for n in ast.walk(stmt.target):
                    if isinstance(n, ast.Name):
                        graph.setdefault(n.id, set()).update(reached)
    grew = True
    while grew:
        grew = False
        for name, r in graph.items():
            if name not in risky and r & risky:
                risky.add(name)
                grew = True
    risky -= set(allow) | {"_try"}
    routes = {"_try"}                  # module-level functions that hand what they are given to `_try`
    grew = True
    while grew:
        grew = False
        for name, n in funcs.items():
            if name not in routes and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    isinstance(c, ast.Call) and _call_name(c) in routes for c in ast.walk(n)):
                routes.add(name)
                grew = True
    hits = []

    def visit(node, func_position=False, refs_ok=False):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, (ast.List, ast.Tuple, ast.Dict, ast.Set)) \
                and all(isinstance(x, ast.Name) for x in (node.targets if isinstance(node, ast.Assign) else [node.target])):
            return                                           # a table of checks is data
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "_try":
                return                                       # its arguments run under _try's limit
            if name in ("exec", "eval") and isinstance(node.func, ast.Name):
                hits.append(f"{name}() on line {node.lineno}")
            elif name in INDIRECT and any(isinstance(a, ast.Name) and a.id in risky for a in node.args):
                hits.append(f"{name}(<student code>) on line {node.lineno}")
            elif (name if isinstance(node.func, ast.Name) else "." + name) in risky:
                hits.append(f"{name}() on line {node.lineno}")
            visit(node.func, True)
            for child in [*node.args, *[k.value for k in node.keywords]]:
                visit(child, refs_ok=name in routes)         # handed to something that runs it under `_try`
            return
        if isinstance(node, ast.For):                        # the iterable is covered by the alias rule above
            for child in [node.target, *node.body, *node.orelse]:
                visit(child)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in risky \
                and not func_position and not refs_ok:
            hits.append(f"a reference to {node.id} on line {node.lineno}")
        for child in ast.iter_child_nodes(node):
            visit(child, refs_ok=refs_ok)
    for stmt in student.body:
        visit(stmt)
    return hits


BLOCK = re.compile(r"(?ms)^# --- COMMONS TIME LIMIT v\d+.*?^# --- END COMMONS TIME LIMIT ---$")
LIMIT_LINE = re.compile(r"(?m)^(_CHECK_SECONDS|_DEMO_SECONDS|_MEMORY_MIB) = [\d.]+\s*#.*$")
TEMPLATE = ROOT / "tools" / "lesson_template" / "lesson.py"
CHECK_CAP, MEMORY_CAP, IMPORT_CAP = 10.0, 2048.0, 135.0    # seconds, MiB, seconds (tools/grade.py: 3 x 45 s)


def block_problems(d: Path) -> list:
    """A lesson that carries the COMMONS TIME LIMIT block carries the template's, byte for byte, but for the
    three limit lines: a copy that has drifted (or an older version) is not the guard that was reviewed."""
    canon = BLOCK.search(TEMPLATE.read_text())
    if not canon:
        return [f"gT: {TEMPLATE.relative_to(ROOT)} carries no COMMONS TIME LIMIT block"]
    canon = LIMIT_LINE.sub(r"\1 = N", canon.group(0))
    bad = []
    for rel in CHECK_FILES:
        f = d / rel
        if f.exists() and (m := BLOCK.search(f.read_text())) and LIMIT_LINE.sub(r"\1 = N", m.group(0)) != canon:
            bad.append(f"gT: {rel}: the COMMONS TIME LIMIT block differs from tools/lesson_template/lesson.py "
                       "(internal/retrofit_guard.py --upgrade, or copy the block unchanged)")
    return bad


def _constants(tree) -> dict:
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, SyntaxError):
                pass
    return out


def limit_problems(d: Path, meta: dict) -> list:
    """Gate T, limits: every lesson declares `_CHECK_SECONDS` (an exercise's check), `_DEMO_SECONDS` (any other
    cell) and `_MEMORY_MIB` as module-level numbers, its `_try` arms the guard with them, and they are sane: an
    exercise check stops within 10 s, a demo within the lesson's budget, memory growth within 2048 MiB, and every
    exercise cell looping must fit the grader's 135 s import limit. A lesson that needs more says why in
    meta.yaml `guard_limits_reason`."""
    bad = []
    reason = str(meta.get("guard_limits_reason") or "").strip()
    src = (d / "lesson.py").read_text()
    tree = ast.parse(src)
    consts = _constants(tree)
    has_block = bool(BLOCK.search(src))
    check, demo = consts.get("_CHECK_SECONDS"), consts.get("_DEMO_SECONDS")
    mem = consts.get("_MEMORY_MIB", None)
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (check, demo)):
        return [f"gT: lesson.py must declare numeric module-level _CHECK_SECONDS and _DEMO_SECONDS "
                "(an exercise's limit and every other cell's), and arm its guard with them"]
    if not has_block:
        # an own guard: _try must arm it with the declared constants, not with limits of its own
        funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        runner, guards = funcs.get("_try"), _guard_functions(tree)
        armed = [c for c in ast.walk(runner)
                 if isinstance(c, ast.Call) and _call_name(c) in guards and c.args] if runner else []
        fed = {}                    # local names that `_try` assigns from an expression naming the constants
        for n in (ast.walk(runner) if runner else ()):
            if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in ("_CHECK_SECONDS", "_DEMO_SECONDS")
                                                 for x in ast.walk(n.value)):
                for tgt in n.targets:
                    fed.update({x.id: True for x in ast.walk(tgt) if isinstance(x, ast.Name)})
        if not armed or not all(any(isinstance(n, ast.Name) and (n.id in ("_CHECK_SECONDS", "_DEMO_SECONDS") or n.id in fed)
                                    for n in ast.walk(c.args[0])) for c in armed):
            bad.append("gT: lesson.py's _try must arm its time limit with _CHECK_SECONDS / _DEMO_SECONDS, so the "
                       "declared limits are the ones in force")
        if mem is None:
            consts_mem = [c for c in armed if len(c.args) > 2 or any(k.arg == "memory_mib" for k in c.keywords)]
            if not consts_mem and "memory_mib: float = 2048" not in src:
                bad.append("gT: the time limit has no memory tripwire (_time_limit(..., memory_mib=2048))")
    elif mem is None:
        bad.append("gT: lesson.py's block declares no _MEMORY_MIB")
    by = consts.get("_CHECK_SECONDS_BY_LABEL") or {}     # limits of single exercises, for the few that need more
    if by and not (isinstance(by, dict) and all(isinstance(v, (int, float)) for v in by.values())):
        bad.append("gT: _CHECK_SECONDS_BY_LABEL must map exercise labels to numbers")
        by = {}
    budget = float(meta.get("budget_seconds") or 0)
    labels = set(re.findall(r"""_try\(\s*["'](exercise[^"']*)["']""", src))
    ex = consts.get("_EXERCISES")
    labels |= set(ex) if isinstance(ex, (dict, list, tuple, set)) else set()
    if not reason:
        if check > CHECK_CAP:
            bad.append(f"gT: _CHECK_SECONDS = {check:g} s is above {CHECK_CAP:g} s (a loop that never ends fails its row "
                       "only after that long); lower it or explain in meta.yaml guard_limits_reason")
        if budget and demo > budget:
            bad.append(f"gT: _DEMO_SECONDS = {demo:g} s is above the lesson's {budget:g} s budget "
                       "(meta.yaml guard_limits_reason explains an exception)")
        if mem is not None and mem > MEMORY_CAP:
            bad.append(f"gT: _MEMORY_MIB = {mem:g} is above {MEMORY_CAP:g}")
        if labels and sum(by.get(l, check) for l in labels) > IMPORT_CAP:
            bad.append(f"gT: {len(labels)} exercise cells x _CHECK_SECONDS = {check:g} s is more than the grader's "
                       f"{IMPORT_CAP:g} s import limit: one looping answer per cell would cost every row")
        over = sorted(l for l, v in by.items() if v > CHECK_CAP)
        if over:
            bad.append(f"gT: _CHECK_SECONDS_BY_LABEL gives {', '.join(over)} more than {CHECK_CAP:g} s "
                       "(meta.yaml guard_limits_reason explains an exception)")
    # per-call overrides: _try("...", check, limit=N) / seconds=N
    for c in ast.walk(tree):
        if isinstance(c, ast.Call) and _call_name(c) == "_try":
            for k in c.keywords:
                if k.arg in ("limit", "seconds"):
                    try:
                        v = ast.literal_eval(k.value)
                    except (ValueError, SyntaxError):
                        v = consts.get(k.value.id) if isinstance(k.value, ast.Name) else None
                    label = c.args[0].value if c.args and isinstance(c.args[0], ast.Constant) else ""
                    cap = CHECK_CAP if str(label).lower().startswith("exercise") else (budget or 1e9)
                    if not isinstance(v, (int, float)):
                        bad.append(f"gT: a _try call passes {k.arg}= that is not a number or a module constant")
                    elif v > cap and not reason:
                        bad.append(f"gT: _try({label!r}, ..., {k.arg}={v:g}) exceeds {cap:g} s "
                                   "(meta.yaml guard_limits_reason explains an exception)")
    return bad


def checks_guarded(d: Path) -> list:
    """Gate T, second half: every check the notebook runs is time-limited (see the module docstring), no student
    code runs outside `_try`, the guard is the template's, and its limits are sane."""
    import yaml
    bad = guard_wiring(d)
    meta = yaml.safe_load((d / "meta.yaml").read_text()) or {}
    hits = outside_try(d, tuple(meta.get("guard_allow") or ()))
    if hits:
        bad.append(f"gT: lesson.py runs student code outside _try, where no time limit reaches it: {', '.join(hits[:4])}"
                   + (f" (+{len(hits) - 4})" if len(hits) > 4 else "")
                   + "; route it through _try, or list the function in meta.yaml guard_allow once you have checked "
                     "that it runs none")
    return bad + block_problems(d) + child_problems(d, meta) + limit_problems(d, meta)


CHILD_BLOCK = re.compile(r"(?ms)^# --- COMMONS CHILD LIMIT v\d+.*?^# --- END COMMONS CHILD LIMIT ---$")
CHILD_TEMPLATE = ROOT / "tools" / "lesson_template" / "child_limit.py"


def child_problems(d: Path, meta: dict) -> list:
    """A C or C++ lesson runs the student's compiled binary, and `make test`, through `_run_child` (or under
    `_Ceiling`), from the COMMONS CHILD LIMIT block of tools/lesson_template/child_limit.py, copied unchanged: the
    time limit bounds how long a runaway answer runs, not how much memory a compiled loop fills (3 GB a second)."""
    if str(meta.get("language")) not in ("c", "cpp"):
        return []
    canon = CHILD_BLOCK.search(CHILD_TEMPLATE.read_text())
    bad = []
    for rel in CHECK_FILES:
        f = d / rel
        if not f.exists():
            continue
        text = f.read_text()
        m = CHILD_BLOCK.search(text)
        if not m:
            bad.append(f"gT: {rel} runs compiled code but carries no COMMONS CHILD LIMIT block (tools/lesson_template/child_limit.py)")
            continue
        if canon and m.group(0) != canon.group(0):
            bad.append(f"gT: {rel}: the COMMONS CHILD LIMIT block differs from tools/lesson_template/child_limit.py")
        tree = ast.parse(text)
        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if top.name in ("_group_mib", "_run_child"):
                    continue
                uses = any(isinstance(n, ast.Name) and n.id in ("_run_child", "_Ceiling") for n in ast.walk(top))
                for c in ast.walk(top):
                    if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name)
                            and c.func.value.id == "subprocess" and c.func.attr in ("run", "Popen", "check_output", "call")):
                        head = (c.args[0].elts[0] if c.args and isinstance(c.args[0], (ast.List, ast.Tuple)) and c.args[0].elts
                                else None)
                        toolchain = isinstance(head, ast.Constant) and head.value in (*TOOLCHAIN, "ps", "pip") \
                            and not {"test", "run", "check"} & {e.value for e in c.args[0].elts if isinstance(e, ast.Constant)}
                        given = isinstance(head, ast.Attribute) and head.attr == "executable"
                        if not (toolchain or given or uses):
                            bad.append(f"gT: {rel}: {top.name}() runs a program (line {c.lineno}) outside _run_child / _Ceiling")
    return bad


def guard_wiring(d: Path) -> list:
    """Does `_try` run every check under a guard, and does no cell call a `_check_` outside `_try`?"""
    bad = []
    for rel in CHECK_FILES:
        f = d / rel
        if not f.exists():
            continue
        tree = ast.parse(f.read_text())
        guards = _guard_functions(tree)
        funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        runner = funcs.get("_try")
        params = [a.arg for a in runner.args.args] if runner else []
        if len(params) < 2:
            bad.append(f"gT: {rel} has no `_try(label, check, ...)` runner, so nothing time-limits its checks")
            continue
        check = params[1]
        verdicts = _param_calls(runner, check, guards)
        # ...or _try hands `check` to a module-level helper that calls it under a guard
        for call in ast.walk(runner):
            helper = funcs.get(_call_name(call)) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) else None
            if helper is None or helper is runner:
                continue
            names = [a.arg for a in helper.args.args]
            passed = [names[i] for i, a in enumerate(call.args)
                      if i < len(names) and isinstance(a, ast.Name) and a.id == check]
            passed += [k.arg for k in call.keywords
                       if k.arg in names and isinstance(k.value, ast.Name) and k.value.id == check]
            for p in passed:
                verdicts += _param_calls(helper, p, guards)
        if not verdicts or not all(verdicts):
            bad.append(f"gT: {rel}: _try runs a check with no time limit; call it inside the canonical guard "
                       "(tools/lesson_template), so a loop that never ends fails its own row")
        loose = _loose_checks(tree, guards)
        if loose:
            bad.append(f"gT: {rel} calls {', '.join(loose[:3])} outside _try, with no time limit")
    return bad


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    wb = ["--write-back"] if "--write-back" in sys.argv else []
    rows, failed = [], 0
    every = list(lessons())
    known = {d.name for d in every}
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    if wanted:                                       # a few lessons only (internal/linux_check.py in a container)
        every = [d for d in every if any(d.name == Path(a).name for a in wanted)]
        missing = sorted(set(Path(a).name for a in wanted) - {d.name for d in every})
        if missing:
            print(f"  not lessons: {', '.join(missing)}")
            return 2
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
    backlog = guard_backlog()
    if backlog:
        print(f"  gate T: {len(backlog & known)} lessons still run some checks without a time limit and are "
              f"waived until retrofitted ({_backlog_name()})")
        stale = sorted(backlog - known)
        if stale:
            failed += 1
            print(f"  gate T: the backlog names {len(stale)} directories that are not lessons: {', '.join(stale)}")
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
