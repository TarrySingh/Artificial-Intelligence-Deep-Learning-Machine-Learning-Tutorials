#!/usr/bin/env python3
"""Student-facing autograder. Zero dependencies: runs on a plain Python 3.10+.

    python tools/grade.py lessons/<lesson-id>              # grade the student's work
    python tools/grade.py lessons/<lesson-id> --solution   # CI: the reference must score 100%

Each lesson's tests/test_lesson.py defines:

    RUBRIC = [(callable, points, hint), ...]

Every test is run in isolation; a failure never stops the rest. Partial credit is the
point: a student should always see which part works.
"""
import importlib.util, os, sys, traceback
from pathlib import Path


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        print(__doc__)
        return 2
    lesson = Path(args[0]).resolve()
    if "--solution" in sys.argv:
        # grade the reference implementation instead of the student stub file
        os.environ["COMMONS_LESSON_SRC"] = "solutions/lesson_solution.py"
    tests = lesson / "tests" / "test_lesson.py"
    if not tests.exists():
        print(f"no tests at {tests}")
        return 2
    sys.path.insert(0, str(lesson))
    mod = load(tests, "test_lesson")
    rubric = getattr(mod, "RUBRIC", None)
    if not rubric:
        print("tests/test_lesson.py defines no RUBRIC")
        return 2

    earned = total = 0
    rows = []
    for fn, points, hint in rubric:
        total += points
        try:
            fn()
            earned += points
            rows.append(("PASS", points, points, fn.__name__, ""))
        except NotImplementedError:
            rows.append(("TODO", 0, points, fn.__name__, "not implemented yet"))
        except AssertionError as e:
            rows.append(("FAIL", 0, points, fn.__name__, f"{e} | hint: {hint}"))
        except Exception:
            tb = traceback.format_exc(limit=1).strip().splitlines()[-1]
            rows.append(("ERROR", 0, points, fn.__name__, f"{tb} | hint: {hint}"))

    w = max(len(r[3]) for r in rows) + 2
    print()
    for status, got, pts, name, msg in rows:
        print(f"  [{status:5s}] {name:<{w}} {got}/{pts}  {msg}")
    pct = 100 * earned / total if total else 0
    print(f"\n  score: {earned}/{total}  ({pct:.0f}%)")
    if pct < 100:
        print("  keep going: fix the first FAIL, then re-run this grader.\n")
    else:
        print("  all checks green.\n")
    return 0 if pct == 100 else 1


if __name__ == "__main__":
    raise SystemExit(main())
