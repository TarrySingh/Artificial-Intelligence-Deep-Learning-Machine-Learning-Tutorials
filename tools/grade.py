#!/usr/bin/env python3
"""Student-facing autograder. Zero dependencies: runs on a plain Python 3.10+.

    python tools/grade.py lessons/<lesson-id>              # grade the student's work
    python tools/grade.py lessons/<lesson-id> --solution   # CI: the reference must score 100%

Each lesson's tests/test_lesson.py defines:

    RUBRIC = [(callable, points, hint), ...]

Every test is run in isolation; a failure never stops the rest. Partial credit is the
point: a student should always see which part works.

A test that runs longer than COMMONS_GRADE_TIMEOUT seconds (default 45) is stopped and
scored as a failure: an answer that loops forever must cost its own points, not hang the
grader. This needs SIGALRM (Linux, macOS: every place these notebooks run); without it the
timeout is skipped.
"""
import importlib.util, os, signal, sys, time, traceback
from pathlib import Path

TIMEOUT = float(os.environ.get("COMMONS_GRADE_TIMEOUT", "45"))


class _TooSlow(BaseException):
    """BaseException, so a student's `except Exception:` inside a hanging loop cannot swallow the alarm."""


def _run_with_timeout(fn, seconds=None):
    """Call fn(); raise _TooSlow if it takes longer than `seconds` (default TIMEOUT; where SIGALRM exists).

    Nested-safe: the previous handler and the time left on any previous timer are restored on exit, and an
    OUTER deadline that falls due sooner than ours wins (its own handler, or the default, which ends the
    process, gets the signal), so an external `alarm` around this grader is never postponed. Our alarm
    repeats every 0.25 s, so a student's swallowed first alarm is followed by another."""
    seconds = TIMEOUT if seconds is None else seconds
    if not hasattr(signal, "setitimer") or seconds <= 0:
        return fn()
    started = time.monotonic()
    old_left, old_every = signal.getitimer(signal.ITIMER_REAL)
    outer_first = 0 < old_left < seconds
    old_handler = signal.getsignal(signal.SIGALRM)

    def _alarm(signum, frame):
        if outer_first:                      # the outer deadline is due: hand the signal back to its owner
            signal.signal(signal.SIGALRM, old_handler)
            os.kill(os.getpid(), signal.SIGALRM)
            return
        raise _TooSlow()
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, old_left if outer_first else seconds, 0.25)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_left > 0:
            signal.setitimer(signal.ITIMER_REAL, max(old_left - (time.monotonic() - started), 1e-3), old_every)


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
    try:
        # importing the lesson can itself run the student's code (the notebook's checks): time-box that too
        mod = _run_with_timeout(lambda: load(tests, "test_lesson"), 3 * TIMEOUT)
    except _TooSlow:
        print(f"importing the lesson took longer than {3 * TIMEOUT:g} s and was stopped: a check in the notebook "
              "is looping forever. Find the loop that never reaches its stopping condition, then re-run.")
        return 1
    rubric = getattr(mod, "RUBRIC", None)
    if not rubric:
        print("tests/test_lesson.py defines no RUBRIC")
        return 2

    earned = total = 0
    rows = []
    for fn, points, hint in rubric:
        total += points
        try:
            _run_with_timeout(fn)
            earned += points
            rows.append(("PASS", points, points, fn.__name__, ""))
        except NotImplementedError:
            rows.append(("TODO", 0, points, fn.__name__, "not implemented yet"))
        except _TooSlow:
            rows.append(("FAIL", 0, points, fn.__name__,
                         f"took longer than {TIMEOUT:g} s and was stopped: an infinite loop, or a loop that never "
                         f"reaches its stopping condition? | hint: {hint}"))
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
