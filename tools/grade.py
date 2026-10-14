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
grader. All rows together get COMMONS_GRADE_TOTAL seconds (default 300): once earlier rows have
used that up, later rows are reported as not run, so several looping answers cannot keep the
grader busy for many minutes. This needs SIGALRM (Linux, macOS: every place these notebooks run); without it the
timeout is skipped.
"""
import contextlib, importlib.util, os, signal, sys, threading, time, traceback
from pathlib import Path

TIMEOUT = float(os.environ.get("COMMONS_GRADE_TIMEOUT", "45"))
TOTAL = float(os.environ.get("COMMONS_GRADE_TOTAL", "300"))


class _TooSlow(BaseException):
    """BaseException, so a student's `except Exception:` inside a hanging loop cannot swallow the alarm."""


@contextlib.contextmanager
def _deadline(seconds: float):
    """Raise _TooSlow in the block if it runs longer than `seconds` (Linux and macOS, main thread; a no-op
    elsewhere). The same mechanism as the canonical lesson guard (tools/lesson_template/lesson.py, which
    explains each step): nested-safe, so the previous handler and the time left on any previous timer, with
    its interval, are restored on exit, and an OUTER deadline that falls due sooner than ours wins (its own
    handler, or the default, which ends the process, gets the signal). Our alarm is one-shot, re-armed every
    0.25 s once the time is up so a swallowed first alarm is followed by another; it never raises inside this
    guard's own code, and an alarm of ours that went off but was not yet handled is taken before the old
    handler is put back (a late one used to end the process)."""
    if not (all(hasattr(signal, name) for name in ("setitimer", "SIGALRM", "pthread_sigmask", "sigpending", "sigwait"))
            and threading.current_thread() is threading.main_thread()):
        yield
        return
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    outer_left, outer_every = signal.getitimer(signal.ITIMER_REAL)   # an alarm already running
    state = {"next": started + seconds, "outer": started + outer_left if outer_left > 0 else None,
             "every": outer_every, "unseen": False, "fired": False, "done": False,
             "me": sys._getframe(), "owner": None}
    owner = state["me"].f_back
    while owner is not None and owner.f_globals is vars(contextlib):    # contextlib's __enter__
        owner = owner.f_back
    state["owner"] = owner                               # the frame running the `with`

    def _arm(now, soonest=0.001):                        # one shot, at whichever deadline is next
        due = [t for t in (state["next"], state["outer"]) if t is not None]
        if due:
            signal.setitimer(signal.ITIMER_REAL, max(min(due) - now, soonest))
            state["unseen"] = True

    def _in_guard(frame):                                # is the alarm landing in this guard's own
        while frame is not None and frame is not state["owner"]:     # set-up, exit or clean-up (or
            if (frame is state["me"] or frame.f_code in (_fire.__code__, _arm.__code__)    # in code
                    or (frame.f_back is state["owner"] and frame.f_globals is vars(contextlib))):
                return True                              # run from contextlib's __exit__ under the `with`)?
            frame = frame.f_back
        return False

    def _fire(signum, frame):
        state["unseen"] = False
        if state["done"]:
            return                                       # a late alarm of ours: already handled
        if _in_guard(frame):                             # never raise inside the guard: it is
            _arm(time.monotonic(), 0.01)                 # finishing; look again shortly
            return
        now = time.monotonic()
        if state["outer"] is not None and now >= state["outer"] - 0.005:
            every = state["every"]                       # the outer alarm is due: act as it would
            state["outer"] = state["outer"] + every * (1 + (now - state["outer"]) // every) if every else None
            try:
                if callable(previous):
                    previous(signum, frame)
                elif previous != signal.SIG_IGN:         # its default action ends the process
                    signal.signal(signal.SIGALRM, signal.SIG_DFL)
                    os.kill(os.getpid(), signal.SIGALRM)
            except BaseException:
                state["next"] = None                     # the outer one is stopping the block
                raise
            finally:
                left, every = signal.getitimer(signal.ITIMER_REAL)
                if left > 0:                             # the outer handler armed its next alarm
                    state["outer"], state["every"] = time.monotonic() + left, every
                _arm(time.monotonic())
            now = time.monotonic()
        if state["next"] is not None and now >= state["next"] - 0.005:
            state["fired"] = True
            state["next"] = now + 0.25                   # again, in case a bare `except:` swallows it
            _arm(now)
            raise _TooSlow(state)
        _arm(now)

    signal.signal(signal.SIGALRM, _fire)
    _arm(time.monotonic())
    try:
        try:
            yield
        finally:
            state["done"] = True                         # from here on an alarm of ours only counts
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            try:
                if signal.setitimer(signal.ITIMER_REAL, 0)[0] == 0 and state["unseen"]:
                    give_up = time.monotonic() + 0.5     # ours went off, not yet handled: take it
                    while state["unseen"] and time.monotonic() < give_up:
                        if signal.SIGALRM in signal.sigpending():
                            signal.sigwait({signal.SIGALRM})
                            state["unseen"] = False
                        else:
                            time.sleep(0.001)            # another thread has it: _fire runs next
                signal.signal(signal.SIGALRM, signal.SIG_DFL if previous is None else previous)
                outer = state["outer"]                   # re-arm the alarm we found, last
                signal.setitimer(signal.ITIMER_REAL, *((max(outer - time.monotonic(), 0.001), state["every"])
                                                       if outer is not None else (0, 0)))
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
                state["me"] = state["owner"] = None
    except _TooSlow as exc:
        if not exc.args or exc.args[-1] is not state:    # an enclosing deadline's: not ours to report
            raise
        raise _TooSlow() from None
    if state["fired"]:                                   # the block caught our alarm and carried on
        raise _TooSlow()


def _run_with_timeout(fn, seconds=None):
    """Call fn(); raise _TooSlow if it takes longer than `seconds` (default TIMEOUT; where SIGALRM exists)."""
    seconds = TIMEOUT if seconds is None else seconds
    if seconds <= 0:
        return fn()
    with _deadline(seconds):
        return fn()


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
    budget_end = time.monotonic() + TOTAL
    for fn, points, hint in rubric:
        total += points
        left = budget_end - time.monotonic()
        if left <= 0:
            rows.append(("FAIL", 0, points, fn.__name__,
                         f"not run: earlier rows used up the grader's {TOTAL:g} s total budget (look for a loop "
                         f"that never stops, fix it, re-run) | hint: {hint}"))
            continue
        try:
            _run_with_timeout(fn, min(TIMEOUT, left))
            earned += points
            rows.append(("PASS", points, points, fn.__name__, ""))
        except NotImplementedError:
            rows.append(("TODO", 0, points, fn.__name__, "not implemented yet"))
        except _TooSlow:
            rows.append(("FAIL", 0, points, fn.__name__,
                         f"was stopped after {min(TIMEOUT, max(left, 0)):.3g} s"
                         f" (row limit {TIMEOUT:g} s, total budget {TOTAL:g} s): an infinite loop, or a loop that never "
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
