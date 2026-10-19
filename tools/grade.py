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
grader. So is a test that makes the process's resident memory grow by more than
COMMONS_GRADE_MEMORY_MIB (default 2048): an answer that appends to a list for ever reaches
gigabytes in seconds, long before the time limit. All rows together get COMMONS_GRADE_TOTAL seconds
(default 300): once earlier rows have used that up, later rows are reported as not run, so several
looping answers cannot keep the grader busy for many minutes. This needs SIGALRM (Linux, macOS:
every place these notebooks run); without it the limits are skipped. A loop wrapped in a bare
`except:` is stopped too (see _deadline).

A row whose honest work is slow on an emulated or shared CPU may ask for more than the default by
setting an attribute on its test function, `test_x.grade_seconds = 120`. It never gets less than
COMMONS_GRADE_TIMEOUT, and the total budget above still caps it. A grader without this feature
simply ignores the attribute and applies the default.
"""
import contextlib, gc, importlib.util, os, signal, sys, threading, time, traceback
from pathlib import Path

TIMEOUT = float(os.environ.get("COMMONS_GRADE_TIMEOUT", "45"))
TOTAL = float(os.environ.get("COMMONS_GRADE_TOTAL", "300"))
MEMORY_MIB = float(os.environ.get("COMMONS_GRADE_MEMORY_MIB", "2048"))


class _TooSlow(BaseException):
    """BaseException, so a student's `except Exception:` inside a hanging loop cannot swallow the alarm.
    args[0] says why (time or memory) once the block has been left."""


def _resident() -> float:
    """This process's resident memory now, in MiB (Linux: /proc; macOS: libproc); its peak so far elsewhere. The
    NOW matters: a peak only rises, so after one runaway row the next would be allowed another `memory_mib` on top."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
        if sys.platform == "darwin":
            import ctypes, struct
            info = ctypes.create_string_buffer(96)           # struct proc_taskinfo
            if ctypes.CDLL("/usr/lib/libproc.dylib").proc_pidinfo(os.getpid(), 4, 0, info, 96) == 96:
                return struct.unpack_from("QQ", info.raw)[1] / 2**20     # pti_resident_size
    except Exception:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)


@contextlib.contextmanager
def _deadline(seconds: float, memory_mib: float = 0):
    """Raise _TooSlow in the block if it runs longer than `seconds`, or (when `memory_mib` is set) if the
    process's resident memory grows by more than that while it runs (Linux and macOS, main thread; a no-op
    elsewhere). The same mechanism as the canonical lesson guard (tools/lesson_template/lesson.py, which
    explains each step): nested-safe, so the previous handler and the time left on any previous timer, with
    its interval, are restored on exit, and an OUTER deadline that falls due sooner than ours wins (its own
    handler, or the default, which ends the process, gets the signal). Our alarm is one-shot, re-armed every
    0.25 s once the time is up so a swallowed first alarm is followed by another, and from the second stop on
    a line tracer raises it again at the next line the code runs (inside the handler of a bare `except:` that
    swallowed the first); it never raises inside this guard's own code, and an alarm of ours that went off
    but was not yet handled is taken before the old handler is put back (a late one used to end the process)."""
    if not (all(hasattr(signal, name) for name in ("setitimer", "SIGALRM", "pthread_sigmask", "sigpending", "sigwait"))
            and threading.current_thread() is threading.main_thread()):
        yield
        return
    if getattr(_deadline, "stopped", False):             # a stopped row's frames keep what they built in a
        _deadline.stopped = False                        # reference cycle until now: free it, so it is not counted
        gc.collect()                                     # as the growth of this row
    started = time.monotonic()
    used_mib = None
    if memory_mib:
        used_mib = _resident
        base_mib = used_mib()                            # resident memory now; only growth beyond it counts
    previous = signal.getsignal(signal.SIGALRM)
    outer_left, outer_every = signal.getitimer(signal.ITIMER_REAL)   # an alarm already running
    state = {"next": started + seconds, "outer": started + outer_left if outer_left > 0 else None,
             "every": outer_every, "unseen": False, "fired": False, "done": False,
             "message": f"was stopped after {seconds:g} s", "count": 0, "budget": 0, "traced": False, "tracer": None, "tool": None,
             "at": None, "thread": threading.get_ident(), "me": sys._getframe(), "owner": None}
    owner = state["me"].f_back
    while owner is not None and owner.f_globals is vars(contextlib):    # contextlib's __enter__
        owner = owner.f_back
    state["owner"], state["at"] = owner, owner.f_lineno   # the frame running the `with`, and its line

    def _arm(now, soonest=0.001):                        # one shot, at whichever deadline is next
        due = [t for t in (state["next"], state["outer"]) if t is not None]
        if used_mib and state["next"] is not None and not state["fired"]:
            due.append(now + 0.1)                        # and look at the memory every 0.1 s
        if due:
            signal.setitimer(signal.ITIMER_REAL, max(min(due) - now, soonest))
            state["unseen"] = True

    def _in_guard(frame):                                # is the alarm landing in a guard's own set-up, exit
        while frame is not None and frame is not state["owner"]:     # or clean-up (this one's or another's:
            code = frame.f_code                          # guards nest), or in contextlib's __exit__ under
            if (frame is state["me"] or code in _own_code or code.co_name in _GUARDS   # a `with`?
                    or frame.f_globals.get("__name__") == "contextlib"):
                return True
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
        grown = used_mib() - base_mib if used_mib and not state["fired"] else 0
        if grown > memory_mib and state["next"] is not None:       # out of memory before out of time
            state["message"] = f"was stopped after {now - started:.1f} s: its memory grew by over {memory_mib:g} MiB"
            state["next"] = now
        if state["next"] is not None and now >= state["next"] - 0.005:
            swallowed = state["fired"]                   # a second stop: the first did not end the block
            state["fired"] = True
            state["count"] += 1
            state["next"] = now + 0.25                   # again, in case a bare `except:` swallows it
            _arm(now)
            if swallowed:
                state["budget"] = state["count"] - 1     # the tracer raises this many times, then leaves clean-up alone
                _trace_from(frame)
            raise _TooSlow(state["message"], state)
        _arm(now)

    def _trace(frame, event, arg):                       # raises the stop at the next line of code that
        if state["done"] or _in_guard(frame):            # runs: inside the handler that swallowed the last
            return None                                  # (the interpreter then unsets it; the next alarm
        if event == "line":                              # sets it again)
            raise _TooSlow(state["message"], state)
        return _trace

    def _line(code, line):                               # Python 3.12+: sys.monitoring, which is not unset
        if state["done"] or state["budget"] <= 0 or threading.get_ident() != state["thread"]:   # not unset when a
            return                                       # callback raises: the next line the code runs raises too, handlers
        frame = sys._getframe(1)                         # included (this thread only), but only `budget` times:
        if _in_guard(frame) or (frame is state["owner"] and line == state["at"]):   # clean-up that runs once the stop has
            return                                       # left the swallowing loops must not be interrupted again
        state["budget"] -= 1                             # (not the `with` line: its exit runs the guard)
        raise _TooSlow(state["message"], state)

    def _trace_from(frame):                              # trace the running frames, and the ones called next
        if not state["traced"]:
            state["traced"] = True
            mon = getattr(sys, "monitoring", None)
            for tool in (5, 4, 3, 2, 1, 0) if mon is not None else ():
                try:
                    mon.use_tool_id(tool, "commons-grade-deadline")
                except ValueError:                       # a debugger or a profiler has this one
                    continue
                state["tool"] = tool
                mon.register_callback(tool, mon.events.LINE, _line)
                mon.set_events(tool, mon.events.LINE)
                return
            state["tracer"] = sys.gettrace()
            sys.settrace(_trace)
        while frame is not None and frame is not state["owner"]:
            frame.f_trace = _trace
            frame = frame.f_back

    _GUARDS = ("_time_limit", "_deadline", "_resident", "_fire", "_arm", "_trace", "_trace_from", "_line", "_in_guard")
    _own_code = (_fire.__code__, _arm.__code__, _trace.__code__, _trace_from.__code__, _line.__code__)
    signal.signal(signal.SIGALRM, _fire)
    _arm(time.monotonic())
    try:
        try:
            yield
        finally:
            state["done"] = True                         # from here on an alarm of ours only counts
            if state["traced"]:
                state["traced"] = False
                if state["tool"] is not None:
                    sys.monitoring.set_events(state["tool"], 0)
                    sys.monitoring.register_callback(state["tool"], sys.monitoring.events.LINE, None)
                    sys.monitoring.free_tool_id(state["tool"])
                else:
                    sys.settrace(state["tracer"])        # the tracer that was there before, if any
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
        _deadline.stopped = True
        raise _TooSlow(state["message"]) from None
    if state["fired"]:                                   # the block caught our alarm and carried on
        _deadline.stopped = True
        raise _TooSlow(state["message"])


def _run_with_timeout(fn, seconds=None, memory_mib=None):
    """Call fn(); raise _TooSlow if it takes longer than `seconds` (default TIMEOUT) or its resident memory grows
    by more than `memory_mib` (default MEMORY_MIB); where SIGALRM exists."""
    seconds = TIMEOUT if seconds is None else seconds
    if seconds <= 0:
        return fn()
    with _deadline(seconds, MEMORY_MIB if memory_mib is None else memory_mib):
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
    except _TooSlow as stop:
        why = stop.args[0] if stop.args else f"was stopped after {3 * TIMEOUT:g} s"
        print(f"importing the lesson {why}: a check in the notebook is looping forever or keeps growing a list. "
              "Find the loop that never reaches its stopping condition, then re-run.")
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
        limit = max(TIMEOUT, float(getattr(fn, "grade_seconds", 0) or 0))   # a row may ask for longer
        gc.collect()      # what the last stopped row built stays in a reference cycle until now; it is not this row's growth
        try:
            _run_with_timeout(fn, min(limit, left))
            earned += points
            rows.append(("PASS", points, points, fn.__name__, ""))
        except NotImplementedError:
            rows.append(("TODO", 0, points, fn.__name__, "not implemented yet"))
        except _TooSlow as stop:
            why = stop.args[0] if stop.args else f"was stopped after {min(limit, max(left, 0)):.3g} s"
            rows.append(("FAIL", 0, points, fn.__name__,
                         f"{why} (row limit {limit:g} s, total budget {TOTAL:g} s, memory growth limit "
                         f"{MEMORY_MIB:g} MiB): an infinite loop, a list that grows for ever, or a loop that never "
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
