# %% [markdown]
# # <Lesson title: the thing they will build>
#
# **You will build:** one sentence naming the artefact.
# **Time:** ~NN minutes · **Runs on:** a laptop CPU, 8 GiB RAM · **Prerequisites:** none
#
# By the end you will be able to:
# 1. Implement ...
# 2. Measure ...
# 3. Explain why ...

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import sys
print(sys.version.split()[0])


# --- COMMONS TIME LIMIT v4 --- every check in this notebook runs under a time limit
import contextlib, gc, os, signal, subprocess, sys, threading, time


# A time limit for every check that calls the student's code: a loop that never ends must fail
# its check, not freeze the notebook or the grader (which imports the notebook). Copy this block
# unchanged; change only the three limits below it. Why it is this careful:
#  * Whoever runs the notebook may already have an alarm running (tools/grade.py stops every
#    rubric row after 45 s; a harness may have one whose default action ends the process). This
#    guard saves it, lets whichever deadline comes first win (handing the signal to the outer
#    handler when it is the outer's turn) and on the way out re-arms it with the time it had left
#    and its interval. Switching it off instead would let the NEXT hang run for ever.
#  * Its own alarm is one-shot, re-armed by hand every 0.25 s once the time is up (in case a bare
#    `except:` swallows the first). A second alarm alone is not enough: `while True: try: ...
#    except: pass` is interrupted inside the `try` every time and carries on. So from the second
#    stop on, a line tracer (sys.settrace, on the running frames too) raises the stop again at the
#    first line the code runs afterwards, which is in the handler, outside the `try`; it is put
#    back as it was on the way out. A repeating alarm kept firing while a runaway loop's huge list
#    was being torn down; a repeat that arrived after the default handler was back ended the
#    process (exit -14, a dead Jupyter kernel), and one that arrived inside contextlib's exit
#    escaped as a raw timeout and left this handler installed.
#    On Python 3.12+ that tracer is sys.monitoring, which stays on after it raises, so a loop
#    inside TWO bare `except:` blocks is stopped as well: the Nth stop lets it raise N-1 times in
#    a row (one per handler it has to leave), and no more, so the `finally:` blocks and context
#    managers that run as the stop travels out are not interrupted again. On 3.11 sys.settrace is
#    unset by the interpreter each time it raises, so only the innermost one is peeled per alarm.
#  * It also watches memory: every 0.1 s it compares the process's resident memory with what it
#    was when the block started, and stops the block as soon as the growth passes `memory_mib`.
#    A loop that appends to a list reaches gigabytes in seconds, long before any sensible time
#    limit, so the time limit alone does not bound it. (Resident memory NOW, not the peak so
#    far: a peak only ever rises, so after one runaway check the next would be allowed another
#    `memory_mib` on top of it, and five looping exercises would reach ten gigabytes. Where the
#    system will not say, /proc on Linux and libproc on macOS being the two that do, it falls
#    back to the peak.)
#  * It never raises inside its own set-up, exit or clean-up (nor inside another guard's: guards
#    nest, the grader's wraps this one), and on the way out it takes any alarm of its own that has
#    gone off but not yet been handled (with SIGALRM blocked) before the old handler is put back.
# tools/verify_all.py fails a lesson whose guard does less.
class _OutOfTime(BaseException):
    """Raised inside the guarded block when its time is up. A BaseException, so `except
    Exception:` in the code under test cannot swallow it; it repeats every 0.25 s, and from the
    second time on a line tracer raises it again at the next line the code runs (inside the
    handler that swallowed the first), in case a bare `except:` does. `_time_limit` turns it into
    an AssertionError once it has left that code."""


def _resident() -> float:
    """This process's resident memory now, in MiB (Linux: /proc; macOS: libproc); its peak so far elsewhere."""
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
def _time_limit(seconds: float, what: str, memory_mib: float = 0):
    """Fail the block with an AssertionError naming `what` if it runs longer than `seconds`, or
    (when `memory_mib` is set) if the process's resident memory grows by more than that while it runs.
    Uses the operating system's alarm (Linux and macOS, main thread); a no-op elsewhere."""
    if not (all(hasattr(signal, name) for name in ("setitimer", "SIGALRM", "pthread_sigmask", "sigpending", "sigwait"))
            and threading.current_thread() is threading.main_thread()):
        yield
        return
    if getattr(_time_limit, "stopped", False):           # a stopped check's frames keep what they built in a
        _time_limit.stopped = False                      # reference cycle until now: free it, so it is not counted
        gc.collect()                                     # as the growth of this block
    message = (f"{what} was still running after {seconds:g} s: a loop whose stopping condition "
               "is never met runs for ever; give every loop a cap")
    started = time.monotonic()
    used_mib = None
    if memory_mib:
        used_mib = _resident
        base_mib = used_mib()                            # resident memory now; only growth beyond it counts
    previous = signal.getsignal(signal.SIGALRM)
    outer_left, outer_every = signal.getitimer(signal.ITIMER_REAL)   # an alarm already running
    # One alarm at a time, never a repeating one. next: when ours is due; outer: when the alarm we
    # found is due (every: its interval); unseen: an alarm is armed and has not reached _fire yet.
    state = {"next": started + seconds, "outer": started + outer_left if outer_left > 0 else None,
             "every": outer_every, "unseen": False, "fired": False, "done": False, "message": message, "count": 0, "budget": 0,
             "traced": False, "tracer": None, "tool": None, "at": None, "thread": threading.get_ident(),
             "me": sys._getframe(), "owner": None}
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
            state["message"] = (f"{what} was using over {memory_mib:g} MiB more memory than when it started, "
                                f"after {now - started:.1f} s: a loop that keeps adding to a list or a string "
                                "never stops; give every loop a cap")
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
            raise _OutOfTime(state["message"], state)
        _arm(now)

    def _trace(frame, event, arg):                       # raises the stop at the next line of code that
        if state["done"] or _in_guard(frame):            # runs: inside the handler that swallowed the last
            return None                                  # (the interpreter then unsets it; the next alarm
        if event == "line":                              # sets it again)
            raise _OutOfTime(state["message"], state)
        return _trace

    def _line(code, line):                               # Python 3.12+: sys.monitoring, which is not unset
        if state["done"] or state["budget"] <= 0 or threading.get_ident() != state["thread"]:   # not unset when a
            return                                       # callback raises: the next line the code runs raises too, handlers
        frame = sys._getframe(1)                         # included (this thread only), but only `budget` times:
        if _in_guard(frame) or (frame is state["owner"] and line == state["at"]):   # clean-up that runs once the stop has
            return                                       # left the swallowing loops must not be interrupted again
        state["budget"] -= 1                             # (not the `with` line: its exit runs the guard)
        raise _OutOfTime(state["message"], state)

    def _trace_from(frame):                              # trace the running frames, and the ones called next
        if not state["traced"]:
            state["traced"] = True
            mon = getattr(sys, "monitoring", None)
            for tool in (5, 4, 3, 2, 1, 0) if mon is not None else ():
                try:
                    mon.use_tool_id(tool, "commons-time-limit")
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
                    # Ours went off and has not reached _fire: it is pending here, or on its way to
                    # another thread. Take it before the old handler (by default, one that ends the
                    # process) is back.
                    give_up = time.monotonic() + 0.5
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
    except _OutOfTime as exc:
        if exc.args[-1] is not state:                    # an enclosing guard's: not ours to report
            raise
        _time_limit.stopped = True
        raise AssertionError(state["message"]) from None
    if state["fired"]:                                   # the block caught our alarm and carried on
        _time_limit.stopped = True
        raise AssertionError(state["message"])


# The three limits. A correct answer needs a small fraction of the first two (at least five times
# less, measured on the worked solution, on the slower of a Mac and an emulated Linux machine), so
# only a loop that never stops, or one that keeps appending until memory runs out, ever meets one.
_CHECK_SECONDS = 2     # an exercise's check
_DEMO_SECONDS = 10     # any other cell that runs your code: a demo, a table, a figure
_MEMORY_MIB = 2048     # how far a check may grow the process's resident memory before it is stopped


def _limited(label: str, check):
    """Run `check()` under `_time_limit` and return what it returns: _CHECK_SECONDS for an
    exercise's check, _DEMO_SECONDS for anything else. A check that is stopped (too long, or too
    much memory) fails with an AssertionError that says so; whatever the stopped code had built (a
    runaway list can hold gigabytes) is freed at once, not whenever the garbage collector next
    runs; and every program it had started that is still running is stopped too."""
    seconds = _CHECK_SECONDS if label.lower().startswith("exercise") else _DEMO_SECONDS
    before = _processes()
    try:
        with _time_limit(seconds, "this check", _MEMORY_MIB):
            return check()
    except AssertionError as exc:
        if not str(exc).startswith("this check was "):
            raise                                        # the check's own verdict, unchanged
        message = str(exc)
    gc.collect()          # the stopped frames hold what they built in a reference cycle until now
    _stop_strays(before)
    raise AssertionError(message)


def _processes() -> dict:
    """{pid: (parent, process group)} of every process (Linux: /proc; macOS: ps); {} if unreadable."""
    table = {}
    try:
        if os.path.isdir("/proc/self"):
            for entry in os.listdir("/proc"):
                if entry.isdigit():
                    try:
                        with open(f"/proc/{entry}/stat") as f:
                            fields = f.read().rsplit(")", 1)[1].split()
                        table[int(entry)] = (int(fields[1]), int(fields[2]))
                    except (OSError, ValueError, IndexError):
                        continue
        else:
            listing = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,pgid="], capture_output=True, text=True,
                                     timeout=10).stdout
            for row in listing.splitlines():
                pid, ppid, pgid = (int(x) for x in row.split())
                table[pid] = (ppid, pgid)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
    return table


def _stop_strays(before: dict) -> None:
    """Stopping a check stops the program it was waiting for, but not what THAT program started:
    stop `make test` and the test binary make launched runs on, at full speed, for ever. Stop every
    process that did not exist before the check (`before`, from _processes()) and either descends
    from this one or sits in its process group (an orphan: its parent is gone, so it now belongs to
    pid 1 or to a subreaper, whichever this machine has). Nothing that was already running is
    touched, the launcher that started this notebook included. Best effort."""
    now, me, group = _processes(), os.getpid(), os.getpgrp()
    if not before or not now:
        return
    def descends(pid):
        seen = set()
        while pid in now and pid not in seen:
            seen.add(pid)
            pid = now[pid][0]
            if pid == me:
                return True
        return False
    for pid, (_, pgid) in now.items():
        if pid != me and pid not in before and (pgid == group or descends(pid)):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
# --- END COMMONS TIME LIMIT ---

# %% [markdown]
# ## 1. The idea, in the smallest form that is still true
#
# Short. Then straight into something that runs.

# %%
# A cell the student RUNS to see the phenomenon before they are asked to build it.

# %% [markdown]
# ## 2. Exercise 1 — <what they implement>
#
# Fill in the function. Run the checks below it; they tell you what is wrong, not just that
# something is.

# %%
def exercise_one(x):
    """One line on what it returns.

    Example:
        >>> exercise_one(2)
        4
    """
    # YOUR CODE HERE
    raise NotImplementedError


# Public checks — run these as often as you like.
def _check_one():
    assert exercise_one(2) == 4, "exercise_one(2) should be 4 — are you returning, not printing?"
    print("exercise 1 looks right")


# %% [markdown]
# ## 3. Common mistakes
#
# - The one that catches most people, and how to spot it.

# %% [markdown]
# ## 4. Self-check
#
# 1. Question probing the misconception this lesson exists to fix.
#    - (a) ... (b) ... (c) ...
#
# Answers in `solutions/`.

# %% [markdown]
# ## What you built, and where it goes next
#
# One sentence tying this to the flagship or capstone it feeds.

# %%
if __name__ == "__main__":
    _limited("exercise 1", _check_one)
