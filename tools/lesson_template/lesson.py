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
import contextlib, os, signal, sys, threading, time
print(sys.version.split()[0])


# A time limit for every check that calls the student's code: a loop that never ends must fail
# its check, not freeze the notebook or the grader (which imports the notebook). Copy this block
# unchanged; change only the message. Why it is this careful:
#  * Whoever runs the notebook may already have an alarm running (tools/grade.py stops every
#    rubric row after 45 s; a harness may have one whose default action ends the process). This
#    guard saves it, lets whichever deadline comes first win (handing the signal to the outer
#    handler when it is the outer's turn) and on the way out re-arms it with the time it had left
#    and its interval. Switching it off instead would let the NEXT hang run for ever.
#  * Its own alarm is one-shot, re-armed by hand every 0.25 s once the time is up (in case a bare
#    `except:` swallows the first). A repeating alarm kept firing while a runaway loop's huge list
#    was being torn down; a repeat that arrived after the default handler was back ended the
#    process (exit -14, a dead Jupyter kernel), and one that arrived inside contextlib's exit
#    escaped as a raw timeout and left this handler installed.
#  * It never raises inside its own set-up, exit or clean-up, and on the way out it takes any
#    alarm of its own that has gone off but not yet been handled (with SIGALRM blocked) before
#    the old handler is put back.
# tools/verify_all.py fails a lesson whose guard does less.
class _OutOfTime(BaseException):
    """Raised inside the guarded block when its time is up. A BaseException, so `except
    Exception:` in the code under test cannot swallow it; it repeats every 0.25 s in case a bare
    `except:` does. `_time_limit` turns it into an AssertionError once it has left that code."""


@contextlib.contextmanager
def _time_limit(seconds: float, what: str):
    """Fail the block with an AssertionError naming `what` if it runs longer than `seconds`.
    Uses the operating system's alarm (Linux and macOS, main thread); a no-op elsewhere."""
    if not (all(hasattr(signal, name) for name in ("setitimer", "SIGALRM", "pthread_sigmask", "sigpending", "sigwait"))
            and threading.current_thread() is threading.main_thread()):
        yield
        return
    message = (f"{what} was still running after {seconds:g} s: a loop whose stopping condition "
               "is never met runs for ever; give every loop a cap")
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    outer_left, outer_every = signal.getitimer(signal.ITIMER_REAL)   # an alarm already running
    # One alarm at a time, never a repeating one. next: when ours is due; outer: when the alarm we
    # found is due (every: its interval); unseen: an alarm is armed and has not reached _fire yet.
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
            raise _OutOfTime(message, state)
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
        raise AssertionError(message) from None
    if state["fired"]:                                   # the block caught our alarm and carried on
        raise AssertionError(message)

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
    _check_one()
