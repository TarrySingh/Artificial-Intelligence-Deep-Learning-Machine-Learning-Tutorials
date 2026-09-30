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
# unchanged; change only the message. Why it is this careful: whoever runs the notebook may
# already have an alarm running (tools/grade.py stops every rubric row after 45 s and repeats
# its alarm every 0.25 s; a harness may have an alarm that ends the process). Earlier guards
# switched that alarm off on the way out (`setitimer(ITIMER_REAL, 0)`), so the NEXT hang ran for
# ever. This one saves the running alarm, lets whichever deadline comes first win (handing the
# signal to the outer handler when it is the outer's turn), and on the way out re-arms the outer
# alarm with the time it had left and its interval. tools/verify_all.py fails a lesson whose
# guard does less.
class _OutOfTime(BaseException):
    """Raised inside the guarded block when its time is up. A BaseException, so `except
    Exception:` in the code under test cannot swallow it; it repeats every 0.25 s in case a bare
    `except:` does. `_time_limit` turns it into an AssertionError once it has left that code."""


@contextlib.contextmanager
def _time_limit(seconds: float, what: str):
    """Fail the block with an AssertionError naming `what` if it runs longer than `seconds`.
    Uses the operating system's alarm (Linux and macOS, main thread); a no-op elsewhere."""
    if not (hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()):
        yield
        return
    message = (f"{what} was still running after {seconds:g} s: a loop whose stopping condition "
               "is never met runs for ever; give every loop a cap")
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGALRM)
    outer_left, outer_every = signal.getitimer(signal.ITIMER_REAL)   # an alarm already running
    state = {"done": False, "handed": False, "fired": False}

    def _fire(signum, frame):
        if state["done"]:
            return
        elapsed = time.monotonic() - started
        if outer_left > 0 and elapsed >= outer_left - 0.01 and (outer_every or not state["handed"]):
            state["handed"] = True                       # the outer alarm is due: act as it would
            if callable(previous):
                previous(signum, frame)
            elif previous != signal.SIG_IGN:             # its default action ends the process
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGALRM)
        if elapsed >= seconds - 0.01:
            state["fired"] = True
            raise _OutOfTime(message, state)
        wait = seconds - elapsed                         # the outer handler returned: go on
        signal.setitimer(signal.ITIMER_REAL, max(min(wait, outer_every or wait), 0.001), 0.25)

    def _outer_rest():                                   # what the outer alarm has left now
        if outer_left <= 0:
            return 0.0, 0.0
        late = time.monotonic() - started - outer_left
        if not state["handed"]:
            return max(-late, 0.001), outer_every
        if outer_every:
            return max(outer_every - late % outer_every, 0.001), outer_every
        return 0.0, 0.0

    signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, min(seconds, outer_left) if outer_left > 0 else seconds, 0.25)
    try:
        try:
            yield
        finally:
            state["done"] = True
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            try:
                signal.setitimer(signal.ITIMER_REAL, *_outer_rest())
                signal.signal(signal.SIGALRM, signal.SIG_DFL if previous is None else previous)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
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
