# %% [markdown]
# # T00-L01 · The 8 GB track: measure before you believe
#
# **You will build:** `measure()`, a profiler that reports what a piece of code actually
# costs, and `tier_check()`, the gate this repository runs against every lesson it ships.
#
# **Time:** ~40 minutes · **Runs on:** a laptop CPU, 8 GiB RAM, no GPU, no download
# · **Prerequisites:** none — this is the first lesson in the Atlas.
#
# Every other lesson here claims it fits in 8 GiB and ten minutes. This is the lesson that
# makes those claims checkable, so it comes first.
#
# By the end you will be able to:
#
# 1. Implement `measure(fn)` reporting wall time, Python-allocator peak and process RSS.
# 2. Measure the gap between what `tracemalloc` sees and what the operating system charges.
# 3. Profile four implementations of one task and rank them on time, memory *and* correctness.
# 4. Implement `tier_check()` so it reproduces this repository's own gate, boundary included.
# 5. Explain why a lesson that busts its budget gets rewritten, not given a bigger budget.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import mmap
import resource
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Callable, Mapping, NamedTuple

import numpy as np

MIB = 1024 * 1024
_LESSON_T0 = time.perf_counter()
print("python", sys.version.split()[0], "· numpy", np.__version__, "· platform", sys.platform)

# The two clocks this lesson is careful to keep apart, described by the interpreter itself
# rather than by me. One of them cannot go backwards; the other one can.
_PERF, _WALL = time.get_clock_info("perf_counter"), time.get_clock_info("time")
print(f"perf_counter: monotonic={_PERF.monotonic}, resolution {_PERF.resolution:.0e}s"
      f"  ·  time(): monotonic={_WALL.monotonic}")

# `ru_maxrss` is not in the same unit everywhere: the Linux man page documents kibibytes and
# the macOS and FreeBSD man pages say kilobytes, while a Darwin kernel is widely reported to
# hand back bytes. Guessing is a good way to be wrong by a factor of 1024, so the next section
# takes nobody's word for it — mine included. It calibrates this divisor against an allocation
# of known size, and says out loud whether the calibration held.
RU_MAXRSS_DIVISOR = MIB if sys.platform == "darwin" else 1024


def rss_hwm_mib() -> float:
    """Peak resident set size of THIS process so far, in MiB.

    This is a high-water mark: it only ever goes up. Freeing memory does not lower it.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RU_MAXRSS_DIVISOR


def touch_mmap(mib: int) -> int:
    """Map `mib` MiB of anonymous memory and write to every page, so it is really resident.

    The pages come from the operating system, never from Python's allocator, which is the
    whole point: `tracemalloc` cannot see them and the OS charges you for all of them.
    """
    page = b"\xff" * MIB
    buffer = mmap.mmap(-1, mib * MIB)
    try:
        for _ in range(mib):
            buffer.write(page)
        return len(buffer)
    finally:
        buffer.close()


_FAILED_CHECKS: list[str] = []


def _try(label: str, check: Callable[[], None]) -> None:
    """Run a check, or a demo that depends on your code, without derailing the notebook.

    A stub you have not filled in yet simply says so. A wrong answer prints the check's own
    message — which names the likely mistake — and the notebook carries on to the next cell,
    so one broken exercise never hides the feedback on the other three.
    """
    try:
        check()
    except NotImplementedError:
        print(f"{label}: not implemented yet — fill in the stub above, then re-run this cell.")
    except AssertionError as exc:
        _FAILED_CHECKS.append(label)
        print(f"{label}: FAILED — {exc}")
    except Exception as exc:  # a half-finished implementation raising something else
        _FAILED_CHECKS.append(label)
        print(f"{label}: raised {type(exc).__name__}: {exc}")


print(f"this process has already peaked at {rss_hwm_mib():.1f} MiB just by starting up")

# %% [markdown]
# ## 1. The phenomenon: two honest tools that disagree
#
# Python ships two ways to ask "how much memory did that cost?", and they answer different
# questions.
#
# - `tracemalloc` traces memory blocks **allocated by Python**. Precise, per-call, and blind
#   to anything that never passes through Python's allocators.
# - `resource.getrusage(...).ru_maxrss` is the **operating system's** high-water mark for the
#   whole process. It sees everything, forgets nothing, and cannot be scoped to one call.
#
# Run this. The workload allocates a known amount of memory outside Python's allocator.

# %%
tracemalloc.start()
tracemalloc.reset_peak()
_rss_before = rss_hwm_mib()
_mapped = touch_mmap(256)
_py_peak = tracemalloc.get_traced_memory()[1] / MIB
_rss_after = rss_hwm_mib()
tracemalloc.stop()

print(f"asked the OS for        {_mapped / MIB:.1f} MiB, and wrote to every page")
print(f"tracemalloc saw         {_py_peak:.2f} MiB")
print(f"the OS high-water mark  rose by {_rss_after - _rss_before:.1f} MiB")

_calibration_error = abs((_rss_after - _rss_before) - _mapped / MIB)
print(f"\ncalibration: divisor {RU_MAXRSS_DIVISOR} reproduces a known {_mapped / MIB:.0f} MiB "
      f"allocation to within {_calibration_error:.1f} MiB")
print("  → the unit is right on this machine" if _calibration_error < 32 else
      "  → the unit is WRONG on this machine: every MiB figure below is off by a constant\n"
      "    factor, so fix RU_MAXRSS_DIVISOR before believing anything this notebook prints")

# %% [markdown]
# Neither tool is lying. `tracemalloc` reported honestly on the blocks Python allocated —
# there were almost none. The number that decides whether a lesson ships is the second one,
# because it is the one that makes a laptop swap and a free notebook tier kill the kernel.
#
# A profiler worth having reports both, plus the clock. That is exercise 1.

# %% [markdown]
# ## 2. Exercise 1 — `measure()`
#
# Fill in the function. `Measurement` is given to you; you supply how each field is obtained.
#
# Three traps are deliberately in your way:
#
# - `time.time()` can go **backwards** when the system clock is adjusted. Use
#   `time.perf_counter()`, which is monotonic and includes time spent asleep.
# - `tracemalloc`'s peak is cumulative while tracing runs, so a second call inherits the
#   first call's peak until something resets it.
# - `ru_maxrss` never comes down, so the *rise* during your call is only meaningful when your
#   call set a new record for the whole process.

# %%
class Measurement(NamedTuple):
    """What one run of one callable cost."""

    result: Any           # whatever fn returned, so measuring is never in the way
    wall_s: float         # elapsed seconds on a monotonic clock
    py_peak_mib: float    # peak memory Python's own allocator held during the call
    rss_hwm_mib: float    # the whole process's high-water RSS after the call, absolute
    rss_delta_mib: float  # how much that high-water mark rose during the call, never negative


def measure(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Measurement:
    """Call `fn(*args, **kwargs)` and report what it cost.

    `result` must be exactly what `fn` returned, so a measured call can be dropped in
    anywhere the plain call was.

    Requirements, each of which is graded:
      * `wall_s` comes from `time.perf_counter()`, not `time.time()`.
      * `py_peak_mib` is this call's own peak in MiB. Whichever route you take — resetting the
        peak once tracing is on, stopping tracing when the call returns, or both — a later
        small call must never report a big earlier call's number.
      * `rss_hwm_mib` is the process high-water mark after the call, in MiB.
      * `rss_delta_mib` is `rss_hwm_mib` minus the same reading taken before the call,
        clamped at 0.0.

    Example:
        >>> m = measure(sum, [1, 2, 3])
        >>> m.result
        6
        >>> m.wall_s < 1.0 and m.rss_hwm_mib > 0.0
        True
    """
    # YOUR CODE HERE
    raise NotImplementedError


# Public checks — run these as often as you like.
def _check_measure() -> None:
    passthrough = measure(sum, [1, 2, 3])
    assert isinstance(passthrough, Measurement), (
        "measure must return a Measurement, not a tuple, dict or bare number — "
        "build it with Measurement(result=..., wall_s=..., ...)"
    )
    assert passthrough.result == 6, (
        "measure lost the return value: call fn(*args, **kwargs), keep what it returns, "
        "and put it in Measurement.result"
    )

    slow = measure(time.sleep, 0.25)
    assert 0.2 <= slow.wall_s <= 1.5, (
        f"measure(time.sleep, 0.25) reported {slow.wall_s:.3f}s — near zero means you are "
        "timing the wrong thing; ~250 means you are reporting milliseconds, not seconds"
    )

    big = measure(bytearray, 48 * MIB)
    assert 40.0 <= big.py_peak_mib <= 80.0, (
        f"a 48 MiB bytearray showed as {big.py_peak_mib:.1f} MiB — are you dividing bytes by "
        "1024*1024, and reading the PEAK (second element) of get_traced_memory()?"
    )

    small = measure(bytearray, 1024)
    assert small.py_peak_mib < 5.0, (
        f"a 1 KiB allocation reported {small.py_peak_mib:.1f} MiB, which is the previous "
        "call's peak — reset it with tracemalloc.reset_peak() once tracing is on, and call "
        "tracemalloc.stop() when the call returns"
    )

    assert big.rss_hwm_mib > 5.0, (
        "rss_hwm_mib is the whole process's peak, so it is tens of MiB even for a tiny call; "
        "a near-zero value means you subtracted a baseline that belongs in rss_delta_mib"
    )
    assert small.rss_delta_mib >= 0.0, "rss_delta_mib must be clamped at 0.0, never negative"
    print("exercise 1 looks right — measure() reports the clock, Python's peak and the OS peak")


# %%
_try("exercise 1", _check_measure)

# %% [markdown]
# ### What `measure()` can and cannot tell you
#
# Run the two cells below once `measure()` works. The first is the high-water trap: the *same*
# workload, measured twice, and neither answer is the amount of memory it used. The second
# re-proves the blind spot through your own profiler. A `rss_delta_mib` of 0.0 never means
# "this was free"; it means "this set no new record".

# %%
def _show_high_water_trap() -> None:
    asked = 512
    first = measure(touch_mmap, asked)
    second = measure(touch_mmap, asked)
    print(f"first  touch_mmap({asked}): rose {first.rss_delta_mib:6.1f} MiB "
          f"(process peak now {first.rss_hwm_mib:.0f} MiB)")
    print(f"second touch_mmap({asked}): rose {second.rss_delta_mib:6.1f} MiB "
          f"(process peak now {second.rss_hwm_mib:.0f} MiB)")
    print(f"\nsame work, two answers, and neither of them is {asked} MiB:")
    print("  the first understates it, because the mark did not start from the floor;")
    print("  the second reports nothing at all, because it set no new record.")
    print("a high-water mark measures records, not usage — which is why the repository's")
    print("gate runs every lesson in a FRESH process, where the record starts at the floor.")


_try("high-water demo", _show_high_water_trap)

# %%
def _show_blind_spot() -> None:
    m = measure(touch_mmap, 256)
    print(f"asked the OS for      {m.result / MIB:.0f} MiB, and wrote to every page")
    print(f"tracemalloc saw       {m.py_peak_mib:.2f} MiB of it")
    print(f"process high-water is {m.rss_hwm_mib:.0f} MiB")
    print(f"the call took         {m.wall_s * 1000:.0f} ms")
    print("\nwhat tracemalloc did see is the reusable page buffer inside touch_mmap, which is")
    print("a genuine Python bytes object. The mapping itself never reached Python's allocator,")
    print("so tracemalloc cannot report it — and the gate charges you for it regardless.")


_try("blind-spot demo", _show_blind_spot)

# %% [markdown]
# ## 3. The floor you never asked for
#
# A fresh interpreter costs memory before your code runs at all, and importing a library costs
# more. On an 8 GiB tier that floor is not yours to spend. Measure it the way this
# repository's own gate does: in a child process, through `RUSAGE_CHILDREN`.
#
# The snippets run in ascending order of cost on purpose. `RUSAGE_CHILDREN` is also a
# high-water mark, over *all* finished children, so a cheap child measured after an expensive
# one reports the expensive one's number. Same trap, one level up.

# %%
def measure_subprocess(snippet: str) -> tuple[float, float]:
    """Run `snippet` in a fresh interpreter; return (wall seconds, peak child RSS in MiB)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if proc.returncode != 0:
        raise RuntimeError(f"child failed: {proc.stderr.strip().splitlines()[-1:]}")
    return wall, max(after - before, after) / RU_MAXRSS_DIVISOR


for _label, _snippet in (("a bare interpreter", "pass"),
                         ("+ import numpy", "import numpy"),
                         ("+ import numpy, json, urllib.request",
                          "import numpy, json, urllib.request")):
    _wall, _peak = measure_subprocess(_snippet)
    print(f"{_label:38s} {_peak:7.1f} MiB   {_wall * 1000:5.0f} ms to start")

# %% [markdown]
# ## 4. Exercises 2 and 3 — profile one task, four ways
#
# The task: the exact sum of `i * i` for `i` in `range(n)`. Three implementations are given.
# They differ in memory and in speed — and one of them is **silently wrong** at the size we
# are about to run, which no profiler would ever have told you.
#
# You will write the fourth.

# %%
N_DEMO = 4_000_000


def exact_sum_squares(n: int) -> int:
    """Closed form, for checking the others: the sum of i*i for i in range(n)."""
    return (n - 1) * n * (2 * n - 1) // 6


def sum_squares_list(n: int) -> int:
    """Materialise every square in a list, then add them up."""
    return sum([i * i for i in range(n)])


def sum_squares_generator(n: int) -> int:
    """Same arithmetic, one square alive at a time."""
    return sum(i * i for i in range(n))


def sum_squares_numpy_whole(n: int) -> int:
    """One big int64 array, squared and summed by numpy."""
    values = np.arange(n, dtype=np.int64)
    return int((values * values).sum())


_truth = exact_sum_squares(N_DEMO)
print(f"int64 holds up to   {np.iinfo(np.int64).max}")
print(f"the exact answer is {_truth}\n")
print(f"{'implementation':22s} {'seconds':>8s} {'returns':>21s} {'exact?':>7s}")
for _name, _impl in (("list of squares", sum_squares_list),
                     ("generator", sum_squares_generator),
                     ("numpy, whole array", sum_squares_numpy_whole)):
    _t0 = time.perf_counter()
    _got = _impl(N_DEMO)
    _dt = time.perf_counter() - _t0
    print(f"{_name:22s} {_dt:8.3f} {_got:>21} {str(_got == _truth):>7s}")

# %% [markdown]
# Read the table you just produced: the quickest of the three is the only one that is wrong,
# and nothing about the way it failed looks like failure. The running total overflowed int64
# and wrapped around, where Python's own `int` has no ceiling to wrap at. No profiler would
# have caught it, because a profiler answers "what did that cost", never "was that right".
#
# So the fourth implementation has to get numpy's speed and bounded memory *and* Python's
# exactness: work in slices, and accumulate the running total in a Python `int`.
#
# Two stubs: the slice boundaries first, then the sum that uses them.

# %%
def chunk_bounds(n: int, chunk: int) -> list[tuple[int, int]]:
    """Split range(n) into consecutive half-open [start, stop) slices of at most `chunk`.

    The slices must cover range(n) exactly once, in order, with no gaps and no overlap.
    Raise ValueError if `chunk` is not positive.

    Example:
        >>> chunk_bounds(10, 4)
        [(0, 4), (4, 8), (8, 10)]
        >>> chunk_bounds(0, 4)
        []
    """
    # YOUR CODE HERE
    raise NotImplementedError


def sum_squares_chunked(n: int, chunk: int = 500_000) -> int:
    """Exact sum of i*i for i in range(n), never holding more than `chunk` elements.

    Build each slice with `np.arange(start, stop, dtype=np.int64)`, square it, sum it, and add
    that slice's total into a running Python `int`. Accumulating in a Python int is what keeps
    the answer exact once the total passes int64's ceiling.

    Example:
        >>> sum_squares_chunked(7, chunk=3)
        91
        >>> sum_squares_chunked(0)
        0
    """
    # YOUR CODE HERE
    raise NotImplementedError


def profile(impls: Mapping[str, Callable[[int], int]], n: int) -> dict[str, Measurement]:
    """Measure every implementation in `impls` on the same input `n`.

    Return a dict with the same keys, each mapped to the `Measurement` from calling that
    implementation with `n`. Do not sort, do not print, do not drop the results.

    Example:
        >>> table = profile({"exact": exact_sum_squares}, 7)
        >>> table["exact"].result
        91
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_chunking() -> None:
    assert chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)], (
        "chunk_bounds(10, 4) must be [(0, 4), (4, 8), (8, 10)] — the last slice is short, "
        "and stop is exclusive"
    )
    assert chunk_bounds(0, 4) == [], "chunk_bounds(0, 4) is an empty list, not [(0, 0)]"
    assert chunk_bounds(8, 4) == [(0, 4), (4, 8)], (
        "when chunk divides n exactly there is no extra empty slice at the end"
    )
    try:
        chunk_bounds(10, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("chunk_bounds(10, 0) must raise ValueError, not loop forever")

    assert sum_squares_chunked(7, chunk=3) == 91, (
        "sum_squares_chunked(7, chunk=3) should be 91 — do your slices cover range(n) "
        "exactly once?"
    )
    assert sum_squares_chunked(0) == 0, "an empty range sums to 0, not an error"
    big = sum_squares_chunked(N_DEMO)
    assert big == exact_sum_squares(N_DEMO), (
        f"off by {big - exact_sum_squares(N_DEMO)} at n={N_DEMO}: each slice's sum is fine, "
        "but the running total must be a Python int — add int(slice_total) to an int, do not "
        "accumulate into a numpy int64"
    )
    print("exercise 2 looks right — the chunked sum is exact at a size where numpy overflows")


def _check_profile() -> None:
    table = profile({"exact": exact_sum_squares, "generator": sum_squares_generator}, 7)
    assert set(table) == {"exact", "generator"}, (
        "profile must return one entry per implementation, keyed by the same names"
    )
    assert all(isinstance(m, Measurement) for m in table.values()), (
        "each value must be the Measurement returned by measure(), not just a duration"
    )
    assert table["exact"].result == 91, (
        "profile must call each implementation with n and keep its result — "
        "measure(fn, n) already does that for you"
    )
    print("exercise 3 looks right — profile() returns a Measurement per implementation")


# %%
_try("exercise 2", _check_chunking)
_try("exercise 3", _check_profile)

# %% [markdown]
# Now the trade-off, measured rather than asserted. Every number below is produced by your
# `profile()`; none of it was typed into this notebook.

# %%
def _show_tradeoff() -> None:
    impls = {
        "list of squares": sum_squares_list,
        "generator": sum_squares_generator,
        "numpy, whole array": sum_squares_numpy_whole,
        "numpy, chunked": sum_squares_chunked,
    }
    table = profile(impls, N_DEMO)
    truth = exact_sum_squares(N_DEMO)
    print(f"{'implementation':22s} {'seconds':>8s} {'py peak MiB':>12s} {'exact?':>7s}")
    for name, m in table.items():
        print(f"{name:22s} {m.wall_s:8.3f} {m.py_peak_mib:12.1f} {str(m.result == truth):>7s}")
    fastest = min((m.wall_s, name) for name, m in table.items() if m.result == truth)
    leanest = min((m.py_peak_mib, name) for name, m in table.items() if m.result == truth)
    print(f"\nfastest correct: {fastest[1]} at {fastest[0]:.3f}s")
    print(f"leanest correct: {leanest[1]} at {leanest[0]:.1f} MiB")
    whole, chunked = table["numpy, whole array"], table["numpy, chunked"]
    print(f"whole array vs chunked: {whole.py_peak_mib / max(chunked.py_peak_mib, 1e-9):.1f}x "
          f"the memory, {whole.wall_s / max(chunked.wall_s, 1e-9):.1f}x the time, and "
          f"exact={whole.result == truth} against exact={chunked.result == truth}")


_try("trade-off table", _show_tradeoff)

# %% [markdown]
# ### The profiler is not free
#
# One row of that table is badly distorted, and `measure()` is what distorted it. Tracing
# every allocation costs time, so the implementation that allocates four million objects pays
# a tax the one that allocates almost none never sees. Measure the same function twice — once
# under tracing, once without it — and see how large the tax is on this machine.

# %%
def _show_observer_effect() -> None:
    t0 = time.perf_counter()
    sum_squares_list(N_DEMO)
    untraced = max(time.perf_counter() - t0, 1e-9)
    traced = measure(sum_squares_list, N_DEMO).wall_s
    print(f"sum_squares_list, tracemalloc off: {untraced:.3f}s")
    print(f"sum_squares_list, tracemalloc on:  {traced:.3f}s")
    print(f"tracing cost a factor of {traced / untraced:.1f} on this workload")
    print("\nthe ranking in the table above survives this; the absolute seconds do not.")
    print("tools/execute.py times a lesson with tracing OFF, in a fresh process, which is why")
    print("its wall-clock number — not this one — is what the budget is written against.")


_try("observer effect", _show_observer_effect)

# %% [markdown]
# ## 5. Exercise 4 — `tier_check()`, the gate itself
#
# This repository sorts every lesson into a compute tier and refuses to ship one that does not
# fit. `cpu8` — the tier this lesson runs in — means 8 GiB and ten minutes. `phone` and
# `browser` mean 2 GiB. The GPU tiers mean what their names say.
#
# Reproduce the gate. The rules, in order:
#
# 1. An unknown tier fails immediately with exactly one reason: there is no ceiling to compare
#    against, so nothing else is checked.
# 2. A budget that is not positive fails. An undeclared budget is not an infinite one.
# 3. Wall time above the declared budget fails.
# 4. Peak memory above the tier's ceiling fails.
# 5. Being exactly *at* a limit passes. The comparison is `>`, not `>=`.

# %%
TIER_MIB = {
    "phone": 2048,
    "browser": 2048,
    "cpu8": 8192,
    "free-gpu": 16384,
    "gpu24": 24576,
    "gpu80": 81920,
    "multi-gpu": 163840,
    "api": 8192,
}


class Verdict(NamedTuple):
    """The gate's answer: did it pass, and if not, every reason why not."""

    passed: bool
    reasons: tuple[str, ...]


def tier_check(measured: Mapping[str, float], tier: str, budget_seconds: float) -> Verdict:
    """Decide whether a measured run fits its declared tier and budget.

    `measured` is a mapping with the keys "wall_s" and "peak_mib". Return a `Verdict` whose
    `reasons` is an empty tuple when it passes, and otherwise holds one human-readable string
    per broken rule, in the rule order given above. A memory reason must contain the tier's
    ceiling and a time reason the budget, so the author can see how far over they are. Format
    those numbers so they survive a small budget: `f"{budget_seconds:.1f}"` turns 0.05 into
    "0.1", which is how a gate ends up reporting "0.2s exceeds 0.1s". `:.3g` keeps both
    legible at either scale; bare `:g` spends six significant figures getting there.

    Example:
        >>> tier_check({"wall_s": 12.0, "peak_mib": 900.0}, "cpu8", 600).passed
        True
        >>> tier_check({"wall_s": 12.0, "peak_mib": 9000.0}, "cpu8", 600).passed
        False
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_tier_check() -> None:
    ok = tier_check({"wall_s": 12.0, "peak_mib": 900.0}, "cpu8", 600)
    assert ok.passed and ok.reasons == (), (
        "a run well inside its tier must pass with an EMPTY reasons tuple"
    )
    edge = tier_check({"wall_s": 600.0, "peak_mib": 8192.0}, "cpu8", 600)
    assert edge.passed, (
        "exactly at the limit passes: compare with > (over), not >= (at or over)"
    )
    unknown = tier_check({"wall_s": 1.0, "peak_mib": 1.0}, "cpu9000", 600)
    assert not unknown.passed and len(unknown.reasons) == 1, (
        "an unknown tier fails with exactly one reason and stops there — there is no ceiling "
        "to compare memory against"
    )
    both = tier_check({"wall_s": 900.0, "peak_mib": 9000.0}, "cpu8", 600)
    assert not both.passed and len(both.reasons) == 2, (
        f"over on both axes should give 2 reasons, got {len(both.reasons)}: append one reason "
        "per broken rule instead of returning at the first one"
    )
    assert "8192" in both.reasons[1], (
        "the memory reason must name the ceiling it broke, so the author can see the gap"
    )
    print("exercise 4 looks right — tier_check reproduces the gate, boundary included")


# %%
_try("exercise 4", _check_tier_check)

# %% [markdown]
# ### Run the gate on this very notebook
#
# Nothing below is hypothetical. It measures this process and asks your own gate whether this
# lesson may ship.

# %%
def _gate_this_lesson() -> None:
    elapsed = time.perf_counter() - _LESSON_T0
    peak = rss_hwm_mib()
    verdict = tier_check({"wall_s": elapsed, "peak_mib": peak}, "cpu8", 90)
    print(f"this notebook so far: {elapsed:.1f}s and {peak:.0f} MiB peak")
    print(f"tier_check says: passed={verdict.passed} reasons={verdict.reasons}")


_try("gate on this lesson", _gate_this_lesson)

# %% [markdown]
# ## 6. Why the budget does not move
#
# When a lesson comes in over its tier there are two things an author can do, and only one of
# them is allowed here.
#
# Raising the budget keeps the author comfortable and moves the cost onto the student. It is
# invisible in review — the gate goes green — and it is paid by exactly the people who cannot
# argue back: the ones on a four-year-old laptop, a shared machine, a free notebook tier that
# kills the kernel at its own ceiling. A lesson that needs 12 GiB does not teach 8 GiB
# students slowly. It does not teach them at all.
#
# Rewriting is the other option, and it nearly always produces a better lesson: fewer steps, a
# smaller model, a shorter horizon, a slice instead of the whole array. The constraint is the
# pedagogy, not an obstacle to it. You have already written the proof — `sum_squares_chunked`
# holds a fraction of the memory the whole-array version needs, gives up nothing in speed to
# do it (your own table printed the ratio), and is exact at a size where the version it
# replaced silently wraps. Shrinking the problem cost nothing. It found a bug.
#
# The budget is a promise to a student you will never meet. Measure, then keep it.

# %% [markdown]
# Run it rather than take my word for it. Below, one task is judged twice against the `phone`
# tier — as first written, then rewritten — and the rewrite passes a gate the original fails
# while returning an identical answer.

# %%
def _show_why_the_budget_does_not_move() -> None:
    n = N_DEMO
    budget = 0.05
    print(f"n={n}, judged against the phone tier ({TIER_MIB['phone']} MiB) "
          f"and a {budget}s budget.")
    print("timed with tracing OFF, the way tools/execute.py times a lesson; the memory figure")
    print("comes from a separate traced run, because there is no per-call RSS number to take —")
    print("a high-water mark belongs to the process, not to the call.\n")
    answers = []
    for name, fn in (("as written (list)", sum_squares_list),
                     ("rewritten (chunked)", sum_squares_chunked)):
        t0 = time.perf_counter()
        answers.append(fn(n))
        untraced = time.perf_counter() - t0
        py_peak = measure(fn, n).py_peak_mib
        verdict = tier_check({"wall_s": untraced, "peak_mib": py_peak}, "phone", budget)
        print(f"  {name:22s} {untraced:6.3f}s {py_peak:7.1f} MiB  passed={verdict.passed}")
        for reason in verdict.reasons:
            print(f"  {'':22s}   {reason}")
    print(f"\nboth return the same answer: {answers[0] == answers[1]}")
    print("the rewrite passes a gate the original fails, and costs the student nothing.")
    print("a bigger budget would have hidden that difference instead of finding it.")


_try("budget versus rewrite", _show_why_the_budget_does_not_move)

# %% [markdown]
# ## 7. Common mistakes
#
# - **Timing with `time.time()`.** It follows the system clock and can jump backwards
#   mid-run, producing negative durations. `time.perf_counter()` is monotonic.
# - **Trusting a zero `rss_delta_mib`.** A high-water mark only moves on a new record. The
#   second run of an identical workload usually reports a rise of 0.0.
# - **Believing `tracemalloc` is the whole story.** It reports what Python allocated. Memory
#   mapped straight from the OS is invisible to it, and the gate counts it anyway.
# - **Reporting `get_traced_memory()[0]`.** That is the *current* size; the peak is `[1]`.
# - **Measuring once.** The first run pays import and page-fault costs the second never sees.
# - **Letting numpy pick the accumulator.** `int64` wraps silently. Sum slices into a Python
#   `int` whenever the total can outgrow the dtype.
# - **Profiling in the same process as everything else.** Measure in a fresh interpreter, the
#   way `tools/execute.py` does, or the previous cell's peak becomes your result.
# - **Quoting a traced timing in a budget.** `tracemalloc` taxes allocation-heavy code far
#   more than it taxes numpy, so timings taken under tracing are not comparable to timings
#   taken without it. Time with tracing off; trace memory in a separate run.
#
# The fourth one is worth seeing rather than believing. Run the cell below: it allocates a
# 32 MiB buffer and throws it away inside a single call, which is what almost every real
# function does with its working memory.

# %%
def _show_current_versus_peak() -> None:
    tracemalloc.start()
    tracemalloc.reset_peak()
    blob = bytearray(32 * MIB)
    alive = tracemalloc.get_traced_memory()
    del blob
    freed = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"while the buffer is alive: current={alive[0] / MIB:6.1f} MiB  peak={alive[1] / MIB:6.1f} MiB")
    print(f"after it is thrown away:   current={freed[0] / MIB:6.1f} MiB  peak={freed[1] / MIB:6.1f} MiB")
    print("\n[0] is `current` and forgets; [1] is `peak` and remembers. A profiler that reads")
    print("[0] after the call has returned will report that this function cost nothing at all.")


_show_current_versus_peak()

# %% [markdown]
# ## 8. Self-check
#
# 1. Your `measure()` reports `rss_delta_mib = 0.0` for a function that builds a 300 MiB
#    array. The most likely explanation is:
#    - (a) the array was freed before the measurement ended, so it did not count
#    - (b) something earlier in this process already peaked higher, so no new record was set
#    - (c) `ru_maxrss` does not count numpy arrays
#
# 2. `tracemalloc` reports 0.00 MiB for `touch_mmap(256)` while the OS charges 256 MiB. This
#    means:
#    - (a) `tracemalloc` is broken on this platform
#    - (b) the pages were never resident, so nothing was really used
#    - (c) the memory never passed through Python's allocator, which is all `tracemalloc` sees
#
# 3. `sum_squares_numpy_whole` beat both pure-Python versions by more than an order of
#    magnitude and still returned the wrong answer at `n = 4_000_000`. No profiler caught
#    that, because:
#    - (a) profilers measure cost, not correctness; only a reference answer catches this
#    - (b) the error is too small to show up in a timing measurement
#    - (c) overflow affects only memory, never results
#
# 4. A lesson measures 11 minutes and 9 GiB on the `cpu8` tier. The correct response is:
#    - (a) raise `budget_seconds` and declare a larger tier
#    - (b) shrink the problem — fewer steps, smaller model, shorter horizon — and re-measure
#    - (c) keep the declared tier and note in the README that slower machines may struggle
#
# 5. In the trade-off table, `list of squares` looks far slower than it is when you call it
#    on its own. The reason is:
#    - (a) the list implementation really is that slow; the table is accurate
#    - (b) `measure()` traces every allocation, and that implementation makes millions of them
#    - (c) numpy released the GIL, so the other implementations were given more CPU
#
# Answers are published in the course solution bundle.

# %% [markdown]
# ## What you built, and where it goes next
#
# `measure()` and `tier_check()` are the instruments every other lesson in this Atlas is held
# to: each one declares a tier, and the executor writes back the numbers it actually observed
# instead of the ones its author hoped for. You now own the gate that judges them.

# %%
if __name__ == "__main__":
    for _name, _check in (("exercise 1", _check_measure),
                          ("exercise 2", _check_chunking),
                          ("exercise 3", _check_profile),
                          ("exercise 4", _check_tier_check)):
        _try(_name, _check)
    # A stub you have not reached yet is not a failure. A check that ran and came back wrong
    # is, and it ends this run non-zero rather than letting a green exit code paper over it.
    if _FAILED_CHECKS:
        raise SystemExit("checks failed: " + ", ".join(dict.fromkeys(_FAILED_CHECKS)))
