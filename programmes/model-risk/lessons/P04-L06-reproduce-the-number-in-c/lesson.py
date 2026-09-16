# %% [markdown]
# # P04-L06 · Reproducing the first line's number, exactly
#
# **You will build:** three summation routines in C — left-to-right, pairwise and
# compensated — plus the two instruments a validator needs to talk about a disagreement
# (a distance in representable doubles, and a count of agreeing decimal digits), plus the
# decision function that says whether a gap between two teams' figures is arithmetic or a
# finding.
#
# **Time:** ~75 minutes · **Runs on:** a laptop CPU, no GPU, no download, no network
# · **Prerequisites:** P04-L01 (the validation suite), T00-L01 (the tier gate)
#
# By the end you will be able to:
# 1. Implement left-to-right, pairwise and Kahan-Babuska-Neumaier summation in C and
#    reproduce a Python reference bit for bit.
# 2. Measure how far each one lands from the exactly-rounded total, in representable doubles
#    rather than in adjectives.
# 3. Demonstrate that the same values summed in a different order give a different answer,
#    and that compensated summation does not move.
# 4. Implement a distance in ULPs and a count of agreeing significant digits, and use them to
#    report a disagreement precisely.
# 5. Implement `reproduce()` — the worst-case bound and the three verdicts — and explain why
#    the only verdict that closes a finding is bit-for-bit equality.
#
# Most of what you write lives in **`lesson.c`**. This notebook builds it, drives it, measures
# it and grades it. Two exercises are in Python, because the instruments a validator uses to
# describe a gap belong next to the report, not next to the loop.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import hashlib
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

print("numpy", np.__version__, "· python", sys.version.split()[0])

# True in a notebook and when this file is run as a script; False when the autograder imports
# it. Every check below is called under this guard, so the cell you are sitting in reports on
# itself, while importing the lesson never runs anything.
_IS_MAIN = __name__ == "__main__"

try:
    LESSON_DIR = Path(__file__).resolve().parent
except NameError:  # a notebook has no __file__
    LESSON_DIR = Path.cwd()

C_SRC = "lesson.c"
BIN = "lesson_bin"

# Mirrored exactly from lesson.c. Both languages must build the same array out of the same
# bits or nothing in this lesson means anything.
SEED = 20260916
GAMMA = 0x9E3779B97F4A7C15
N = 2_000_000
PAIRWISE_BLOCK = 128
EXP_SPAN, EXP_SHIFT = 41, 20

DATA_NOTE = ("SYNTHETIC. Every exposure and every probability of default in this lesson is "
             "generated from a counter-based pseudo-random stream seeded with 20260916, in "
             "the notebook and in lesson.c. No real portfolio is used anywhere.")

_BUILD = None
_CACHE = {}


def build(verbose: bool = True):
    """Compile the C source with make. Cached: the compiler runs once per session."""
    global _BUILD
    if _BUILD is None:
        proc = subprocess.run(
            ["make", "-C", str(LESSON_DIR), f"PYTHON={sys.executable}",
             f"SRC={C_SRC}", f"BIN={BIN}"],
            capture_output=True, text=True, timeout=600)
        _BUILD = (proc.returncode == 0, (proc.stdout + proc.stderr).strip())
    ok, out = _BUILD
    if verbose:
        print("build OK" if ok else "BUILD FAILED\n" + out)
    return ok, out


def run_c(*args, timeout: int = 300) -> str:
    """Run the compiled binary and return its stdout.

    Exit code 2 means one of the four C exercises is still a stub, so it is re-raised as
    NotImplementedError — the grader then reports TODO instead of an error.
    """
    ok, out = build(verbose=False)
    if not ok:
        raise RuntimeError("the C build failed; run build() to see the compiler output\n" + out)
    proc = subprocess.run([str(LESSON_DIR / BIN), *args],
                          cwd=LESSON_DIR, capture_output=True, text=True, timeout=timeout)
    if proc.returncode == 2:
        raise NotImplementedError(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"{BIN} {' '.join(args)} exited {proc.returncode}\n"
                           f"{proc.stderr.strip()}")
    return proc.stdout


def cached(*args) -> str:
    """run_c, but each distinct command line is executed once per session."""
    if args not in _CACHE:
        _CACHE[args] = run_c(*args)
    return _CACHE[args]


def metrics(text: str) -> dict:
    """Parse the binary's `@ key=value` report lines into a dict of floats."""
    return {k: float(v) for k, v in
            (line[2:].split("=", 1) for line in text.splitlines() if line.startswith("@ "))}


def rows(text: str, tag: str) -> list:
    """Parse the binary's `<tag> v1 v2 ...` lines into a list of lists of floats."""
    return [[float(v) for v in line.split()[1:]]
            for line in text.splitlines() if line.startswith(tag + " ")]


def make_test():
    """Run the C self-test — what `make test` runs. Returns (exit code, combined output).

    The binary is built with make and then invoked directly, so its OWN exit code survives:
    make reports any failed recipe as 2, which would make a real failure (the binary's 1)
    indistinguishable from an unfinished stub (the binary's 2).
    """
    ok, out = build(verbose=False)
    if not ok:
        return 1, out
    proc = subprocess.run([str(LESSON_DIR / BIN), "selftest"],
                          cwd=LESSON_DIR, capture_output=True, text=True, timeout=600)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


_FAILURES = []


def _try(fn, *a, **k):
    """Run a public check, print what it says, and remember whether it passed."""
    try:
        fn(*a, **k)
    except NotImplementedError as e:
        print(f"  TODO {fn.__name__}: {e}")
        _FAILURES.append(fn.__name__)
    except AssertionError as e:
        print(f"  FAIL {fn.__name__}: {e}")
        _FAILURES.append(fn.__name__)
    else:
        print(f"  ok   {fn.__name__}")


if _IS_MAIN:
    build()

# %% [markdown]
# ## 1. The disagreement
#
# The first line hands you a portfolio total. You recompute it from the same extract and get
# a different number — not wildly different, different in the sixth decimal place. The first
# line says "floating point". You have to decide whether that sentence closes the item or
# opens it.
#
# The 2026 interagency guidance on model risk management, issued on 17 April 2026 and carried
# by the Federal Reserve as SR 26-2, says the agencies revised it to "clarify model risk
# management principles and to emphasize a risk-based approach" (`claims.yaml`). Nothing in it
# tells you how many digits to agree on. That is the job below.
#
# This module is in C because summation order is the whole subject, and a one-line
# `np.sum(x)` hides it. Start by asking this machine what its doubles are made of — nothing
# in this lesson is a number somebody typed.

# %%
if _IS_MAIN:
    _facts = metrics(cached("facts"))
    print(f"  a double carries {_facts['mant_dig']:.0f} bits of significand")
    print(f"  DBL_EPSILON     = {_facts['dbl_epsilon']!r}")
    print(f"  unit roundoff u = {_facts['unit_roundoff']!r}  (= DBL_EPSILON / 2 = 2^-53)")
    print(f"  python agrees:    {sys.float_info.epsilon!r}, "
          f"{sys.float_info.mant_dig} bits")
    print(f"  pairwise block  = {_facts['pairwise_block']:.0f}")
    _wider = _facts["ldbl_mant_dig"] > _facts["mant_dig"]
    print(f"  long double     = {_facts['sizeof_long_double']:.0f} bytes, "
          f"{_facts['ldbl_mant_dig']:.0f} bits of significand — "
          + ("wider than a double here" if _wider
             else "THE SAME as a double here, so accumulating in one buys nothing"))
    print(f"\n  {DATA_NOTE}")

# %% [markdown]
# ## 2. The portfolio, built out of exact bits
#
# Two teams cannot argue about a number unless they are summing the same array. So the array
# is not a file: it is a counter-based stream, and every element is built by operations that
# are exact in IEEE arithmetic — an integer of at most 53 bits converted to a double, then
# scaled by a power of two. C and numpy therefore produce the *same 64 bits* for every
# element, and any difference in the total is the summation, not the data.
#
# Two million signed positions, each scaled by a power of two from `2^-20` to `2^+20`. The
# mantissa is a uniform 53-bit fraction on top of that scale, so the smallest magnitudes
# fall well below `2^-20` — the cell below prints the range it actually generated, which
# is the only version of that sentence worth trusting. The spread is the point: a rounding
# error in the large positions is bigger than an entire small position.

# %%
def generate_portfolio(n: int = N):
    """The synthetic portfolio, generated in numpy exactly as lesson.c generates it in C.

    Returns (exposure, pd): signed position values, and probabilities of default in [0, 1).
    Given, not an exercise — but read it, because the exactness is deliberate. `ldexp` by a
    power of two and a 53-bit integer converted to a double are the only two operations here
    that touch floating point, and neither one rounds.
    """
    k = np.arange(3 * n, dtype=np.uint64)
    z = np.uint64(SEED) + (k + np.uint64(1)) * np.uint64(GAMMA)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    u0, u1, u2 = z[0::3].copy(), z[1::3].copy(), z[2::3].copy()
    mant = (u0 >> np.uint64(11)).astype(np.float64)
    e = (u1 % np.uint64(EXP_SPAN)).astype(np.int64) - EXP_SHIFT
    sign = np.where((u1 >> np.uint64(63)) != np.uint64(0), -1.0, 1.0)
    exposure = sign * np.ldexp(mant, (e - 53).astype(np.int32))
    pd = np.ldexp((u2 >> np.uint64(11)).astype(np.float64), np.int32(-53))
    return exposure, pd


def _check_bit_identity() -> None:
    """The C array and the numpy array must be the same bits, or nothing else is evidence."""
    ex, pd = generate_portfolio(N)
    head = rows(cached("dump", "--dump", "4000"), "x")
    assert head, "the binary printed no dump rows"
    bad = [(int(i), c_ex, float(ex[int(i)])) for i, c_ex, c_pd in head
           if c_ex != float(ex[int(i)]) or c_pd != float(pd[int(i)])]
    assert not bad, (
        f"{len(bad)} of the first {len(head)} records differ between C and numpy; the first "
        f"is index {bad[0][0]}, C said {bad[0][1]!r} and numpy said {bad[0][2]!r}. The "
        "generator must use the same seed, the same multipliers and the same exponent range "
        "in both languages.")
    print(f"    {len(head)} records checked, C and numpy identical to the bit")
    print(f"    magnitudes run from {float(np.abs(ex).min())!r} to "
          f"{float(np.abs(ex).max())!r}")


if _IS_MAIN:
    _try(_check_bit_identity)

# %% [markdown]
# ## 3. Exercise 1 — `naive_sum()` in `lesson.c`
#
# One accumulator, left to right. This is what a spreadsheet does, what a `for` loop does,
# and what the first line almost certainly ran. Open `lesson.c`, find `naive_sum`, and fill
# it in; the stub's comment carries the requirements and a worked example.
#
# Your feedback loop is `make test`, and the cell below runs it. The Python mirror is given
# so you can see the two languages agree: same order, same additions, same 64 bits out.

# %%
def python_naive_sum(xs) -> float:
    """Left-to-right summation in Python. GIVEN — the mirror of your C exercise 1."""
    s = 0.0
    for v in xs:
        s += v
    return s


def _check_naive_sum() -> None:
    ex, _pd = generate_portfolio(N)
    xs = ex.tolist()
    want = python_naive_sum(xs)
    got = metrics(cached("sums", "--n", str(N)))["naive"]
    assert got == want, (
        f"your C naive_sum returned {got!r} where the same left-to-right loop in Python "
        f"returns {want!r}. These must agree to the last bit: both are IEEE doubles added in "
        "index order. A difference means you reordered, used a partial sum, accumulated in a "
        "different type, or skipped an element.")
    exact = math.fsum(xs)
    print(f"    C and Python agree exactly: {got!r}")
    print(f"    the exactly-rounded total is {exact!r}")
    print(f"    naive summation is off by     {abs(got - exact):.3e}")


if _IS_MAIN:
    _code, _out = make_test()
    print(_out)
    print(f"exit code {_code}  (0 = all pass, 2 = something is still a stub, 1 = a failure)\n")
    _try(_check_naive_sum)

# %% [markdown]
# ## 4. Exercise 2 — `pairwise_sum()` in `lesson.c`
#
# Split, sum each half, add. Errors then accumulate down a tree of depth `log2(n)` rather
# than along a chain of length `n`. The convention is fixed in the stub — blocks of
# `PAIRWISE_BLOCK = 128`, split at `n / 2` — and it is fixed *because* reproducibility is the
# subject: two pairwise implementations that split differently do not reproduce each other.
#
# That is not hypothetical. numpy's `sum` documentation says it uses "partial pairwise
# summation" (`claims.yaml`), and the cell below shows numpy's answer and yours are both
# pairwise and still not the same number.

# %%
def _check_pairwise_sum() -> None:
    ex, _pd = generate_portfolio(N)
    xs = ex.tolist()
    exact = math.fsum(xs)
    m = metrics(cached("sums", "--n", str(N)))
    pw, nv = m["pairwise"], m["naive"]
    assert pw != nv, (
        f"pairwise_sum returned the same double as naive_sum ({pw!r}) over {N} values. With "
        f"{N} elements the recursion is about {math.log2(N / PAIRWISE_BLOCK):.0f} levels "
        "deep; if the two agree, the split is probably not happening at all.")
    assert abs(pw - exact) < abs(nv - exact), (
        f"pairwise_sum ({pw!r}) is further from the exactly-rounded total ({exact!r}) than "
        f"naive_sum ({nv!r}) is. A log-depth tree should beat a length-n chain; check that "
        "the recursion splits at half = n / 2 and that the base case sums left to right.")
    print(f"    yours   {pw!r}")
    print(f"    numpy   {float(np.sum(ex))!r}   <- also pairwise, also not the same number")
    print(f"    exact   {exact!r}")


if _IS_MAIN:
    _try(_check_pairwise_sum)

# %% [markdown]
# ## 5. Exercise 3 — `kahan_sum()` in `lesson.c`
#
# Compensated summation keeps a second accumulator for the part each addition threw away, and
# adds it back at the end. Implement the Kahan-Babuska-Neumaier form, which branches on which
# operand is larger; the stub explains why, and the self-test's
# `kahan_sum({1, 1e16, -1e16})` case is the one the original 1965 form
# (Kahan, *Communications of the ACM* 8(1):40 — `claims.yaml`) gets wrong.
#
# Python's `math.fsum` is the reference here: its documentation says it "avoids loss of
# precision by tracking multiple intermediate partial sums" (`claims.yaml`), which is a
# different and more expensive algorithm than yours. Landing within a step or two of it is
# the target — a step or two rather than none, because the same documentation records that an
# intermediate sum can be double-rounded and come out wrong in its last bit.
#
# This lesson measures the ranking rather than deriving it. The analysis is in Higham, *The
# accuracy of floating-point summation*, SIAM J. Sci. Comput. 14(4):783-799, 1993
# (`claims.yaml`).

# %%
def _key(x: float) -> int:
    """The double's position in the ordered list of all doubles. GIVEN; exercise 4 uses it."""
    (u,) = struct.unpack("<Q", struct.pack("<d", x))
    return -(u & 0x7FFFFFFFFFFFFFFF) if u >> 63 else u


def _check_kahan_sum() -> None:
    ex, _pd = generate_portfolio(N)
    xs = ex.tolist()
    exact = math.fsum(xs)
    m = metrics(cached("sums", "--n", str(N)))
    kh, nv, pw = m["kahan"], m["naive"], m["pairwise"]
    steps = abs(_key(kh) - _key(exact)) if kh != exact else 0
    assert steps <= 2, (
        f"kahan_sum returned {kh!r}, which is {steps} representable doubles away from the "
        f"exactly-rounded total {exact!r}. One or two steps is the algorithm; hundreds means "
        "the compensation is being overwritten rather than accumulated, or you returned s "
        "instead of s + c, or you used the unbranched 1965 form.")
    print(f"    exact          {exact!r}")
    print("    distance from exact, in representable doubles:")
    for name, val in (("naive", nv), ("pairwise", pw), ("kahan", kh)):
        d = 0 if val == exact else abs(_key(val) - _key(exact))
        print(f"      {name:<9s} {val!r}  {d:>6d} steps")
    gross = metrics(cached("sums", "--n", str(N)))
    print(f"\n    and on the GROSS total, where nothing cancels:")
    print(f"      naive     {gross['gross_naive']!r}")
    print(f"      kahan     {gross['gross_kahan']!r}")


if _IS_MAIN:
    _try(_check_kahan_sum)

# %% [markdown]
# ## 6. The same numbers, four orders
#
# Here is the thing a validator has to be able to say out loud. Addition of doubles is
# commutative but **not associative**: `(a + b) + c` and `a + (b + c)` are different
# computations, and on a portfolio spanning twenty binary orders of magnitude they are
# different numbers. Below, the identical two million values are summed as generated, backwards,
# smallest-magnitude-first and largest-magnitude-first.
#
# Watch which column moves.

# %%
def _check_order_dependence() -> None:
    ex, _pd = generate_portfolio(N)
    exact = math.fsum(ex.tolist())
    labels = {0: "as generated", 1: "reversed", 2: "ascending |x|", 3: "descending |x|"}
    block = rows(cached("order", "--n", str(N)), "order")
    assert len(block) == 4, f"the order command printed {len(block)} rows, expected 4"
    naive_vals = {int(c): nv for c, nv, _kh in block}
    kahan_vals = {int(c): kh for c, _nv, kh in block}
    assert len(set(naive_vals.values())) > 1, (
        "all four orders gave the SAME naive total. On two million signed values whose "
        "scales span 2^-20 to 2^20 that cannot happen if the sum is really left to right.")
    assert len(set(kahan_vals.values())) == 1, (
        f"compensated summation gave {len(set(kahan_vals.values()))} different answers across "
        "the four orders; it should give one. Check that the compensation accumulates.")
    print("    order            naive sum              steps off   compensated sum")
    for c in sorted(labels):
        nv, kh = naive_vals[c], kahan_vals[c]
        d = 0 if nv == exact else abs(_key(nv) - _key(exact))
        print(f"    {labels[c]:<15s}  {nv!r}  {d:>6d}     {kh!r}")
    print(f"\n    exactly rounded: {exact!r}")
    print("    the folklore is 'add the small ones first'. It is about all-positive sums;")
    print("    measure it on your own signed portfolio before quoting it in a report.")


if _IS_MAIN:
    _try(_check_order_dependence)

# %% [markdown]
# ## 7. Exercise 4 — `ulps_between(a, b)`, in Python
#
# "They differ in the last digit" is not a measurement. The measurement is how many
# representable doubles lie between the two figures, and it is what turns an argument into a
# number a report can carry.
#
# The recipe is in the stub. The trap is signed zero and the crossing of zero: `-0.0` and
# `+0.0` are equal and must be zero steps apart, while the two doubles either side of zero
# are one step apart even though their bit patterns differ in the top bit.

# %%
def ulps_between(a: float, b: float) -> int:
    """How many representable doubles lie between a and b, as a non-negative int.

    Build the ordering key for each number and subtract:

        u = the raw 64 bits of the double, as an unsigned integer
            (struct.unpack("<Q", struct.pack("<d", x))[0])
        key = -(u & 0x7FFFFFFFFFFFFFFF)  if the sign bit (u >> 63) is set,
              u                          otherwise

    That key increases monotonically with the value, so the distance is abs(key(a) - key(b)).
    Do the subtraction on the integers; Python integers are exact, but converting a key of
    about 4.6e18 to a float first would round it to a multiple of 1024 and a gap of a few
    hundred steps would vanish.

    Requirements, all graded:
      * return 0 when a == b, which covers -0.0 against +0.0;
      * raise ValueError if either argument is a NaN — a NaN has no position on the line;
      * the result is symmetric and never negative.

    Worked example: ulps_between(1.0, 1.0 + 2**-52) is 1, because 1.0 + 2**-52 is the very
    next double above 1.0. ulps_between(-0.0, 5e-324) is also 1: 5e-324 is the smallest
    positive subnormal, and the key of -0.0 is 0, not -(2**63).
    """
    # YOUR CODE HERE
    raise NotImplementedError("implement ulps_between")


def _check_ulps_between() -> None:
    assert ulps_between(1.0, 1.0) == 0, "equal numbers are zero steps apart"
    assert ulps_between(-0.0, 0.0) == 0, (
        f"ulps_between(-0.0, 0.0) returned {ulps_between(-0.0, 0.0)!r}; the two zeros compare "
        "equal, so they must be zero steps apart. Check a == b before touching the bits.")
    nxt = math.nextafter(1.0, 2.0)
    assert ulps_between(1.0, nxt) == 1, (
        f"1.0 and the next double above it are one step apart; you returned "
        f"{ulps_between(1.0, nxt)!r}")
    assert ulps_between(nxt, 1.0) == 1, "the distance must be symmetric"
    assert ulps_between(-0.0, 5e-324) == 1, (
        f"you returned {ulps_between(-0.0, 5e-324)!r} for the two doubles either side of "
        "zero. Subtracting raw bit patterns gives 2**63 here; the key must map the sign bit "
        "onto a negative number so the line joins up at zero.")
    try:
        ulps_between(math.nan, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a NaN argument must raise ValueError, not return a number")
    ex, _pd = generate_portfolio(N)
    exact = math.fsum(ex.tolist())
    m = metrics(cached("sums", "--n", str(N)))
    print(f"    naive is {ulps_between(m['naive'], exact)} doubles from the exact total")
    print(f"    kahan is {ulps_between(m['kahan'], exact)}")


if _IS_MAIN:
    _try(_check_ulps_between)

# %% [markdown]
# ## 8. Exercise 5 — `agreeing_significant_digits(a, b)`, in Python
#
# ULPs are the right unit for arguing with an engineer. Digits are the unit a committee
# reads. Both belong in the finding, and they say different things: two numbers can be
# hundreds of ULPs apart and still agree to thirteen printed digits.

# %%
def agreeing_significant_digits(a: float, b: float) -> int:
    """How many leading significant decimal digits a and b share, from 0 to 17.

    Seventeen significant digits determine a double uniquely, so seventeen is the maximum and
    it means "identical".

    The rules, in order:
      * raise ValueError if either argument is a NaN;
      * return 17 if a == b;
      * return 0 if either is an infinity (they were not equal, so they do not agree);
      * return 0 if the signs differ — use math.copysign(1.0, x), so -1.0 and 1.0 score 0;
      * write both with f"{x:.16e}", giving "d.dddddddddddddddde+NN"; if the exponent parts
        differ, return 0;
      * otherwise strip the sign and the decimal point from each mantissa, leaving 17 digits
        each, and count the leading digits that match, stopping at the first that does not.

    Worked example: agreeing_significant_digits(1.2345678901234567, 1.2345699999999999).
    The mantissa digit strings are "12345678901234567" and "12345699999999999"; they agree on
    1, 2, 3, 4, 5, 6 and differ at the seventh, so the answer is 6.

    The exponent test is not a formality. 1.0 and 10.0 have IDENTICAL mantissa digit strings;
    skip the test and a figure and the same figure off by a factor of ten come back agreeing
    on all seventeen.
    """
    # YOUR CODE HERE
    raise NotImplementedError("implement agreeing_significant_digits")


def _check_agreeing_significant_digits() -> None:
    f = agreeing_significant_digits
    assert f(1.0, 1.0) == 17, "identical numbers agree on every digit you could print"
    assert f(1.2345678901234567, 1.2345699999999999) == 6, (
        f"the worked example in the docstring must return 6; you returned "
        f"{f(1.2345678901234567, 1.2345699999999999)!r}")
    assert f(-1.0, 1.0) == 0, (
        f"opposite signs agree on nothing; you returned {f(-1.0, 1.0)!r}")
    assert f(9.999, 10.001) == 0, (
        f"different decimal exponents return 0; you returned {f(9.999, 10.001)!r}")
    assert f(1.0, 10.0) == 0, (
        f"1.0 and 10.0 agree on nothing; you returned {f(1.0, 10.0)!r} — their 17-digit "
        "mantissas are identical and only the exponent tells them apart, so compare the "
        "exponents before the digits")
    assert f(-0.0, 0.0) == 17, (
        f"the two zeros are equal, so they agree on everything; you returned {f(-0.0, 0.0)!r}"
        " — check a == b before you look at the sign")
    try:
        f(math.nan, math.nan)
    except ValueError:
        pass
    else:
        raise AssertionError("a NaN argument must raise ValueError")
    m = metrics(cached("sums", "--n", str(N)))
    block = rows(cached("order", "--n", str(N)), "order")
    fwd, rev = block[0][1], block[1][1]
    print(f"    two teams, same data, one summed it backwards:")
    print(f"      {fwd!r}")
    print(f"      {rev!r}")
    print(f"      {f(fwd, rev)} significant digits agree, "
          f"{ulps_between(fwd, rev)} representable doubles apart")
    print(f"    and against the exactly-rounded total, naive agrees to "
          f"{f(m['naive'], math.fsum(generate_portfolio(N)[0].tolist()))} digits")


if _IS_MAIN:
    _try(_check_agreeing_significant_digits)

# %% [markdown]
# ## 9. Exercise 6 — `reproduce()` in `lesson.c`
#
# Now the decision. Two figures, `n` values, and `sum_abs` — the sum of the absolute values,
# which is the only thing that bounds how big the intermediate partial sums could have got.
# The defensible bound for left-to-right summation is
#
# > `bound = (n - 1) * u * sum_abs`, with `u = 2^-53`
#
# — each of the `n - 1` additions can round by up to half an ulp of its own partial sum, and
# no partial sum exceeds `sum_abs`. Three verdicts: `REPRODUCED` (0) when the two are equal
# bit for bit, `EXPLAINED` (1) when the gap is inside the bound, `FINDING` (2) otherwise.
#
# Fill in `reproduce` in `lesson.c`, then run this.

# %%
def verdict_of(a: float, b: float, n: int, sum_abs: float) -> dict:
    """Ask the C binary for a verdict. GIVEN."""
    out = cached("reproduce", "--a", repr(a), "--b", repr(b), "--n", str(n),
                 "--sum-abs", repr(sum_abs))
    m = metrics(out)
    m["name"] = next(l.split()[1] for l in out.splitlines() if l.startswith("verdict_name "))
    return m


def _check_reproduce() -> None:
    same = verdict_of(1.25, 1.25, 1000, 1.0)
    assert same["verdict"] == 0, (
        f"identical figures are REPRODUCED (0); you returned {same['verdict']:.0f} "
        f"({same['name']})")
    near = verdict_of(1.0, math.nextafter(1.0, 2.0), 1000, 1.0)
    assert near["verdict"] == 1, (
        f"a one-ulp gap over 999 additions of values summing to 1.0 sits far inside the "
        f"bound {999 * 2.0 ** -53!r}; expected EXPLAINED (1), got {near['name']}")
    far = verdict_of(1.0, 1.5, 1000, 1.0)
    assert far["verdict"] == 2, (
        f"half a unit apart is not arithmetic; expected FINDING (2), got {far['name']}")
    assert near["bound"] == 999 * 2.0 ** -53, (
        f"the bound came back {near['bound']!r}; it must be (n - 1) * 2**-53 * sum_abs = "
        f"{999 * 2.0 ** -53!r}, computed in that order")
    signed = verdict_of(1.0, math.nextafter(1.0, 2.0), 1000, -1.0)
    assert signed["bound"] == near["bound"], (
        f"handed a sum_abs of -1.0 the bound came back {signed['bound']!r} instead of "
        f"{near['bound']!r}. sum_abs is a magnitude; a negative bound makes gap <= bound "
        "false for every gap, so every reconciliation would come back a finding.")

    ex, _pd = generate_portfolio(N)
    m = metrics(cached("sums", "--n", str(N)))
    block = rows(cached("order", "--n", str(N)), "order")
    fwd, rev = block[0][1], block[1][1]
    real = verdict_of(fwd, rev, N, m["gross_kahan"])
    print(f"    the two teams' totals, adjudicated:")
    print(f"      gap   {real['gap']!r}")
    print(f"      bound {real['bound']!r}")
    print(f"      -> {real['name']}")
    print(f"    the bound is {real['bound'] / real['gap']:,.0f} times the observed gap.")
    print("    A worst case that loose explains away almost anything, which is the point of")
    print("    section 10.")


if _IS_MAIN:
    _try(_check_reproduce)

# %% [markdown]
# ## 10. Why `EXPLAINED` is not a resolution
#
# The bound is honest and it is nearly useless. It has to assume every one of the two million
# additions rounded the wrong way by the maximum, against the largest partial sum the data
# permits. Real accumulation is a random walk, so the observed gap is smaller by several
# orders of magnitude — the ratio printed above.
#
# So a difference that `EXPLAINED` covers could equally be a dropped record, a sign flip, or a
# filter applied on one side and not the other. `EXPLAINED` means *this bound cannot reject
# the explanation*. It does not mean the explanation is true. The cell below plants a genuine
# error — one position dropped — and asks the bound to notice.

# %%
def _check_a_real_error_hides_inside_the_bound() -> None:
    ex, _pd = generate_portfolio(N)
    xs = ex.tolist()
    exact = math.fsum(xs)
    m = metrics(cached("sums", "--n", str(N)))
    # Drop one ordinary, unremarkable position — the one closest to a single currency unit.
    # A real defect: a filter applied on one side and not the other, a record that failed to
    # load. Not arithmetic.
    i = int(np.argmin(np.abs(np.abs(ex) - 1.0)))
    dropped = math.fsum(xs[:i] + xs[i + 1:])
    v = verdict_of(exact, dropped, N, m["gross_kahan"])
    assert v["verdict"] == 1, (
        f"dropping one position should still come back EXPLAINED here; got {v['name']}. If "
        "it did not, the bound is not being computed from n and sum_abs.")
    print(f"    a position worth {float(ex[i])!r} was dropped from the extract")
    print(f"    the total moved by {abs(exact - dropped)!r}, against a bound of "
          f"{v['bound']!r}")
    print(f"    verdict: {v['name']} — the worst-case bound cannot see it")
    print(f"    but bit-for-bit reproduction can: {ulps_between(exact, dropped)} doubles apart,")
    print(f"    agreeing to {agreeing_significant_digits(exact, dropped)} significant digits")
    print("    The only verdict that closes a finding is REPRODUCED. Agree the order, agree")
    print("    the algorithm, agree the build flags, and compare the bits.")


if _IS_MAIN:
    _try(_check_a_real_error_hides_inside_the_bound)

# %% [markdown]
# ## 11. The metric, not just the total
#
# Module 1 built a support-weighted mean: a numerator, a denominator, a division. Every
# weighted metric a validation suite reports has that shape, and every one of them inherits
# whatever the two sums did. Here is the portfolio's exposure-weighted mean probability of
# default under each of your three summations.

# %%
def _check_weighted_mean() -> None:
    ex, pd = generate_portfolio(N)
    mag = np.abs(ex)
    exact = math.fsum((mag * pd).tolist()) / math.fsum(mag.tolist())
    block = rows(cached("wmean", "--n", str(N)), "wmean")
    assert len(block) == 3, f"the wmean command printed {len(block)} rows, expected 3"
    vals = {int(c): v for c, v in block}
    d_naive = ulps_between(vals[0], exact)
    d_kahan = ulps_between(vals[2], exact)
    assert d_kahan <= 2, (
        f"the compensated weighted mean is {d_kahan} doubles from the reference {exact!r}; "
        "one or two steps is the algorithm, more means the sums are wrong")
    assert d_naive > d_kahan, (
        "the naive weighted mean landed no further from the reference than the compensated "
        "one; on this portfolio it should be clearly worse")
    for name, code in (("naive", 0), ("pairwise", 1), ("kahan", 2)):
        print(f"    {name:<9s} {vals[code]!r}  {ulps_between(vals[code], exact):>5d} steps  "
              f"{agreeing_significant_digits(vals[code], exact)} digits agree")
    print(f"    exact     {exact!r}")


if _IS_MAIN:
    _try(_check_weighted_mean)

# %% [markdown]
# ## 12. Common mistakes
#
# **A binary built before you moved the checkout.** `make` decides a binary is up to date by
# comparing its timestamp against its source, and a binary that arrived with a copied or
# moved directory is *newer* than the `.c` file beside it — so `make` will not rebuild it and
# you will be grading yesterday's answers. Worse, a compiled lesson that links a library out
# of the virtualenv has that library's absolute path baked into it by the linker, and after a
# move the loader cannot find it at all (`dyld: Library not loaded`). **Run `make clean` after
# moving or copying your checkout, or after rebuilding the virtualenv.** This particular
# lesson links nothing but the system maths library, so only the stale-timestamp half can
# bite here — but the habit is the point, and the repository's execution gate deletes compiled
# artefacts before grading any C lesson for exactly this reason.
#
# **Reaching for `long double`.** It is the first thing everyone tries, and whether it is
# wider than a `double` is a property of the machine, not of C. Section 1 printed what it is
# on yours. Where it is the same 53 bits, the "higher-precision accumulator" is the same
# accumulator with a longer name, and a reference built on it is a reference that silently is
# not one. That is why this lesson's exact total comes from an algorithm rather than a type.
#
# **A tolerance with no `n` and no scale in it.** `abs(a - b) < 1e-9` is a number somebody
# made up. A defensible bound has the element count and the size of the data in it.
#
# **Believing `np.sum` is a plain sum.** It is documented as pairwise (`claims.yaml`), so a
# reconciliation against a spreadsheet is comparing two different algorithms.
#
# **Compensating into a variable you overwrite.** `c = (s - t) + x` instead of `c += ...`
# throws away everything except the last element's correction.
#
# **Reporting a difference without a unit.** Run the cell.

# %%
def _demonstrate_common_mistakes() -> None:
    a, b = 0.1 + 0.2, 0.3
    print(f"    0.1 + 0.2 == 0.3 is {a == b}; they are {ulps_between(a, b)} representable "
          f"double(s) apart and agree to {agreeing_significant_digits(a, b)} significant "
          "digits — adjacent doubles with a decimal boundary between them, which is exactly "
          "why a finding carries both numbers and not one of them")
    bad_tol = abs(a - b) < 1e-9
    print(f"    a 1e-9 tolerance calls that equal ({bad_tol}) and would call a 1e-10 error "
          "on a trillion-euro book equal too")
    s = 0.0
    for _ in range(10):
        s += 0.1
    print(f"    ten additions of 0.1 give {s!r}, {ulps_between(s, 1.0)} step(s) from 1.0")
    ex, _pd = generate_portfolio(N)
    print(f"    np.sum on this portfolio: {float(np.sum(ex))!r}")
    print(f"    your pairwise_sum:        "
          f"{metrics(cached('sums', '--n', str(N)))['pairwise']!r}")
    print("    both pairwise, both correct, not the same number")


if _IS_MAIN:
    _try(_demonstrate_common_mistakes)

# %% [markdown]
# ## 13. Self-check
#
# **1.** Two teams sum the same two million exposures and their totals differ by 1e-6. Team A
# used `np.sum`, Team B used a `for` loop. What is the *first* thing you ask for?
# (a) a bigger tolerance · (b) the summation order and algorithm each side used ·
# (c) the data in higher precision · (d) a re-run on a different machine
#
# **2.** Your `reproduce()` returns `EXPLAINED` on a gap of 3e-6. What have you established?
# (a) the difference is caused by floating point · (b) the difference is immaterial ·
# (c) the worst-case rounding bound cannot rule the difference out · (d) both figures are
# within tolerance of the true value
#
# **3.** Compensated summation gave the identical double for all four orderings in section 6.
# Why is that not a guarantee that it always will?
# (a) it is a guarantee · (b) the compensation is itself a floating-point value and can be
# lost when the sum is far larger than it · (c) the compiler may reorder the loop ·
# (d) `fabs` is inexact
#
# **4.** You must hand an auditor one number that settles whether two figures reconcile.
# Which?
# (a) the relative difference · (b) the number of agreeing significant digits ·
# (c) the ULP distance · (d) whether they are equal bit for bit

# %%
SELF_CHECK = {1: "?", 2: "?", 3: "?", 4: "?"}   # <- put a, b, c or d in each


# The marker holds a salted digest of each answer, not the answer. It still tells you
# immediately which questions are wrong and which section settles each one, but reading this
# cell does not hand you the four letters. The worked reasoning is in the course solution
# bundle.
_MARK = {
    1: ("9feb9d2e2977b8a90d7f131f17347f578b669f253f958ddd7f755b071238e910", "section 6"),
    2: ("24e98482e261597c9a194a0710f83b5fcbed38c50dd1ad1d38c85b7bc75540f6", "section 10"),
    3: ("4c1ca940fc644295aed3aca25e24a0b7a355b44e79e993da9cc5da5cc04f41f3", "section 5"),
    4: ("30714737a9e426524fe6e19815f0f74adc7fdff9784a8d1e53254bd9c1bf2384", "section 10"),
}


def _check_self_check(answers: dict = None) -> None:
    """Mark the four multiple-choice answers, naming the section that settles each."""
    answers = SELF_CHECK if answers is None else answers
    key = _MARK
    wrong = [q for q, (want, _) in key.items()
             if hashlib.sha256(
                 f"P04-L06:{q}:{str(answers.get(q, '?')).strip().lower()}".encode()
             ).hexdigest() != want]
    assert not wrong, (
        "questions " + ", ".join(str(q) for q in wrong) + " are still wrong. Look again at "
        + "; ".join(f"q{q}: {key[q][1]}" for q in wrong) + ".")
    print("self-check: all four right")


if _IS_MAIN:
    _try(_check_self_check)

# %% [markdown]
# ## 14. What you built
#
# Three summations, two instruments and a decision rule — and the finding that the defensible
# bound is too loose to decide anything, which is why the answer to "can you reproduce the
# first line's number?" has to be yes or no rather than nearly.
#
# The next compiled module, module 9, keeps the single pass but takes the memory away: the
# monthly monitoring run does not get a bigger machine, and the gate measures peak RSS while
# you produce the same PSI as an in-memory reference.

# %%
if _IS_MAIN:
    if _FAILURES:
        print(f"\n{len(_FAILURES)} check(s) still failing: {', '.join(_FAILURES)}")
        sys.exit(1)
    print("\nall public checks green — now run:  python tools/grade.py <this lesson>")
