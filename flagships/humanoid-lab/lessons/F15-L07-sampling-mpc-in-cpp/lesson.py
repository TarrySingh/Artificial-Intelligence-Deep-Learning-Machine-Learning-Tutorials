# %% [markdown]
# # F15-L07 · Sampling-based MPC, in C++
#
# **You will build:** a Model Predictive Path Integral planner that swings a cart-pole up from
# hanging — the sampling, the scoring and the refit written in C++ — plus the measurement that
# says whether it could run on a real 100 Hz control loop.
#
# **Time:** ~75 minutes · **Runs on:** a laptop CPU, no GPU, no download
# · **Prerequisites:** F15-L01 (mjModel vs mjData), F15-L03 (PD control), F15-L06 (the C loop)
#
# By the end you will be able to:
# 1. Implement the three stages of an MPPI planner in C++ — sample, score, refit — and pass
#    the binary's own self-test.
# 2. Measure this machine's rollout throughput in C++ and in Python and report the ratio.
# 3. Compute the sample-and-horizon budget a 100 Hz control tick affords at a measured
#    throughput, and decide whether a configuration is real-time.
# 4. Explain why subtracting the minimum cost before exponentiating is what stops the MPPI
#    weights underflowing to zero.
# 5. Show that scoring a candidate plan leaves the live simulation untouched, and say what
#    breaks when it does not.
#
# Everything you implement in this lesson lives in **`lesson.cpp`**. This notebook builds it,
# drives it, checks it and measures it. Two small exercises at the end are in Python, and they
# exist to let you feel the difference the language makes.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np

print("mujoco", mujoco.__version__, "· numpy", np.__version__, "· python",
      sys.version.split()[0])

# True in a notebook and when this file is run as a script; False when the autograder
# imports it. Each exercise check below is called under this guard, so the cell you are
# sitting in reports on itself, while importing the lesson never runs anything.
_IS_MAIN = __name__ == "__main__"

# This file is the source of truth for the student build; the reference build lives beside it
# in the reference tree and is compiled from its own .cpp. Detecting which one is running keeps the
# two files identical apart from the exercise bodies.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:  # a notebook has no __file__
    _HERE = Path.cwd()
if _HERE.name == "solutions":
    LESSON_DIR, CPP_SRC, BIN = _HERE.parent, "solutions/lesson.cpp", "solutions/lesson_bin"
else:
    LESSON_DIR, CPP_SRC, BIN = _HERE, "lesson.cpp", "lesson_bin"
MODEL_REL = "assets/cartpole.xml"
MODEL_PATH = LESSON_DIR / MODEL_REL

# The objective, mirrored exactly from lesson.cpp. Both languages must optimise the same
# thing or no comparison between them means anything.
W_ANGLE, W_CART, W_THETADOT, W_XDOT, W_CTRL = 10.0, 1.0, 0.05, 0.05, 0.01

_BUILD = None


def build(verbose: bool = True):
    """Compile the C++ source with make. Cached: the compiler runs once per session."""
    global _BUILD
    if _BUILD is None:
        proc = subprocess.run(
            ["make", "-C", str(LESSON_DIR), f"PYTHON={sys.executable}",
             f"SRC={CPP_SRC}", f"BIN={BIN}"],
            capture_output=True, text=True, timeout=600)
        _BUILD = (proc.returncode == 0, (proc.stdout + proc.stderr).strip())
    ok, out = _BUILD
    if verbose:
        print("build OK" if ok else "BUILD FAILED\n" + out)
    return ok, out


def run_cpp(*args, timeout: int = 900) -> str:
    """Run the compiled binary and return its stdout.

    Exit code 2 means one of the three C++ exercises is still a stub, so it is re-raised as
    NotImplementedError — the grader then reports TODO instead of an error.
    """
    ok, out = build(verbose=False)
    if not ok:
        raise RuntimeError("the C++ build failed; run build() to see the compiler output\n" + out)
    proc = subprocess.run([str(LESSON_DIR / BIN), *args, "--model", MODEL_REL],
                          cwd=LESSON_DIR, capture_output=True, text=True, timeout=timeout)
    if proc.returncode == 2:
        raise NotImplementedError(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"{BIN} {' '.join(args)} exited {proc.returncode}\n{proc.stderr.strip()}")
    return proc.stdout


def metrics(text: str) -> dict:
    """Parse the binary's `@ key=value` report lines into a dict of floats."""
    return {k: float(v) for k, v in
            (line[2:].split("=", 1) for line in text.splitlines() if line.startswith("@ "))}


def rows(text: str, tag: str) -> list:
    """Parse the binary's `<tag> v1 v2 ...` lines into a list of lists of floats."""
    return [[float(v) for v in line.split()[1:]]
            for line in text.splitlines() if line.startswith(tag + " ")]


def make_test():
    """Run `make test` — the C++ self-test. Returns (exit code, combined output)."""
    proc = subprocess.run(
        ["make", "-C", str(LESSON_DIR), f"PYTHON={sys.executable}",
         f"SRC={CPP_SRC}", f"BIN={BIN}", "test"],
        capture_output=True, text=True, timeout=900)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def fresh():
    """A (model, data, scratch) triple reset to the model's `hanging` keyframe."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data, scratch = mujoco.MjData(model), mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data, scratch


def stage_cost(scratch, u: float) -> float:
    """The per-step objective, identical to stage_cost() in lesson.cpp."""
    x, th = float(scratch.qpos[0]), float(scratch.qpos[1])
    xd, thd = float(scratch.qvel[0]), float(scratch.qvel[1])
    return (W_ANGLE * (1.0 - math.cos(th)) + W_CART * x * x
            + W_THETADOT * thd * thd + W_XDOT * xd * xd + W_CTRL * u * u)


build()

# %% [markdown]
# ## 1. The problem a feedback law cannot solve
#
# The cart-pole starts *hanging*: pole straight down, cart centred, at rest. The goal is the
# pole upright and the cart back at the origin.
#
# In F15-L03 you built a PD controller. Point one at this problem and watch what happens. The
# code below is given — run it.

# %%
def pd_baseline(kp: float = 30.0, kd: float = 3.0, ticks: int = 400) -> dict:
    """Drive the cart with proportional-derivative feedback on the upright pole angle."""
    model, data, _ = fresh()
    lo, hi = model.actuator_ctrlrange[0]
    start_error = math.atan2(math.sin(data.qpos[1]), math.cos(data.qpos[1]))
    start_command = -kp * start_error - kd * float(data.qvel[1])
    saturated, window = 0, []
    for _ in range(ticks):
        err = math.atan2(math.sin(data.qpos[1]), math.cos(data.qpos[1]))  # wrapped to (-pi, pi]
        want = -kp * err - kd * float(data.qvel[1])
        data.ctrl[0] = float(np.clip(want, lo, hi))
        saturated += int(abs(want) > float(hi))
        mujoco.mj_step(model, data)
        window.append(abs(math.atan2(math.sin(data.qpos[1]), math.cos(data.qpos[1]))))
    return {"final_abs_theta": window[-1],
            "upright_fraction": sum(1 for a in window[-100:] if a < 0.2) / 100,
            "start_error": start_error, "start_command": start_command,
            "ctrl_limit": float(hi), "saturated_fraction": saturated / ticks,
            "seconds": ticks * float(model.opt.timestep)}


_pd = pd_baseline()
print(f"hanging start: the pole is {_pd['start_error']:.3f} rad from upright, so the PD law "
      f"asks for {_pd['start_command']:+.2f} of control")
print(f"the actuator accepts +/-{_pd['ctrl_limit']:.0f}, so it saturates at once — and stays "
      f"saturated for {_pd['saturated_fraction']:.0%} of the run")
print(f"after {_pd['seconds']:.0f} s of PD: final |theta| = {_pd['final_abs_theta']:.3f} rad, "
      f"upright for {_pd['upright_fraction']:.2f} of the last second")

# %% [markdown]
# The law is not idle — it pins the actuator to its limit for most of the run — and it still
# does not hold the pole up. That is the interesting part. A PD law is a *local* stabiliser: a
# linearisation about upright, asked here to work half a revolution away from the point it was
# linearised at. Swinging up needs the cart driven *away* from where the law wants it, more
# than once, to pump energy in over several swings; a law with no representation of the future
# cannot choose to do that. The actuator is not the limitation, as the planner you are about
# to write will demonstrate on exactly the same one.
#
# A sampling planner has no such problem, because it does not reason about the objective at
# all. It guesses thousands of plans, simulates each one, and keeps what scored well. That is
# the entire idea, and it is why a real stack spends its CPU on rollouts — MuJoCo MPC ships
# exactly this kind of derivative-free planner alongside its gradient-based ones
# (`claims.yaml`, `mjpc-real-time-sampling`).
#
# ## 2. What one rollout costs
#
# Before writing a planner, look at the machine you are planning on.

# %%
_model, _data, _scratch = fresh()
print(f"nq={_model.nq}  nv={_model.nv}  nu={_model.nu}  keyframes={_model.nkey}")
print(f"timestep = {_model.opt.timestep} s  ->  control rate = {1 / _model.opt.timestep:.0f} Hz")
print(f"ctrlrange = {_model.actuator_ctrlrange[0]}")
print(f"hanging keyframe: qpos = {np.array(_data.qpos)}  (hinge = pi is straight down)")
print(f"\na horizon of 40 steps is {40 * _model.opt.timestep:.2f} s of predicted motion;")
print(f"200 samples of that horizon is {200 * 40:,} mj_step calls for ONE control tick.")

# %% [markdown]
# ## 3. The loop you are about to write
#
# Open **`lesson.cpp`**. Three functions are stubs, and together they are the planner:
#
# 1. `sample_controls` — draw `samples` candidate plans around the current plan.
# 2. `rollout_cost` — simulate one plan on a *copy* of the state and score it.
# 3. `refit_mean` — fold the scores back into a new plan.
#
# The harness around them already works: it applies only the first control of each refitted
# plan, steps the real simulation once, shifts the plan along, and repeats.
#
# Your feedback loop is `make test`. Run it now — it should report three TODOs.

# %%
_code, _out = make_test()
print(_out)
print(f"exit code {_code}  (0 = all pass, 2 = something is still a stub, 1 = a real failure)")

# %% [markdown]
# ## 4. Exercise 1 — `sample_controls` in `lesson.cpp`
#
# Fill `out` with `samples` plans of `horizon` controls, row-major, each one the current plan
# plus fresh Gaussian noise, clamped into the actuator's range:
#
# `out[i*horizon + t] = clamp(mean[t] + sigma * z, lo, hi)` with a new `z ~ Normal(0,1)` every
# time. The full brief, including the worked example, is in the comment above the function.
#
# The check below pulls the buffer back out of the binary and tests its statistics from here.

# %%
def _check_sampling() -> None:
    text = run_cpp("sampledump", "--samples", "600", "--horizon", "6",
                   "--sigma", "0.35", "--offset", "-0.2", "--seed", "99")
    head, block = metrics(text), np.array(rows(text, "row"))
    assert block.shape == (600, 6), (
        f"sampledump returned {block.shape}, expected (600, 6) — `out` must hold "
        "samples*horizon values, not one plan")
    assert block.min() >= head["lo"] - 1e-12 and block.max() <= head["hi"] + 1e-12, (
        f"values ran from {block.min():.3f} to {block.max():.3f} but ctrlrange is "
        f"[{head['lo']}, {head['hi']}] — clamp after adding the noise")
    assert not np.allclose(block[0], block[1]), (
        "the first two plans are identical — you reused one noise draw across samples")
    assert abs(float(block.mean()) + 0.2) < 0.04, (
        f"the sample mean is {block.mean():.4f} but the plan being perturbed was -0.2 — "
        "add mean[t] to the noise")
    assert abs(float(block.std()) - 0.35) < 0.04, (
        f"the spread is {block.std():.4f} against sigma = 0.35 — multiply a *standard* normal "
        "by sigma; normal_distribution's second argument is a standard deviation")
    again = np.array(rows(run_cpp("sampledump", "--samples", "600", "--horizon", "6",
                                  "--sigma", "0.35", "--offset", "-0.2", "--seed", "99"), "row"))
    assert np.array_equal(block, again), (
        "the same seed gave a different buffer — draw from the rng you were handed")
    print(f"exercise 1 looks right: 600x6 samples, mean {block.mean():+.4f}, "
          f"sd {block.std():.4f}, all within [{head['lo']:.0f}, {head['hi']:.0f}]")


if _IS_MAIN:
    _check_sampling()


# %% [markdown]
# ## 5. Exercise 2 — `rollout_cost` in `lesson.cpp`
#
# Branch `scratch` off `start` with `load_state`, then for each `t`: write `controls[t]` into
# `scratch->ctrl[0]`, call `mj_step` once, and add `stage_cost(scratch, controls[t])`.
#
# `start` is `const`. Step it instead of the copy and the planner's hypothetical futures
# become real motion — the robot is then driven by its own imagination, and no two samples
# start from the same state, so their costs cannot be compared.

# %%
def _check_rollout() -> None:
    rng = random.Random(20260916)
    plan = [max(-1.0, min(1.0, rng.gauss(0.0, 0.6))) for _ in range(30)]
    # The scratch file lives outside the lesson directory on purpose: anything written under
    # assets/ would be swept into the student bundle if this check were interrupted mid-run.
    fd, ctrl_path = tempfile.mkstemp(suffix=".txt", prefix="f15l07_check_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(repr(v) for v in plan))
        first = metrics(run_cpp("costof", "--controls", ctrl_path))
        second = metrics(run_cpp("costof", "--controls", ctrl_path))
        assert first["cost"] > 0.0 and math.isfinite(first["cost"]), (
            f"the rollout scored {first['cost']} — every weight is non-negative and the pole "
            "starts hanging, so the total must be positive and finite")
        assert first["cost"] == second["cost"], (
            "scoring the same plan twice gave different answers — scratch state is leaking "
            "between calls; branch it from `start` at the top of every call")
        assert abs(first["start_theta_after"] - math.pi) < 1e-12, (
            f"after scoring, the live state's hinge is {first['start_theta_after']} but it "
            "started at pi — you stepped `start` instead of `scratch`")
        assert first["start_time_after"] == 0.0, (
            "the live simulation clock advanced during scoring — again, you are stepping the "
            "real mjData rather than the copy")

        Path(ctrl_path).write_text("\n".join(repr(v) for v in plan * 2))
        longer = metrics(run_cpp("costof", "--controls", ctrl_path))
        assert longer["cost"] > first["cost"], (
            "doubling the horizon did not raise the cost — you are returning the last stage's "
            "cost rather than the sum over the horizon")
        print(f"exercise 2 looks right: 30 steps score {first['cost']:.3f}, 60 steps "
              f"{longer['cost']:.3f}, and the live state never moved")
    finally:
        os.unlink(ctrl_path)


if _IS_MAIN:
    _check_rollout()


# %% [markdown]
# ## 6. Exercise 3 — `refit_mean` in `lesson.cpp`
#
# Turn costs into weights and average the plans under them:
#
# `w_i = exp(-(cost_i - min_j cost_j) / lambda)`, then `mean[t] = sum_i (w_i / sum_j w_j) * u_i[t]`.
#
# This is the MPPI update (Williams, Aldrich and Theodorou, 2015 — see `claims.yaml`). The
# subtraction of the minimum cost is the part everyone omits and it is the part that matters:
# it changes nothing mathematically, because the constant cancels between the numerator and
# the denominator, but without it `exp(-30000/30)` is exactly `0`, every weight vanishes, and
# you divide zero by zero.
#
# The alternative refit is the cross-entropy one: keep the best-scoring *elite* fraction and
# average only those. MuJoCo MPC ships that planner too (`claims.yaml`,
# `mjpc-cross-entropy-planner`). Softmax is a soft version of the same instinct.

# %%
def _check_refit() -> None:
    equal = run_cpp("refitdump", "--samples", "8", "--horizon", "5", "--costs", "equal")
    mean_eq, arith = rows(equal, "mean")[0], rows(equal, "arithmetic")[0]
    assert np.allclose(mean_eq, arith, atol=1e-12), (
        "with every cost equal every weight is equal, so the refit must be the plain "
        "arithmetic mean of the samples — check that you divide by the sum of the weights")

    spike = run_cpp("refitdump", "--samples", "8", "--horizon", "5", "--costs", "spike")
    assert np.allclose(rows(spike, "mean")[0], rows(spike, "first")[0], atol=1e-6), (
        "one sample was astronomically cheaper than the rest, so the refit should be that "
        "sample almost exactly — your weights are not falling off with cost")

    small = rows(run_cpp("refitdump", "--samples", "8", "--horizon", "5",
                         "--costs", "graded", "--offset", "0"), "mean")[0]
    big = rows(run_cpp("refitdump", "--samples", "8", "--horizon", "5",
                       "--costs", "graded", "--offset", "1e6"), "mean")[0]
    assert all(math.isfinite(v) for v in big), (
        "shifting every cost up by 1e6 produced a non-finite plan — exp(-1e6/lambda) "
        "underflows to exactly 0, so every weight vanished and you divided by zero")
    assert np.allclose(small, big, atol=1e-12), (
        "adding the same constant to every cost changed the answer — it must cancel; "
        "subtract the minimum cost before exponentiating")
    print(f"exercise 3 looks right: equal costs give the plain mean, a cost spike gives the "
          f"winner, and a +1e6 shift moves the plan by "
          f"{max(abs(a - b) for a, b in zip(small, big)):.1e}")


if _IS_MAIN:
    _check_refit()


# %% [markdown]
# ## 7. Exercise 4 — the same rollout, in Python
#
# Now write `rollout_cost` again, here, in Python. Same objective, same model, same start —
# the only thing that changes is the language. This is what makes the comparison later an
# apples-to-apples one rather than a slogan.

# %%
def python_rollout_cost(model, data, scratch, controls) -> float:
    """Score one plan in Python, exactly as rollout_cost() does in lesson.cpp.

    Branch `scratch` off `data` with `mujoco.mj_copyData(scratch, model, data)`, then for each
    `u` in `controls`: set `scratch.ctrl[0] = u`, call `mujoco.mj_step(model, scratch)`, and
    add `stage_cost(scratch, u)` to a running total. Leave `data` untouched.

    Example (longer plans cost more, because every weight is non-negative):
        >>> model, data, scratch = fresh()
        >>> two = python_rollout_cost(model, data, scratch, [0.0, 0.0])
        >>> one = python_rollout_cost(model, data, scratch, [0.0])
        >>> two > one > 0
        True

    Returns:
        A single float — the summed stage cost over the plan.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_python_rollout() -> None:
    rng = random.Random(7771)
    plan = [max(-1.0, min(1.0, rng.gauss(0.0, 0.6))) for _ in range(40)]
    model, data, scratch = fresh()
    mine = python_rollout_cost(model, data, scratch, plan)
    assert isinstance(mine, float), (
        f"python_rollout_cost returned {type(mine).__name__} — sum into a plain float and "
        "return it; a numpy array is not a cost")
    assert abs(float(data.qpos[1]) - math.pi) < 1e-12, (
        "the live `data` moved — you stepped it instead of `scratch`; copy first")

    fd, ctrl_path = tempfile.mkstemp(suffix=".txt", prefix="f15l07_py_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(repr(v) for v in plan))
        theirs = metrics(run_cpp("costof", "--controls", ctrl_path))["cost"]
    finally:
        os.unlink(ctrl_path)
    rel = abs(mine - theirs) / abs(theirs)
    assert rel < 1e-9, (
        f"Python scored {mine!r} and C++ scored {theirs!r} (relative gap {rel:.2e}) — the two "
        "loops must be the same computation. Check that you step AFTER setting ctrl, that you "
        "score the state after the step, and that you start from the keyframe, not qpos0")
    print(f"exercise 4 looks right: Python {mine:.9f} vs C++ {theirs:.9f}, "
          f"relative gap {rel:.1e}")


if _IS_MAIN:
    _check_python_rollout()


# %% [markdown]
# ## 8. Exercise 5 — what the control tick can afford
#
# A planner is real-time only if one planning tick fits inside one control period. At 100 Hz
# that is 10 ms to spend on `samples * horizon` calls to `mj_step`. Write the arithmetic that
# turns a measured throughput into a verdict.

# %%
def rollout_budget(samples: int, horizon: int, control_hz: float,
                   steps_per_second: float) -> dict:
    """Decide whether a planner configuration fits inside one control tick.

    Example:
        >>> b = rollout_budget(200, 40, 100.0, 1_000_000.0)
        >>> b["steps_per_tick"], b["steps_per_second_required"]
        (8000, 800000.0)
        >>> b["tick_seconds"], b["plan_seconds_per_tick"]
        (0.01, 0.008)
        >>> b["realtime"], b["max_samples"]
        (True, 250)

    Returns:
        dict with exactly these six keys:
          "steps_per_tick"            int, samples * horizon
          "steps_per_second_required" float, steps_per_tick * control_hz
          "tick_seconds"              float, 1 / control_hz
          "plan_seconds_per_tick"     float, steps_per_tick / steps_per_second
          "realtime"                  bool, True when plan_seconds_per_tick <= tick_seconds.
                                      Planning that exactly fills the tick still counts as
                                      fitting, so compare with <=, not <.
          "max_samples"               int, the largest WHOLE number of samples that still
                                      fits: floor(steps_per_second / (control_hz * horizon)).
                                      Round DOWN — 308.6 affordable samples means 308.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_budget() -> None:
    b = rollout_budget(200, 40, 100.0, 1_000_000.0)
    expected = {"steps_per_tick", "steps_per_second_required", "tick_seconds",
                "plan_seconds_per_tick", "realtime", "max_samples"}
    assert set(b) == expected, (
        f"keys were {sorted(b)}, expected {sorted(expected)} — spell them as in the docstring")
    assert b["steps_per_tick"] == 8000, (
        f"steps_per_tick was {b['steps_per_tick']}, expected 200*40 = 8000")
    assert b["plan_seconds_per_tick"] == 0.008, (
        f"plan_seconds_per_tick was {b['plan_seconds_per_tick']}, expected 8000/1e6 = 0.008 — "
        "divide the work by the throughput, not the other way round")
    assert b["realtime"] is True, (
        "8 ms of planning fits inside a 10 ms tick, so realtime must be True — compare "
        "plan_seconds_per_tick against tick_seconds")
    assert b["max_samples"] == 250, (
        f"max_samples was {b['max_samples']}, expected 1e6/(100*40) = 250 — floor it to a "
        "whole number of samples")
    slow = rollout_budget(200, 40, 100.0, 300_000.0)
    assert slow["realtime"] is False, (
        "at 300k steps/s, 8000 steps take 26.7 ms and cannot fit a 10 ms tick — realtime "
        "must be False here")
    assert slow["max_samples"] == 75, (
        f"max_samples was {slow['max_samples']}, expected 300000/(100*40) = 75")
    # The three cases above all divide out exactly, so they cannot tell flooring from
    # rounding. This one can: 1234567 / (100 * 40) = 308.64...
    odd = rollout_budget(200, 40, 100.0, 1_234_567.0)
    assert odd["max_samples"] == 308, (
        f"max_samples was {odd['max_samples']}, expected floor(1234567/(100*40)) = 308 — "
        "308.64 affordable samples is 308 samples; rounding up buys a tick you cannot pay for")
    # A planner that exactly fills the tick still fits.
    exact = rollout_budget(100, 40, 100.0, 400_000.0)
    assert exact["plan_seconds_per_tick"] == exact["tick_seconds"], (
        f"4000 steps at 400k steps/s is exactly one 10 ms tick, but plan_seconds_per_tick "
        f"came out {exact['plan_seconds_per_tick']} against a tick of {exact['tick_seconds']}")
    assert exact["realtime"] is True, (
        "planning that exactly fills the tick still fits — compare with <=, not <")
    print(f"exercise 5 looks right: at 1e6 steps/s a 200x40 planner needs "
          f"{b['plan_seconds_per_tick'] * 1e3:.1f} ms of a "
          f"{b['tick_seconds'] * 1e3:.0f} ms tick; at 3e5 steps/s it needs "
          f"{slow['plan_seconds_per_tick'] * 1e3:.1f} ms")


if _IS_MAIN:
    _check_budget()


# %% [markdown]
# ## 9. The payoff — swing it up, then measure the language
#
# With all five exercises done, this runs the planner end to end and then benchmarks the same
# rollout loop in both languages on identical work. Every number below is produced by the code
# you just wrote; none of them are typed into this notebook.

# %%
PY_BENCH_ROLLOUTS, PY_BENCH_HORIZON = 800, 40


def python_throughput(rollouts: int = PY_BENCH_ROLLOUTS, horizon: int = PY_BENCH_HORIZON) -> dict:
    """Time the Python rollout loop on exactly the work `bench` does in C++."""
    model, data, scratch = fresh()
    rng = random.Random(31337)
    plans = [[max(-1.0, min(1.0, rng.gauss(0.0, 0.5))) for _ in range(horizon)]
             for _ in range(rollouts)]
    t0 = time.perf_counter()
    for plan in plans:
        python_rollout_cost(model, data, scratch, plan)
    wall = time.perf_counter() - t0
    steps = rollouts * horizon
    return {"rollouts": rollouts, "horizon": horizon, "steps": steps, "wall_seconds": wall,
            "steps_per_second": steps / wall, "rollouts_per_second": rollouts / wall}


def report() -> None:
    swing = metrics(run_cpp("mpc"))
    print("closed-loop swing-up")
    print(f"  {swing['samples']:.0f} samples x {swing['horizon']:.0f} steps "
          f"({swing['horizon_seconds']:.2f} s of prediction) for {swing['ticks']:.0f} ticks")
    print(f"  {swing['planning_steps']:,.0f} mj_step calls in {swing['wall_seconds']:.2f} s "
          f"wall = {swing['steps_per_second']:,.0f} steps/s")
    print(f"  final |theta| {swing['final_abs_theta']:.4f} rad · upright fraction "
          f"{swing['upright_fraction']:.2f} · cart stayed within "
          f"{swing['max_abs_cart']:.2f} m")
    print(f"  swung up: {'yes' if swing['swung_up'] else 'no'}   "
          f"(PD baseline finished at |theta| = {_pd['final_abs_theta']:.3f} rad)")

    cpp = metrics(run_cpp("bench", "--samples", str(PY_BENCH_ROLLOUTS),
                          "--horizon", str(PY_BENCH_HORIZON)))
    py = python_throughput()
    ratio = cpp["steps_per_second"] / py["steps_per_second"]
    print(f"\nidentical work, two languages: {int(cpp['steps']):,} mj_step calls")
    print(f"  C++    {cpp['wall_seconds'] * 1e3:8.1f} ms  {cpp['steps_per_second']:12,.0f} steps/s"
          f"  {cpp['rollouts_per_second']:9,.0f} rollouts/s")
    print(f"  Python {py['wall_seconds'] * 1e3:8.1f} ms  {py['steps_per_second']:12,.0f} steps/s"
          f"  {py['rollouts_per_second']:9,.0f} rollouts/s")
    print(f"  C++ is {ratio:.1f}x faster on the same mj_step calls")

    hz = swing["control_hz"]
    n, h = int(swing["samples"]), int(swing["horizon"])
    print(f"\ncan this planner close a {hz:.0f} Hz loop?  ({n} samples x {h} steps)")
    for name, rate in (("C++", cpp["steps_per_second"]), ("Python", py["steps_per_second"])):
        b = rollout_budget(n, h, hz, rate)
        print(f"  {name:6s} {b['plan_seconds_per_tick'] * 1e3:7.2f} ms per "
              f"{b['tick_seconds'] * 1e3:.0f} ms tick -> "
              f"{'real-time' if b['realtime'] else 'TOO SLOW':9s} "
              f"(affords {b['max_samples']:,} samples at this horizon)")

    print(f"\nthe trade-off, at the C++ rate of {cpp['steps_per_second']:,.0f} steps/s")
    print(f"  {'samples':>8} {'horizon':>8} {'predict':>9} {'plan/tick':>10}  verdict")
    for n_s, h_s in ((100, 20), (200, 40), (500, 40), (1000, 60), (2000, 100)):
        b = rollout_budget(n_s, h_s, hz, cpp["steps_per_second"])
        print(f"  {n_s:8d} {h_s:8d} {h_s / hz:8.2f}s {b['plan_seconds_per_tick'] * 1e3:9.2f}ms  "
              f"{'fits' if b['realtime'] else 'misses the tick'}")


# %% [markdown]
# ## 10. Common mistakes
#
# - **Stepping `start` instead of `scratch`.** The planner's imagined futures become real
#   motion. Symptom: the cost of a plan depends on how many plans you scored before it, and
#   the simulation clock races ahead of the control loop.
# - **Forgetting to subtract the minimum cost.** Works on toy costs, fails silently on real
#   ones: every weight underflows to `0`, the sum is `0`, and the plan becomes `NaN`. Once one
#   `NaN` enters the mean it never leaves.
# - **One noise draw per sample instead of per (sample, step).** Each candidate plan is then a
#   constant offset, the planner can only search over constants, and a swing-up needs a
#   control that changes sign.
# - **Clamping before adding the mean.** Clamp the final value: it is the actuator that has
#   limits, not the noise.
# - **Applying the whole plan.** Applying all `horizon` controls before replanning is
#   open-loop optimisation, not MPC. Apply the first control, then replan from what actually
#   happened.
# - **Re-planning from zero every tick.** The shift-by-one warm start is most of why this
#   works at 200 samples. Without it you are starting the search over 100 times a second.
# - **Hard-coding the timestep.** This model overrides MuJoCo's documented default of 0.002 s
#   (`claims.yaml`). Read `model->opt.timestep` and your horizon stays honest.
# - **Quoting a speed-up you did not measure.** The ratio between C++ and Python here is a
#   property of this model, this machine and this loop. Measure it; do not repeat it.

# %%
# The second mistake, made concrete. There is no C++ here: this is the same exp() your
# refit_mean calls, on costs the size this task actually produces.
def weight_spread(costs, lam: float = 30.0) -> dict:
    """Softmax weights with and without the minimum-cost shift, side by side."""
    best = min(costs)
    return {"unshifted_sum": sum(math.exp(-c / lam) for c in costs),
            "shifted_sum": sum(math.exp(-(c - best) / lam) for c in costs)}


if _IS_MAIN:
    for label, _costs in (("a toy cost", [3.0, 5.0, 8.0]),
                          ("this task", [310.0, 480.0, 900.0]),
                          ("a long horizon", [30000.0, 31000.0, 32000.0])):
        _w = weight_spread(_costs)
        _verdict = "usable" if _w["unshifted_sum"] > 0.0 else "0 -> NaN plan"
        print(f"  {label:<15s} costs {_costs[0]:>8.0f}..{_costs[-1]:<8.0f} "
              f"unshifted sum {_w['unshifted_sum']:.3e} ({_verdict:13s})  "
              f"shifted sum {_w['shifted_sum']:.4f}")
    print("\n  the shifted sum can never be zero: the best sample always weighs exp(0) = 1")

# %% [markdown]
# ## 11. Self-check
#
# 1. From hanging, the PD law saturates the actuator immediately and still fails to hold the
#    pole upright. What is the real reason?
#    - (a) the gains are too low
#    - (b) it is a local stabiliser with no model of the future, so it cannot choose to drive
#          the cart the wrong way first and pump energy in over several swings
#    - (c) the actuator is too weak to get this pole upright at all
#    - (d) contact is disabled in the model
#
# 2. What does subtracting `min(cost)` before the exponential do?
#    - (a) it makes the planner greedier
#    - (b) nothing mathematically — the constant cancels — but it stops every weight
#          underflowing to zero
#    - (c) it normalises the weights so they sum to one
#    - (d) it converts costs into rewards
#
# 3. You replace the softmax refit with "average the best 10% of samples". What have you
#    built?
#    - (a) nothing that works
#    - (b) a cross-entropy-style elite refit, a real alternative that MuJoCo MPC also ships
#    - (c) gradient descent on the cost
#    - (d) the same algorithm, written differently
#
# 4. You accidentally step `start` rather than `scratch` inside `rollout_cost`. What is the
#    first symptom?
#    - (a) the cost goes negative
#    - (b) `mj_step` refuses to run on a const pointer
#    - (c) scoring the same plan twice gives different answers, because every rollout leaves
#          the state somewhere new
#    - (d) nothing, MuJoCo copies `mjData` automatically
#
# 5. Your benchmark says 1,200,000 steps/s. At 100 Hz with 200 samples, you double the horizon
#    from 40 to 80. What happens to the sample count you can afford?
#    - (a) it is unchanged
#    - (b) it halves
#    - (c) it quarters
#    - (d) it doubles
#
# Answers, with reasoning, are published in the course solution bundle.

# %%
# The payoff, end to end. Every number this prints is produced by the code you wrote.
if _IS_MAIN:
    report()

# %% [markdown]
# ## What you built, and where it goes next
#
# A complete sampling planner: draw plans, simulate them on copies of the state, weight them
# by what they cost, and act on the first control of the result — then throw the rest away and
# do it again. You also measured the thing that decides whether it can run at all, which is
# how many times a second this machine can call `mj_step`.
#
# That measurement is the systems argument in one number. The algorithm is indifferent to the
# language; the control rate is not. The next lesson in the Humanoid Lab takes the same loop
# to a model with contact, where each `mj_step` costs far more and the sample budget gets
# decided for you.
