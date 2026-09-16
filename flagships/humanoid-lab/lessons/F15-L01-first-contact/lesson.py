# %% [markdown]
# # F15-L01 · First contact with a humanoid
#
# **You will build:** a three-instrument panel for MuJoCo's own humanoid — a clock
# (`step_for`), a posture sensor (`com_height`) and a stopwatch (`real_time_factor`).
#
# **Time:** ~45 minutes · **Runs on:** a laptop CPU, no GPU, no download
# · **Prerequisites:** none
#
# By the end you will be able to:
# 1. Locate any simulation quantity in either `mjModel` or `mjData`, and say why it lives there.
# 2. Implement `step_for(seconds)` so that simulated time advances by a whole number of steps.
# 3. Compute whole-body centre-of-mass height from state, and show it differs from root height.
# 4. Measure this machine's real-time factor and explain why simulated time is not wall time.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import hashlib
import math
import time
import urllib.request
from pathlib import Path

import mujoco
import numpy as np

print("mujoco", mujoco.__version__, "· numpy", np.__version__)

# The one model we use, shipped next to this notebook in assets/ (Apache-2.0, see SOURCE.md).
MODEL_URL = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/humanoid/humanoid.xml"
)
MODEL_FILENAME = "humanoid.xml"

# Simulated time accumulates in floating point, so an exact comparison against a target is
# unreliable by a few parts in 1e14. Every time comparison in this lesson uses this slack.
TIME_EPS = 1e-9


def humanoid_xml_path() -> Path:
    """Return the path to the cached humanoid XML, downloading it only if it is missing.

    The file ships inside this lesson's assets/ directory, so the normal path is offline and
    nothing is fetched. The download branch exists only for a truncated checkout.
    """
    try:
        here = Path(__file__).resolve().parent
    except NameError:  # a notebook has no __file__
        here = Path.cwd()
    candidates = [
        here / "assets" / MODEL_FILENAME,
        here.parent / "assets" / MODEL_FILENAME,
        Path.cwd() / "assets" / MODEL_FILENAME,
        Path.cwd().parent / "assets" / MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    target = candidates[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"cached model absent; fetching once from {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, target)  # noqa: S310 - pinned https URL above
    return target


def load_humanoid():
    """Compile the humanoid XML and hand back a fresh (model, data) pair."""
    model = mujoco.MjModel.from_xml_path(str(humanoid_xml_path()))
    return model, mujoco.MjData(model)


MODEL, DATA = load_humanoid()
print("compiled:", humanoid_xml_path().name)

# %% [markdown]
# ## 1. Two objects, and the difference is the whole lesson
#
# `mjModel` is the compiled description: what the robot *is*. It does not change while you
# simulate. `mjData` is the state: what the robot is *doing right now* — positions,
# velocities, forces, the clock. One model can serve many independent `mjData` instances.
#
# Run the cell. Every number below is read out of the compiled model, not typed by hand.

# %%
print("mjModel — constant description")
for _name in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nkey"):
    print(f"  {_name:7s} = {getattr(MODEL, _name)}")
print(f"  timestep = {MODEL.opt.timestep} s")
print(f"  gravity  = {MODEL.opt.gravity[2]:.2f} m/s^2 (z)")
print(f"  mass     = {MODEL.body_mass.sum():.3f} kg total")

print("\nmjData — live state")
print(f"  time = {DATA.time} s")
print(f"  qpos holds {DATA.qpos.shape[0]} numbers, qvel holds {DATA.qvel.shape[0]}")

# %%
# Two scratch pads, one model: mjData instances are independent.
_left, _right = mujoco.MjData(MODEL), mujoco.MjData(MODEL)
for _ in range(10):
    mujoco.mj_step(MODEL, _left)
print(f"left.time={_left.time:.3f}s  right.time={_right.time:.3f}s  "
      f"(the model they share was never modified)")

# %% [markdown]
# ## 2. Why `nq` is not `nv`
#
# A free joint carries a 3-vector position and a 4-number quaternion in `qpos`, but only a
# linear and an angular 3-vector in `qvel`. Orientation therefore costs one more slot in
# `qpos` than in `qvel`. Hinges cost one of each.
#
# Run this to see where each joint's numbers actually live.

# %%
def joint_table(model, limit: int = 6) -> list:
    """Rows of (name, joint type, qpos address, qvel address) for the first `limit` joints."""
    rows = []
    for j in range(min(model.njnt, limit)):
        rows.append((
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
            mujoco.mjtJoint(model.jnt_type[j]).name,
            int(model.jnt_qposadr[j]),
            int(model.jnt_dofadr[j]),
        ))
    return rows


print(f"{'joint':18s} {'type':12s} {'qpos@':>6s} {'qvel@':>6s}")
for _jname, _jtype, _qadr, _vadr in joint_table(MODEL):
    print(f"{_jname:18s} {_jtype:12s} {_qadr:6d} {_vadr:6d}")
print(f"\nnq - nv = {MODEL.nq - MODEL.nv}  (one quaternion's worth of extra position slots)")
print(f"free joint qpos block: {np.array(MODEL.qpos0[:7])}")
print("                       ^ x y z, then quaternion w x y z")

# %% [markdown]
# ## 3. One step of physics
#
# `mujoco.mj_step(model, data)` integrates the state forward by exactly `model.opt.timestep`
# seconds of *simulated* time and advances `data.time` by the same amount. It is the only
# thing that moves the clock. Anything you assign to `data.time` simulates nothing.

# %%
_probe = mujoco.MjData(MODEL)
_t_before, _z_before = _probe.time, float(_probe.qpos[2])
mujoco.mj_step(MODEL, _probe)
print(f"time   {_t_before} -> {_probe.time}  (+{_probe.time - _t_before} s)")
print(f"root z {_z_before:.6f} -> {float(_probe.qpos[2]):.6f}  (it has begun to fall)")

# %% [markdown]
# ## 4. Exercise 1 — `step_for(model, data, seconds)`
#
# Simulated time only moves in whole timesteps, so a request for an arbitrary number of
# seconds cannot be honoured exactly. This lesson's contract is **never undershoot**: keep
# stepping until at least `seconds` of simulated time has elapsed, then stop. The answer is
# therefore a *ceiling*, not a truncation, and you must find it by stepping and watching
# `data.time` — not by assigning to the clock.

# %%
def step_for(model, data, seconds: float) -> int:
    """Step `data` until at least `seconds` of simulated time has passed; return step count.

    Measure elapsed time from `data.time` as it is when you are called, so the function also
    works on a `data` that has already been stepped. Use `TIME_EPS` as the slack in your
    comparison, so float drift in the accumulated clock cannot buy an extra step.

    Example (this model's timestep is 0.005 s, so 0.007 s is 1.4 timesteps, rounded up to 2):
        >>> model, data = load_humanoid()
        >>> step_for(model, data, 0.007)
        2
        >>> round(data.time, 3)
        0.01

    Returns:
        The number of mujoco.mj_step calls made. Zero is a valid answer for seconds <= 0.
    """
    # YOUR CODE HERE
    raise NotImplementedError


# Public check — run it as often as you like.
def _check_step_for() -> None:
    model, data = load_humanoid()
    n = step_for(model, data, 1.0)
    expected = math.ceil(1.0 / model.opt.timestep - TIME_EPS)
    assert n == expected, (
        f"step_for(1.0) returned {n}, expected {expected} — divide the request by "
        "model.opt.timestep; do not assume any particular timestep value."
    )
    assert abs(data.time - n * model.opt.timestep) < 1e-6, (
        f"data.time is {data.time} after {n} steps — you counted steps without calling "
        "mujoco.mj_step, so no physics actually happened."
    )

    model2, data2 = load_humanoid()
    partial = step_for(model2, data2, 0.007)
    assert partial == 2, (
        f"step_for(0.007) returned {partial}, expected 2 — int() and // truncate, and the "
        "contract is to never undershoot, so the step count rounds up."
    )

    model3, data3 = load_humanoid()
    assert step_for(model3, data3, 0.0) == 0, (
        "step_for(model, data, 0.0) must take no steps — your loop runs at least once, so "
        "make it a while loop that tests before it steps."
    )

    model4, data4 = load_humanoid()
    first = step_for(model4, data4, 0.5)
    second = step_for(model4, data4, 0.5)
    assert first == second, (
        f"stepping 0.5 s twice gave {first} then {second} — you compared data.time against "
        "`seconds` absolutely instead of against the time when the call started."
    )
    print(f"exercise 1 looks right: {n} steps buy {data.time:.3f} s of simulated time")


# %% [markdown]
# ## 5. Exercise 2 — `com_height(model, data)`
#
# `data.qpos[2]` is the height of the *root body* — the torso the free joint attaches to. The
# whole-body centre of mass is the mass-weighted average over every body: a different number,
# and the one that tells you whether a humanoid has fallen.
#
# Two ingredients: `model.body_mass` (constant, so it lives in the model) and `data.xipos`
# (each body's centre-of-mass position in world coordinates, so it lives in the data). Note
# `xipos`, not `xpos`: `xpos` is the body *frame origin*, which is not where its mass sits.

# %%
def com_height(model, data) -> float:
    """Return the z coordinate of the whole-body centre of mass, in metres.

    Mass-weighted mean over all bodies: sum(mass_i * z_i) / sum(mass_i), where z_i is
    `data.xipos[i][2]`. The world body carries zero mass, so including it changes nothing.

    `data.xipos` is only valid after MuJoCo has run kinematics — after an `mj_step`, an
    `mj_forward`, or a keyframe reset followed by `mj_forward`. This function does not
    refresh it for you; the caller does.

    Example (the default pose of this model stands taller at the root than at the COM):
        >>> model, data = load_humanoid()
        >>> mujoco.mj_forward(model, data)
        >>> com_height(model, data) < float(data.qpos[2])
        True

    Returns:
        A single float, not an array.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_com_height() -> None:
    model, data = load_humanoid()
    mujoco.mj_forward(model, data)
    got = com_height(model, data)
    assert isinstance(got, float), (
        f"com_height returned {type(got).__name__} — index the z component and cast with "
        "float(); do not return the whole 3-vector."
    )
    reference = float(data.subtree_com[0][2])
    assert abs(got - reference) < 1e-9, (
        f"com_height gave {got:.6f}, MuJoCo's own whole-body COM is {reference:.6f} — if you "
        "are close but high you used data.xpos (frame origins) instead of data.xipos; if you "
        f"got {float(data.qpos[2]):.6f} you returned the root height; if you landed between "
        "the two you took an unweighted mean instead of weighting by model.body_mass."
    )
    squat = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, squat, 0)
    mujoco.mj_forward(model, squat)
    crouched = com_height(model, squat)
    assert crouched < got - 0.1, (
        f"the crouched keyframe gave {crouched:.6f} against {got:.6f} standing — a constant, "
        "or anything computed from the model alone, cannot respond to a pose change; the "
        "pose lives in data.xipos."
    )
    print(f"exercise 2 looks right: standing COM {got:.3f} m, crouched {crouched:.3f} m")


# %% [markdown]
# ## 6. Exercise 3 — `real_time_factor(model, data, seconds)`
#
# Simulated time is bookkeeping; wall-clock time is what you actually wait. Their ratio is
# the real-time factor: above 1 the simulation outruns reality, below 1 it lags. It is a
# property of *this machine plus this model*, so it is measured, never assumed.
#
# Reset `data` with `mujoco.mj_resetData` first so every measurement starts from the same
# pose, time the stepping with `time.perf_counter` (not `time.time`, a wall clock subject to
# adjustment mid-measurement), and reuse your own `step_for`.

# %%
def real_time_factor(model, data, seconds: float) -> dict:
    """Measure how much faster than reality this machine simulates `seconds` of motion.

    In order: reset `data`, read `time.perf_counter()`, call your `step_for`, read the
    counter again.

    Example:
        >>> model, data = load_humanoid()
        >>> report = real_time_factor(model, data, 0.05)
        >>> sorted(report)
        ['real_time_factor', 'sim_seconds', 'steps', 'wall_seconds']
        >>> report["sim_seconds"] == report["steps"] * model.opt.timestep
        True

    Returns:
        dict with exactly these four keys:
          "steps"            int, from step_for
          "sim_seconds"      float, steps * model.opt.timestep
          "wall_seconds"     float, measured elapsed real time
          "real_time_factor" float, sim_seconds / wall_seconds
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_real_time_factor() -> None:
    model, data = load_humanoid()
    # 0.507 s is deliberately not a whole number of timesteps: on an exact multiple,
    # steps * timestep equals the seconds you were handed, and returning the argument
    # instead of the measured step count would slip through unnoticed.
    report = real_time_factor(model, data, 0.507)
    expected_keys = {"steps", "sim_seconds", "wall_seconds", "real_time_factor"}
    assert set(report) == expected_keys, (
        f"keys were {sorted(report)}, expected {sorted(expected_keys)} — return a dict with "
        "exactly those four names, spelled as in the docstring."
    )
    assert report["wall_seconds"] > 0.0, (
        "wall_seconds is not positive — you read perf_counter twice before stepping, or "
        "subtracted the two readings the wrong way round."
    )
    assert abs(report["sim_seconds"] - report["steps"] * model.opt.timestep) < 1e-9, (
        "sim_seconds does not equal steps * timestep — derive it from the step count you "
        "actually took, not from the `seconds` argument."
    )
    ratio = report["sim_seconds"] / report["wall_seconds"]
    assert abs(report["real_time_factor"] - ratio) < 1e-6, (
        f"real_time_factor is {report['real_time_factor']:.4f} but sim/wall is {ratio:.4f} — "
        "the ratio is simulated over wall; inverting it reports a fraction near zero."
    )
    assert report["real_time_factor"] > 1.0, (
        f"real_time_factor came out at {report['real_time_factor']:.4f} — a value that small "
        "means the ratio is upside down, or that you timed a reload instead of the stepping."
    )
    print(f"exercise 3 looks right: {report['sim_seconds']:.3f} s simulated in "
          f"{report['wall_seconds']:.4f} s wall, RTF {report['real_time_factor']:.1f}x")


# %% [markdown]
# ## 7. The payoff: a fall, measured
#
# With all three instruments working, this cell watches the humanoid collapse under gravity
# with no control at all, sampling the centre of mass on a fixed simulated-time grid. Every
# number and every bar below is produced by the functions you just wrote.

# %%
def fall_profile(samples: int = 10, interval: float = 0.2) -> list:
    """Return [(simulated time, COM height)] sampled every `interval` simulated seconds."""
    model, data = load_humanoid()
    mujoco.mj_forward(model, data)
    trace = [(data.time, com_height(model, data))]
    for _ in range(samples):
        step_for(model, data, interval)
        trace.append((data.time, com_height(model, data)))
    return trace


def _check_fall_profile() -> None:
    trace = fall_profile()
    start, end = trace[0][1], trace[-1][1]
    print(f"{'sim t (s)':>10s} {'COM z (m)':>10s}  profile")
    for t, z in trace:
        print(f"{t:10.2f} {z:10.4f}  {'#' * max(int(round(z * 40)), 0)}")
    assert end < start, (
        f"COM went from {start:.3f} to {end:.3f} — an uncontrolled humanoid under gravity "
        "must end lower than it started; check that step_for really calls mj_step."
    )
    print(f"\nfell {start - end:.3f} m of centre-of-mass height in {trace[-1][0]:.1f} s of "
          "simulated time, with zero control applied")


# %% [markdown]
# ## 8. Common mistakes
#
# - **Assigning to `data.time`.** It moves the clock and simulates nothing. Only `mj_step`
#   advances state; the clock is a consequence, not a control.
# - **`int(seconds / timestep)`.** Truncation undershoots the request. 1.4 timesteps of work
#   is 2 steps, and the last one overshoots — which is the honest answer.
# - **Comparing `data.time` against `seconds` absolutely.** It works once, then reports zero
#   steps forever, because the clock already exceeds the target. Measure from the time at
#   which you were called.
# - **Hard-coding the timestep.** This model overrides MuJoCo's documented default (sourced
#   in `claims.yaml`). Read `model.opt.timestep` and your code survives the next model.
# - **`data.xpos` for the centre of mass.** `xpos` is the body frame origin; `xipos` is where
#   that body's mass actually sits. The gap is centimetres, which is plenty to be wrong by.
# - **Forgetting kinematics.** Writing `data.qpos` does not update `data.xipos`. Until
#   `mj_forward` or `mj_step` runs, you are reading the previous pose.
# - **`time.time` for benchmarking.** Use `time.perf_counter`: monotonic, higher resolution,
#   and immune to a clock adjustment landing inside your measurement.

# %%
# Watch the last two mistakes happen, measured rather than asserted. Nothing here is typed.
_stale = mujoco.MjData(MODEL)
mujoco.mj_forward(MODEL, _stale)
_torso_before = float(_stale.xipos[1][2])
_stale.qpos[2] += 1.0                    # lift the robot a metre, then read derived state at once
print(f"wrote qpos[2] += 1.0 -> torso xipos z still reads {float(_stale.xipos[1][2]):.4f} "
      f"(it was {_torso_before:.4f})")
mujoco.mj_forward(MODEL, _stale)         # the one line that makes derived state true again
print(f"after mujoco.mj_forward it reads            {float(_stale.xipos[1][2]):.4f}  "
      f"(+{float(_stale.xipos[1][2]) - _torso_before:.4f} m, the lift you asked for)")
print(f"\nperf_counter monotonic={time.get_clock_info('perf_counter').monotonic}, "
      f"time() monotonic={time.get_clock_info('time').monotonic}"
      "  <- only one of these cannot go backwards mid-measurement")

# %% [markdown]
# ## 9. Self-check
#
# 1. You need the mass of the left foot and the current velocity of the left knee. Where do
#    they live?
#    - (a) both in `mjModel`
#    - (b) both in `mjData`
#    - (c) mass in `mjModel`, velocity in `mjData`
#    - (d) mass in `mjData`, velocity in `mjModel`
#
# 2. This model's `nq` exceeds its `nv` by exactly one. Why?
#    - (a) one actuator is unactuated
#    - (b) the free joint stores orientation as a 4-number quaternion but only a 3-number
#          angular velocity
#    - (c) MuJoCo reserves a slot in `qpos` for simulated time
#    - (d) the world body contributes a position but no velocity
#
# 3. `real_time_factor` reports a value far above 1 on your laptop. What follows?
#    - (a) the simulation is running too fast and is therefore inaccurate
#    - (b) the integrator is silently skipping timesteps to keep up
#    - (c) this model is cheap enough that a second of simulated motion costs far less than a
#          second of your time; accuracy is a separate question, set by the timestep
#    - (d) the timestep must be smaller than the documented default
#
# 4. You set `data.qpos[2] = 1.5` and immediately call `com_height(model, data)`. It returns
#    the value from before the assignment. Why?
#    - (a) `qpos` is read-only
#    - (b) `com_height` caches its result
#    - (c) `data.xipos` is a derived quantity, and nothing has recomputed it since you wrote
#          to `qpos`
#    - (d) the free joint ignores its z coordinate
#
# Answers, with reasoning, are published in the course solution bundle. Mark yourself first
# with the cell below, which tells you whether a letter is right without handing it to you.

# %%
# Put your four letters here and run the cell. It marks them without revealing the answer:
# a wrong letter sends you back to the section that measured it, which is the point.
SELF_CHECK = {1: "?", 2: "?", 3: "?", 4: "?"}

_ANSWER_DIGESTS = {1: "d260b0fbba30524f", 2: "0c940df6b5fea42b",
                   3: "9230a39fdad2bf8e", 4: "f22b9958b9c74bec"}
_ANSWER_SECTIONS = {
    1: "section 1 — which of the two objects changed while the other one stayed put",
    2: "section 2 — the joint table you printed, and the free joint's 7-number qpos block",
    3: "section 6 — what your own real_time_factor measured, and what it says nothing about",
    4: "section 8 — the stale-xipos demonstration you just ran",
}


def _check_self_check(answers: dict = None) -> None:
    """Mark the four multiple-choice answers in SELF_CHECK, naming where to look again."""
    answers = SELF_CHECK if answers is None else answers
    wrong = []
    for q, digest in sorted(_ANSWER_DIGESTS.items()):
        got = str(answers.get(q, "?")).strip().lower()
        if hashlib.sha256(f"F15-L01-q{q}-{got}".encode()).hexdigest()[:16] != digest:
            wrong.append(q)
    for q in sorted(_ANSWER_DIGESTS):
        note = f"  -> re-read {_ANSWER_SECTIONS[q]}" if q in wrong else ""
        print(f"  q{q}: {'wrong' if q in wrong else 'right'}{note}")
    assert not wrong, (
        f"questions {wrong} are still wrong. Each one names the section that answers it "
        "above — go back to the cell you ran there rather than guessing another letter."
    )
    print("self-check: all four right")


# %% [markdown]
# ## What you built, and where it goes next
#
# Three instruments — a clock you can trust, a posture sensor that reports the body rather
# than the torso, and an honest stopwatch — plus the habit of reading every constant out of
# `mjModel` instead of typing it. The rest of the Humanoid Lab flagship drives this same
# model with actuators and a controller, and measures success with the centre-of-mass height
# you just implemented.

# %%
if __name__ == "__main__":
    _check_step_for()
    _check_com_height()
    _check_real_time_factor()
    _check_fall_profile()
    _check_self_check()
