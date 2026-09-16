# %% [markdown]
# # F15-L05 · Gait generation, and what a step actually costs
#
# **You will build:** an open-loop gait generator for a planar biped, an energy-accounting
# rollout, a cost-of-transport metric, and a budgeted coordinate search that finds a cheaper
# gait than the one you were given — and proves it by a measured margin.
#
# **Time:** ~50 minutes · **Runs on:** a laptop CPU, no GPU, no download
# · **Prerequisites:** F15-L01 (mjModel/mjData, stepping, reading qpos), F15-L03 (actuators,
# gains, and why a controller's authority is finite)
#
# By the end you will be able to:
# 1. Implement a rhythmic gait as a function of time, and say what each parameter does to it.
# 2. Measure the mechanical energy a gait spends, as the time integral of torque times
#    velocity read back out of `mjData`.
# 3. Implement the cost of transport and explain why it is dimensionless.
# 4. Run a budgeted search over gait parameters and report the margin it opens over a baseline.
#
# Every number this notebook prints is computed by the code you run. The only figures typed by
# a human are the published costs of transport in section 7, and each carries its source.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import hashlib
import math
import time
from pathlib import Path

import mujoco
import numpy as np

print("mujoco", mujoco.__version__, "· numpy", np.__version__)

MODEL_FILENAME = "planar_walker.xml"


def walker_xml_path() -> Path:
    """Locate the walker model that ships next to this notebook.

    There is no download branch. The model was written for this lesson and lives in
    `assets/`; if it is missing the checkout is broken, and saying so beats a silent fetch.
    """
    try:
        here = Path(__file__).resolve().parent
    except NameError:  # a notebook has no __file__
        here = Path.cwd()
    for candidate in (here / "assets" / MODEL_FILENAME,
                      here.parent / "assets" / MODEL_FILENAME,
                      Path.cwd() / "assets" / MODEL_FILENAME,
                      Path.cwd().parent / "assets" / MODEL_FILENAME):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{MODEL_FILENAME} not found next to this lesson. It ships in assets/ and is never "
        "downloaded; restore it from the lesson directory."
    )


def load_walker():
    """Compile the walker and hand back a fresh (model, data) pair."""
    model = mujoco.MjModel.from_xml_path(str(walker_xml_path()))
    return model, mujoco.MjData(model)


MODEL, DATA = load_walker()

# Read the physics out of the model rather than typing it. Gravity is a model property here,
# not a constant of nature you remembered.
DT = float(MODEL.opt.timestep)
TOTAL_MASS = float(MODEL.body_mass.sum())
GRAVITY = float(-MODEL.opt.gravity[2])

# The task: walk, for this long, without the hips dropping or the torso pitching over.
HORIZON_STEPS = 800
MIN_HIP_HEIGHT = 0.55       # metres; the keyframe stands at 0.80
MAX_PITCH = 1.0             # radians of torso pitch before we call it a fall
MIN_DISTANCE = 0.30         # metres; below this it did not walk, it shuffled
FALL_PENALTY = 100.0

print(f"model: nq={MODEL.nq} nv={MODEL.nv} nu={MODEL.nu} dt={DT} s")
print(f"total mass {TOTAL_MASS:.2f} kg, gravity {GRAVITY:.2f} m/s^2, "
      f"weight {TOTAL_MASS * GRAVITY:.1f} N")
print(f"horizon {HORIZON_STEPS} steps = {HORIZON_STEPS * DT:.1f} s of simulated time")

# %% [markdown]
# ## 1. The control input is an angle, not a torque
#
# Every actuator in this model is a MuJoCo `<position>` element, which the engine expands into
# a position servo. So `data.ctrl[i]` is a **target angle in radians** and the tracking law
# lives inside MuJoCo, not in your notebook. That is the whole reason an *open-loop* gait is
# possible: you supply a rhythm of setpoints and the servo chases them.
#
# Do not take that on trust — write a target and read back what the motor actually produced.

# %%
_m, _d = load_walker()
mujoco.mj_resetDataKeyframe(_m, _d, 0)
_d.ctrl[:] = [0.6, -0.8, -0.6, -0.2]      # four target ANGLES, in radians
for _ in range(50):
    mujoco.mj_step(_m, _d)
print(f"targets written to ctrl : {np.round(_d.ctrl, 3)}")
print(f"joint angles reached    : {np.round(_d.qpos[3:], 3)}   <- it is chasing, not arriving")
print(f"torque the servo made   : {np.round(_d.actuator_force, 2)} N*m")
print(f"joint speeds right now  : {np.round(_d.actuator_velocity, 3)} rad/s")
print("\nthe servo is a P-D law inside the engine; you never wrote a torque, and one exists")

# %% [markdown]
# ## 2. Exercise 1 — the gait, as a function of time
#
# A gait is a *rhythm*: each joint follows a periodic function of time, and what makes it a
# walk rather than a twitch is the **phase relationship between the joints**.
#
# Two offsets do all the work here:
#
# - **`leg_phase`** separates the two legs. At `pi` the legs are exactly antiphase — one
#   swings while the other stands. That is walking. At `0` both legs swing together, which is
#   hopping, and section 5 measures what happens when you try it.
# - **`knee_lag`** separates each knee from its own hip, so the knee flexes to clear the
#   ground at the right moment in the swing.
#
# This is the crudest possible **central pattern generator** — a rhythm source that runs
# open-loop, with no feedback from the body at all. Real CPGs are coupled oscillators; see
# Ijspeert's 2008 review in `claims.yaml` for those. A sinusoid is the version you can read.

# %%
PARAM_KEYS = ("freq", "hip_amp", "knee_amp", "knee_lag", "hip_bias")
LEG_PHASE = math.pi          # antiphase legs: the definition of a walk rather than a hop


def gait_targets(t: float, params: dict, leg_phase: float = LEG_PHASE):
    """The four joint targets at time `t`, in actuator order.

    Actuator order is fixed by the model: `(hip_r, knee_r, hip_l, knee_l)`.

    Let `phase = 2 * pi * params["freq"] * t`. Then:

        hip_r  =  hip_amp * sin(phase)                                  + hip_bias
        knee_r = -knee_amp * (1 - cos(phase + knee_lag)) / 2
        hip_l  =  hip_amp * sin(phase + leg_phase)                      + hip_bias
        knee_l = -knee_amp * (1 - cos(phase + leg_phase + knee_lag)) / 2

    Three things that are easy to get wrong, and all three are graded:

    - The knees carry a MINUS sign. This model's knee range is [-1.6, 0]: a knee flexes one
      way only, and a positive target is a knee bending backwards.
    - The `(1 - cos(.))` envelope runs from 0 to 2 on its own, so it is divided by 2 to span
      exactly `0` to `-knee_amp`.
    - `hip_bias` is a forward lean applied to the two HIPS only. It is not a knee parameter.

    Example (a quarter cycle in, where sin(pi/2) = 1):
        >>> p = {"freq": 1.0, "hip_amp": 0.4, "knee_amp": 0.0, "knee_lag": 0.0,
        ...      "hip_bias": 0.1}
        >>> [round(float(v), 3) for v in gait_targets(0.25, p)]
        [0.5, -0.0, -0.3, -0.0]

    Returns:
        A sequence of four floats — a numpy array of shape (4,) is ideal, because it can be
        written straight into `data.ctrl`.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_gait_targets() -> None:
    p = {"freq": 1.0, "hip_amp": 0.4, "knee_amp": 0.6, "knee_lag": 0.0, "hip_bias": 0.1}
    out = np.asarray(gait_targets(0.25, p), dtype=float)
    assert out.shape == (4,), (
        f"expected 4 targets in the order (hip_r, knee_r, hip_l, knee_l), got shape {out.shape}"
    )
    assert abs(out[0] - 0.5) < 1e-9, (
        f"right hip is {out[0]:.4f} at a quarter cycle; sin(pi/2) = 1, so it should be "
        "hip_amp + hip_bias = 0.5. Check the 2*pi in the phase."
    )
    assert abs(out[2] - (-0.3)) < 1e-9, (
        f"left hip is {out[2]:.4f}, expected -0.3. The legs are ANTIPHASE: add leg_phase to "
        "the left leg's phase. If you got 0.5 the two legs are moving together."
    )
    cycle = np.array([np.asarray(gait_targets(t, p), dtype=float)
                      for t in np.linspace(0.0, 1.0, 400)])
    assert cycle[:, [1, 3]].max() <= 1e-9, (
        f"a knee target reached {cycle[:, [1, 3]].max():+.4f}. Knees only flex on this model — "
        "every knee target must be <= 0, so the envelope is multiplied by MINUS knee_amp."
    )
    assert abs(cycle[:, [1, 3]].min() + p["knee_amp"]) < 1e-3, (
        f"the deepest knee flexion was {cycle[:, [1, 3]].min():+.4f}, but it should reach "
        f"-knee_amp = {-p['knee_amp']:.2f} once per cycle. Did you divide (1 - cos) by 2?"
    )
    print(f"exercise 1 looks right: hips span {cycle[:, 0].min():+.2f}..{cycle[:, 0].max():+.2f} "
          f"rad, knees {cycle[:, 1].min():+.2f}..{cycle[:, 1].max():+.2f} rad")


# %% [markdown]
# ## 3. Energy is the integral of torque times velocity
#
# Mechanical power at one actuator is `torque * angular velocity`, in watts. Energy is that
# power integrated over time, which on a fixed timestep is a sum multiplied by `dt`.
#
# MuJoCo hands you both halves, and the header calls them exactly what they are:
# `actuator_force` is the "actuator force in actuation space" and `actuator_velocity` the
# "actuator velocities, one per force output". You never differentiate anything yourself.
#
# Two decisions are worth naming out loud, because they change the number:
#
# - **Absolute value.** `|tau * omega|` charges the gait for braking as well as driving. A
#   real machine with no regenerative braking does pay for both, so this is the honest choice
#   here — but it is *stricter* than the "positive work only" convention used in the
#   literature, which matters in section 7 when we compare against published figures.
# - **`dt`.** Forget it and your energy is out by a factor of `1/dt`, which on this model is
#   over three hundred. It will still look like a plausible number.
#
# Run this. It computes the same energy twice, in actuation space and in joint space.

# %%
_m, _d = load_walker()
mujoco.mj_resetDataKeyframe(_m, _d, 0)
_e_actuation = _e_joint = 0.0
for _i in range(300):
    _d.ctrl[:] = [0.3 * math.sin(6.0 * _i * DT), -0.4, -0.3 * math.sin(6.0 * _i * DT), -0.4]
    mujoco.mj_step(_m, _d)
    _e_actuation += float(np.abs(_d.actuator_force * _d.actuator_velocity).sum()) * DT
    _e_joint += float(np.abs(_d.qfrc_actuator[3:] * _d.qvel[3:]).sum()) * DT
print(f"energy via actuator_force * actuator_velocity : {_e_actuation:8.3f} J")
print(f"energy via qfrc_actuator * qvel (joint space) : {_e_joint:8.3f} J")
print(f"relative difference between the two spaces    : "
      f"{abs(_e_actuation - _e_joint) / _e_actuation:.2e}")
print("\nsame physics, two coordinate systems. If your energy is 300x this, you dropped dt.")

# %% [markdown]
# ## 4. Exercise 2 — the rollout, and exercise 3 — the cost of transport
#
# The rollout is the instrument everything else stands on: it drives the gait for the horizon,
# accumulates energy, tracks how far the hips travelled, and stops the moment the walker falls.
#
# `qpos[0]` is the hip's fore-aft position **in metres** and `qpos[1]` is hip height, because
# the torso's root joints are slides at the origin. Distance is read, not reconstructed.

# %%
BASELINE_GAIT = {"freq": 0.90, "hip_amp": 0.25, "knee_amp": 0.60, "knee_lag": 4.00,
                 "hip_bias": 0.00}


def rollout(model, data, params: dict, horizon: int = HORIZON_STEPS,
            leg_phase: float = LEG_PHASE) -> dict:
    """Drive one gait for the horizon and report what it cost.

    The loop, in order, for each of `horizon` steps `i`:

    1. `mujoco.mj_resetDataKeyframe(model, data, 0)` and `mujoco.mj_forward(model, data)`
       BEFORE the loop, so every candidate starts from the same mid-stride pose. Record the
       starting `data.qpos[0]`.
    2. Write `gait_targets(i * DT, params, leg_phase)` into `data.ctrl[:]`.
    3. `mujoco.mj_step(model, data)`.
    4. Add `abs(data.actuator_force * data.actuator_velocity).sum() * DT` to the energy.
    5. Count the step, then stop early if `data.qpos[1] < MIN_HIP_HEIGHT` or
       `abs(data.qpos[2]) > MAX_PITCH` — mark it fallen.

    Distance is the CHANGE in `data.qpos[0]` from the recorded start, so a walker that goes
    backwards reports a negative distance rather than a small positive one.

    Example (the baseline gait, which is known to survive the full horizon):
        >>> model, data = load_walker()
        >>> r = rollout(model, data, BASELINE_GAIT)
        >>> r["fell"], r["steps"] == HORIZON_STEPS
        (False, True)

    Returns:
        dict with exactly these four keys:
          "distance"  float, metres travelled by the hips (may be negative)
          "energy"    float, joules of absolute mechanical work at the actuators
          "steps"     int, steps actually taken
          "fell"      bool, True if it dropped or pitched over
    """
    # YOUR CODE HERE
    raise NotImplementedError


def cost_of_transport(energy_j: float, distance_m: float, mass_kg: float = TOTAL_MASS,
                      gravity: float = GRAVITY) -> float:
    """The dimensionless specific cost of transport: energy per unit weight per unit distance.

        cost of transport = energy / (mass * gravity * distance)

    Energy is in joules, `mass * gravity` is a weight in newtons and distance is in metres, so
    joules over newton-metres cancels completely. That is the point: a dimensionless number
    compares a 20 kg simulated biped against a 70 kg human without either one being rescaled.

    A machine that spends energy and covers no ground has an INFINITE cost of transport, not
    a zero one. Return `float("inf")` for any distance that is not strictly positive — that
    includes a walker that went backwards — rather than dividing by zero.

    Example:
        >>> round(cost_of_transport(100.0, 2.0, mass_kg=10.0, gravity=9.81), 6)
        0.509684
        >>> cost_of_transport(100.0, 0.0, mass_kg=10.0, gravity=9.81)
        inf

    Returns:
        float, dimensionless.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def gait_score(result: dict) -> float:
    """Lower is better. Provided; it is the objective the search in section 6 minimises.

    A gait that survived and walked far enough scores its cost of transport. Anything else
    scores a penalty that still improves with distance and with steps survived, so the search
    has a gradient to follow even before any candidate walks.
    """
    if result["fell"] or result["distance"] < MIN_DISTANCE:
        return (FALL_PENALTY - 10.0 * result["distance"]
                - 10.0 * (result["steps"] / HORIZON_STEPS))
    return cost_of_transport(result["energy"], result["distance"])


def _check_rollout_and_cot() -> None:
    assert abs(cost_of_transport(100.0, 2.0, mass_kg=10.0, gravity=9.81) - 0.5096840) < 1e-6, (
        "cost_of_transport(100, 2, mass=10, g=9.81) should be 100/(10*9.81*2) = 0.509684; "
        "omitting gravity gives 5.0."
    )
    assert cost_of_transport(50.0, 0.0) == float("inf"), (
        "a distance of zero must give float('inf') — energy was spent and nothing moved."
    )
    assert cost_of_transport(50.0, -0.7) == float("inf"), (
        "a negative distance must also give float('inf'); walking backwards is not efficient "
        "transport, and a negative cost of transport would win every search you ever run."
    )
    model, data = load_walker()
    r = rollout(model, data, BASELINE_GAIT)
    assert set(r) == {"distance", "energy", "steps", "fell"}, (
        f"keys were {sorted(r)} — return a dict with exactly those four names."
    )
    assert not r["fell"] and r["steps"] == HORIZON_STEPS, (
        f"the baseline gait fell={r['fell']} after {r['steps']} steps. It is known to survive "
        "the full horizon, so the loop is wrong: the usual causes are writing targets into "
        "data.qpos instead of data.ctrl, or forgetting mj_step."
    )
    again = rollout(model, data, BASELINE_GAIT)
    assert abs(again["distance"] - r["distance"]) < 1e-9, (
        f"the same gait walked {r['distance']:.3f} m then {again['distance']:.3f} m on the same "
        "data — reset to the keyframe at the TOP of every rollout."
    )
    print(f"exercises 2 and 3 look right: the baseline walks {r['distance']:.3f} m on "
          f"{r['energy']:.1f} J, cost of transport "
          f"{cost_of_transport(r['energy'], r['distance']):.4f}")


# %% [markdown]
# ## 5. Phase is not a detail
#
# Before searching anything, look at what the one offset you have *not* tuned is worth. The
# cell below runs your own rollout three times, changing only `leg_phase`. Nothing else moves.

# %%
def _check_phase_matters() -> None:
    model, data = load_walker()
    print(f"  {'leg_phase':>12s} {'distance':>9s} {'energy':>9s} {'fell':>6s} {'steps':>6s}")
    out = {}
    for name, phase in (("pi (walk)", math.pi), ("pi/2", math.pi / 2), ("0 (hop)", 0.0)):
        r = rollout(model, data, BASELINE_GAIT, leg_phase=phase)
        out[name] = r
        print(f"  {name:>12s} {r['distance']:9.3f} {r['energy']:9.1f} "
              f"{str(r['fell']):>6s} {r['steps']:>6d}")
    # These are claims about this machine, so they are asserted rather than merely printed.
    assert not out["pi (walk)"]["fell"], "the antiphase gait is the one that should survive"
    assert out["0 (hop)"]["fell"], (
        "driving both legs in phase should topple this walker — if it survived on your "
        "machine, re-read the table above before trusting the sentence below it"
    )
    print("\n  Same joints, same amplitudes, same frequency, same energy budget. The only")
    print("  difference is WHEN each leg moves relative to the other, and it decides whether")
    print("  the machine walks or falls over. Phase is the gait.")


# %% [markdown]
# ## 6. Exercise 4 — a search you can afford
#
# Now make it cheaper. Coordinate search is the least clever thing that works: hold every
# parameter but one, try that one across a grid, keep any improvement, move to the next.
#
# The rule that matters is the **budget**. A rollout costs real milliseconds, and a search
# that ignores its cap is how a lesson stops fitting on a laptop. Yours takes `max_rollouts`
# and stops when it is spent — checked before every rollout, not after.

# %%
SEARCH_GRID = {
    "freq":     [0.6, 0.8, 1.0, 1.3, 1.7],
    "hip_amp":  [0.10, 0.20, 0.30, 0.45, 0.60],
    "knee_amp": [0.0, 0.25, 0.50, 0.75, 1.00],
    "knee_lag": [0.0, 1.26, 2.51, 3.77, 5.03],
    "hip_bias": [-0.10, -0.05, 0.0, 0.05, 0.10],
}
MAX_ROLLOUTS = 120


def coordinate_search(start: dict, grid: dict = SEARCH_GRID, passes: int = 2,
                      max_rollouts: int = MAX_ROLLOUTS):
    """Coordinate descent over gait parameters, minimising `gait_score`, under a hard cap.

    The algorithm, exactly:

    1. Evaluate `start` once. That is your incumbent, its score, and rollout number one.
    2. Repeat `passes` times: for each key in `PARAM_KEYS`, in that order, try every value in
       `grid[key]`, skipping the value the incumbent already holds.
    3. Each trial is the incumbent with that ONE key replaced. Score it with
       `gait_score(rollout(model, data, candidate))`.
    4. If it scores strictly lower, it becomes the incumbent immediately — so the next
       parameter is tuned against the improvement you just found, not against `start`.
    5. Before every rollout, stop and return if you have already used `max_rollouts`.

    Build the model once with `load_walker()` and reuse it; `rollout` resets it each time.

    Example (the shape of the return, not a particular gait):
        >>> best, score, used = coordinate_search(BASELINE_GAIT, passes=1)
        >>> sorted(best) == sorted(PARAM_KEYS), used <= MAX_ROLLOUTS
        (True, True)

    Returns:
        (best_params, best_score, rollouts_used) — a dict with the five PARAM_KEYS, the float
        score of that gait, and an int count of every rollout you ran including the first.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_search() -> None:
    model, data = load_walker()
    base = rollout(model, data, BASELINE_GAIT)
    base_score = gait_score(base)
    t0 = time.perf_counter()
    best, score, used = coordinate_search(BASELINE_GAIT, passes=2)
    elapsed = time.perf_counter() - t0
    assert sorted(best) == sorted(PARAM_KEYS), (
        f"the search returned keys {sorted(best)}, expected {sorted(PARAM_KEYS)}"
    )
    assert used <= MAX_ROLLOUTS, f"the search used {used} rollouts against a cap of {MAX_ROLLOUTS}"
    assert score <= base_score, (
        f"the search returned {score:.4f}, worse than the baseline's {base_score:.4f}. Keep the "
        "incumbent unless a candidate strictly improves on it."
    )
    _, _, capped = coordinate_search(BASELINE_GAIT, passes=4, max_rollouts=10)
    assert capped <= 10, (
        f"with max_rollouts=10 the search still ran {capped} rollouts — check the budget "
        "BEFORE each rollout, not after."
    )
    found = rollout(model, data, best)
    print(f"  {used} rollouts in {elapsed:.2f} s ({elapsed / used * 1000:.0f} ms each)\n")
    print(f"  {'':10s} {'distance':>9s} {'energy':>9s} {'cost of transport':>18s}")
    print(f"  {'baseline':10s} {base['distance']:9.3f} {base['energy']:9.1f} {base_score:18.4f}")
    print(f"  {'found':10s} {found['distance']:9.3f} {found['energy']:9.1f} {score:18.4f}")
    print(f"\n  gait found: { {k: round(best[k], 3) for k in PARAM_KEYS} }")
    print(f"  the search cut the cost of transport by a factor of {base_score / score:.2f}")
    assert score < base_score / 1.5, (
        f"only a {base_score / score:.2f}x improvement — the grid holds much better gaits than "
        "that, so check you carry each accepted improvement into the next parameter."
    )


# %% [markdown]
# ## 7. Against the literature, honestly
#
# Your number now has company. Collins, Ruina, Tedrake and Wisse (*Science*, 2005) define the
# dimensionless specific cost of transport exactly as you implemented it, and report figures
# for real machines and for people. Every value below is sourced in `claims.yaml`.
#
# One caveat decides whether the comparison is honest, so the table prints it: their `c_mt`
# counts only the **positive** mechanical work of the actuators, while your rollout charges
# for braking too. Yours is the stricter accounting. The gap is real; the ratio is not
# like-for-like, and a table that implied otherwise would be the sort of thing this course
# exists to stop you doing.

# %%
PUBLISHED_COT = [
    # what, c_et (total energy), c_mt (positive actuator work only)
    ("humans walking", 0.2, 0.05),
    ("Cornell biped (13 kg, 0.4 m/s)", 0.2, 0.055),
    ("Delft biped", None, 0.08),
    ("MIT biped", None, 0.02),
    ("Honda ASIMO (authors' estimate)", 3.2, 1.6),
]


def compare_against_published(measured_cot: float) -> None:
    """Print the measured cost of transport beside the published figures, with the caveat."""
    print(f"  {'machine':<34s} {'c_et':>6s} {'c_mt':>6s}")
    for name, c_et, c_mt in PUBLISHED_COT:
        et = "  -  " if c_et is None else f"{c_et:5.3g}"
        print(f"  {name:<34s} {et:>6s} {c_mt:6.3g}")
    print(f"  {'-' * 48}")
    print(f"  {'your gait (absolute work)':<34s} {'  -  ':>6s} {measured_cot:6.3g}")
    best_published = min(c_mt for _, _, c_mt in PUBLISHED_COT)
    print(f"\n  Your walker is {measured_cot / best_published:.0f}x the cost of the most "
          f"efficient machine in that table,")
    print("  and it is not even a like-for-like measurement: you charged yourself for braking")
    print("  and they did not. Efficient walking is hard, and the honest version of that")
    print("  sentence names the accounting difference instead of hiding inside it.")


# %% [markdown]
# ## 8. Common mistakes
#
# - **Optimising distance instead of cost.** The two are different objectives and they give
#   different gaits. A machine that sprints and burns everything is not efficient; a metric
#   that only counts metres cannot tell you so. The cell below measures both.
# - **Dropping `dt` from the energy sum.** Off by `1/dt` — over three hundred here — and still
#   a plausible-looking number, which is what makes it dangerous.
# - **Summing `|torque|` instead of `|torque * velocity|`.** That is not energy and not power.
#   A joint straining against a limit at zero speed does no mechanical work at all.
# - **Forgetting to reset.** Without a reset at the top of each rollout, every candidate in
#   your search starts wherever the previous one collapsed, and the search is scoring noise.
# - **Letting distance go negative unnoticed.** A backwards walker that spends energy would
#   score a *negative* cost of transport and win the search outright. Guard the denominator.
# - **Treating a surviving gait as a good one.** Survival is a floor, not a result.

# %%
def _check_objective_matters() -> None:
    """Search the same grid under two objectives at EQUAL budget, and print both gaits.

    Equal budget matters: give one objective more passes than the other and the table
    measures the budget rather than the objective, which is a mistake worth not making in
    public.
    """
    def distance_only(result: dict) -> float:
        # Same penalty SHAPE as gait_score, so the only difference is the objective itself.
        if result["fell"] or result["distance"] < MIN_DISTANCE:
            return (FALL_PENALTY - 10.0 * result["distance"]
                    - 10.0 * (result["steps"] / HORIZON_STEPS))
        return -result["distance"]

    model, data = load_walker()

    def search(objective):
        best, used = dict(BASELINE_GAIT), 1
        best_score = objective(rollout(model, data, best))
        for _ in range(2):
            for key in PARAM_KEYS:
                for value in SEARCH_GRID[key]:
                    if value == best[key] or used >= MAX_ROLLOUTS:
                        continue
                    candidate = dict(best, **{key: value})
                    score = objective(rollout(model, data, candidate))
                    used += 1
                    if score < best_score:
                        best_score, best = score, candidate
        return best

    far = rollout(model, data, search(distance_only))
    cheap = rollout(model, data, search(gait_score))
    print(f"  {'objective':<22s} {'distance':>9s} {'energy':>9s} {'cost of transport':>18s}")
    print(f"  {'maximise distance':<22s} {far['distance']:9.3f} {far['energy']:9.1f} "
          f"{gait_score(far):18.4f}")
    print(f"  {'minimise cost':<22s} {cheap['distance']:9.3f} {cheap['energy']:9.1f} "
          f"{gait_score(cheap):18.4f}")
    print("\n  Same grid, same baseline, same rollout budget. On this machine, chasing")
    print(f"  distance alone changed travel by {far['distance'] - cheap['distance']:+.3f} m "
          f"and burned {far['energy'] / cheap['energy']:.1f}x the energy,")
    print(f"  for a cost of transport {gait_score(far) / gait_score(cheap):.1f}x higher.")
    print("\n  The point is not that one number beat the other. It is that a distance")
    print("  objective cannot SEE energy, so it could never have told you what it cost you.")
    print("  Write down the metric you actually care about before you optimise anything.")


# %% [markdown]
# ## 9. Self-check
#
# 1. A gait burns 300 J and ends exactly where it started. Its cost of transport is:
#    - (a) zero — it did no useful work
#    - (b) 300 divided by the machine's weight
#    - (c) infinite — energy was spent and no ground was covered, which is exactly why the
#          function guards against a distance that is not strictly positive
#    - (d) undefined, so the rollout should raise an exception
#
# 2. You set `leg_phase` to 0, so both legs swing together. The walker covers a little ground
#    and then falls. Why?
#    - (a) nothing is wrong; the horizon is too short to show the gait working
#    - (b) with both legs in phase there is never a stance leg under the body — it hops and
#          pitches over instead of walking, which is what the antiphase offset prevents
#    - (c) the knees are at fault, not the phase
#    - (d) MuJoCo cannot simulate a hopping machine
#
# 3. Your rollout sums `abs(actuator_force)` each step instead of
#    `abs(actuator_force * actuator_velocity) * DT`. What have you measured?
#    - (a) energy, in different units
#    - (b) power, which over a fixed horizon is the same ranking anyway
#    - (c) neither — an accumulated force has the wrong dimensions entirely, and it rewards a
#          gait for straining hard while standing still
#    - (d) the right answer, scaled by the timestep
#
# 4. You measure a cost of transport near 0.6 and read that humans walk at `c_mt` near 0.05.
#    The honest comparison is:
#    - (a) your gait is exactly twelve times worse than a human
#    - (b) the two are the same shape but not the same accounting — you charged for negative
#          work and the published figure counts only positive actuator work, so the gap is
#          real while the precise ratio is not a like-for-like measurement
#    - (c) the numbers are unrelated and cannot be compared at all
#    - (d) the whole difference is explained by it being a simulator
#
# Answers, with reasoning, are published in the course solution bundle.

# %%
# Put your four letters here and run the cell. It marks them without revealing the answer:
# a wrong letter sends you back to the section that measured it, which is the point.
SELF_CHECK = {1: "?", 2: "?", 3: "?", 4: "?"}

_ANSWER_DIGESTS = {1: "0dcc785a7cf671de", 2: "b43f87a1f5173e77",
                   3: "a6d7100ef7fa03e4", 4: "a0f1eca7d1f50472"}
_ANSWER_SECTIONS = {
    1: "section 4 — the guard you wrote into cost_of_transport, and why it is not zero",
    2: "section 5 — the leg_phase table you ran, and which row fell over",
    3: "section 3 — the two-space energy cross-check, and what dt is doing in it",
    4: "section 7 — the caveat column of the comparison table you printed",
}


def _check_self_check(answers: dict = None) -> None:
    """Mark the four multiple-choice answers in SELF_CHECK, naming where to look again."""
    answers = SELF_CHECK if answers is None else answers
    wrong = []
    for q, digest in sorted(_ANSWER_DIGESTS.items()):
        got = str(answers.get(q, "?")).strip().lower()
        if hashlib.sha256(f"F15-L05-q{q}-{got}".encode()).hexdigest()[:16] != digest:
            wrong.append(q)
    for q in sorted(_ANSWER_DIGESTS):
        note = f"  -> re-read {_ANSWER_SECTIONS[q]}" if q in wrong else ""
        print(f"  q{q}: {'wrong' if q in wrong else 'right'}{note}")
    assert not wrong, (
        f"questions {wrong} are still wrong. Each one names the section that answers it above "
        "— go back to the measurement you ran there rather than guessing a letter."
    )
    print("self-check: all four right")


# %% [markdown]
# ## What you built, and where it goes next
#
# A gait generator whose parameters you can name, an energy account read straight out of the
# engine, a dimensionless cost of transport, and a budgeted search that beat its baseline by a
# margin you measured rather than hoped for.
#
# F15-L08 takes the rollout you wrote here as its unit of evaluation and asks a harder
# question: does any of it survive when the simulator stops telling the truth about mass,
# latency and friction? The capstone asks for a walk of a measured distance under randomised
# parameters, using exactly this gait-and-rollout contract.

# %%
if __name__ == "__main__":
    _check_gait_targets()
    _check_rollout_and_cot()
    _check_phase_matters()
    _check_search()
    _model, _data = load_walker()
    _best, _score, _ = coordinate_search(BASELINE_GAIT, passes=2)
    compare_against_published(_score)
    _check_objective_matters()
    _check_self_check()
