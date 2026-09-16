#!/usr/bin/env python3
"""The F15 capstone grading harness. You are given it: grading against a script you cannot
run yourself would be a trick, not an assessment.

    .venv/bin/python flagships/humanoid-lab/capstone/measure_walk.py \
        --policy flagships/humanoid-lab/capstone/policy.py --seed dev --episodes 24

Your policy module must expose:
    build_policy()                     -> called once, returns whatever your controller needs
    act(policy, observation, t)        -> returns shape (4,) joint targets

The observation is the 14 numbers of the planar walker, in this order:
    qpos[0:7]  hip x, hip height, torso pitch, hip_r, knee_r, hip_l, knee_l
    qvel[0:7]  the matching velocities

Fall criteria, distance and cost of transport are fixed here, not by you. NOTE on one
threshold: F15-L05's own gait search tolerated a torso pitch up to 1.0 rad, because it was
searching for any gait at all. The capstone is stricter and uses CAPSTONE.md's 0.6 rad, so a
gait that merely survived L05's search may be scored as a fall here. That is deliberate.
"""
import argparse
import importlib.util
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "assets" / "planar_walker.xml"

# Fixed by the harness (CAPSTONE.md), not by the submission.
MIN_HIP_HEIGHT = 0.55      # metres
MAX_PITCH = 0.6            # radians
DEFAULT_SECONDS = 8.0

# Domain-randomisation spec, identical to F15-L08's DR_SPEC. The draw ORDER is the contract:
# it is what makes a seed reproducible.
DR_SPEC = {
    "mass_scale": (0.85, 1.30),
    "frictionloss": (0.0, 15.0),
    "delay_steps": (0, 9),       # integers, upper bound exclusive
    "noise_std": (0.0, 0.03),
}
DEV_SEED = 707                   # the development seed students are given
NOMINAL = {"mass_scale": 1.0, "frictionloss": 0.0, "delay_steps": 0, "noise_std": 0.0, "seed": 0}


def load_model():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return model, mujoco.MjData(model)


def sample_conditions(n: int, seed: int, spec: dict = DR_SPEC) -> list:
    """Draw n conditions. Same generator and draw order as F15-L08, so seeds agree."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({
            "mass_scale": float(rng.uniform(*spec["mass_scale"])),
            "frictionloss": float(rng.uniform(*spec["frictionloss"])),
            "delay_steps": int(rng.integers(*spec["delay_steps"])),
            "noise_std": float(rng.uniform(*spec["noise_std"])),
            "seed": int(rng.integers(0, 10000)),
        })
    return out


def apply_condition(model, condition: dict, base_masses: np.ndarray) -> None:
    """Model edits: torso mass and joint dry friction, always from the baseline."""
    model.body_mass[:] = base_masses * condition["mass_scale"]
    model.dof_frictionloss[:] = condition["frictionloss"]


def observe(data) -> np.ndarray:
    return np.concatenate([data.qpos[:7], data.qvel[:7]]).astype(float)


def run_episode(model, data, policy_module, policy, condition: dict, seconds: float) -> dict:
    """One episode under one condition. Returns the per-episode scorecard."""
    dt = float(model.opt.timestep)
    horizon = int(round(seconds / dt))
    rng = np.random.default_rng(condition["seed"])

    try:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    except Exception:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    start_x = float(data.qpos[0])
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    span = np.maximum(hi - lo, 1e-9)

    # Latency: the policy sees an observation `delay_steps` old.
    delay = max(int(condition["delay_steps"]), 0)
    history = deque([observe(data)] * (delay + 1), maxlen=delay + 1)

    energy = 0.0
    saturated = 0
    fell_at_step = None

    for step in range(horizon):
        obs = history[0].copy()
        if condition["noise_std"] > 0:
            obs += rng.normal(0.0, condition["noise_std"], size=obs.shape)

        ctrl = np.asarray(policy_module.act(policy, obs, step * dt), dtype=float).reshape(-1)
        if ctrl.shape[0] != model.nu:
            raise ValueError(f"act() returned shape {ctrl.shape}, expected ({model.nu},)")

        # A submission may return out of range; MuJoCo clamps, and we record that it happened.
        clamped = np.clip(ctrl, lo, hi)
        if np.any(np.abs(ctrl - clamped) > 1e-9 * span):
            saturated += 1
        data.ctrl[:] = clamped

        mujoco.mj_step(model, data)
        energy += float(np.abs(data.actuator_force * data.actuator_velocity).sum()) * dt
        history.append(observe(data))

        if data.qpos[1] < MIN_HIP_HEIGHT or abs(data.qpos[2]) > MAX_PITCH:
            fell_at_step = step
            break

    steps_run = (fell_at_step + 1) if fell_at_step is not None else horizon
    distance = float(data.qpos[0]) - start_x
    survived = fell_at_step is None
    total_mass = float(model.body_mass.sum())
    gravity = float(-model.opt.gravity[2])
    cot = (energy / (total_mass * gravity * distance)) if distance > 0 else float("inf")

    return {
        "distance_m": distance,
        "survival_rate": 1.0 if survived else 0.0,
        "cost_of_transport": cot,
        "saturated_fraction": saturated / steps_run if steps_run else 0.0,
        "fell_at_step": fell_at_step,
    }


def measure_walk(policy_module, conditions: list, seconds: float = DEFAULT_SECONDS) -> dict:
    """Run one episode per condition and return the aggregate scorecard.

    Returns a dict with exactly these keys:
      "distance_m"          mean forward hip travel over SURVIVING episodes
      "survival_rate"       fraction of episodes that never fell
      "cost_of_transport"   mean dimensionless CoT over surviving episodes
      "saturated_fraction"  mean fraction of steps against a control limit
      "episodes"            per-condition dicts, same keys plus "fell_at_step"
    """
    if not conditions:
        raise ValueError("no conditions given")
    model, data = load_model()
    base_masses = model.body_mass.copy()
    policy = policy_module.build_policy()

    episodes = []
    for condition in conditions:
        apply_condition(model, condition, base_masses)
        episodes.append(run_episode(model, data, policy_module, policy, condition, seconds))

    survivors = [e for e in episodes if e["survival_rate"] == 1.0]
    finite = [e["cost_of_transport"] for e in survivors if np.isfinite(e["cost_of_transport"])]
    return {
        "distance_m": float(np.mean([e["distance_m"] for e in survivors])) if survivors else 0.0,
        "survival_rate": len(survivors) / len(episodes),
        "cost_of_transport": float(np.mean(finite)) if finite else float("inf"),
        "saturated_fraction": float(np.mean([e["saturated_fraction"] for e in episodes])),
        "episodes": episodes,
    }


def load_policy_module(path: str):
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("submitted_policy", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submitted_policy"] = mod
    spec.loader.exec_module(mod)
    for fn in ("build_policy", "act"):
        if not hasattr(mod, fn):
            raise AttributeError(f"{p.name} does not define {fn}()")
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade a walking policy.")
    ap.add_argument("--policy", required=True, help="path to your policy module")
    ap.add_argument("--seed", default="dev", help="'dev' for the development seed, or an integer")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    a = ap.parse_args()

    seed = DEV_SEED if a.seed == "dev" else int(a.seed)
    conditions = sample_conditions(a.episodes, seed)
    report = measure_walk(load_policy_module(a.policy), conditions, seconds=a.seconds)

    print(f"\n  conditions      {a.episodes} drawn from seed {seed} ({a.seed})")
    print(f"  distance_m         {report['distance_m']:.3f}   (mean over survivors)")
    print(f"  survival_rate      {report['survival_rate']:.3f}")
    cot = report["cost_of_transport"]
    print(f"  cost_of_transport  {cot:.3f}" if np.isfinite(cot) else "  cost_of_transport  inf")
    print(f"  saturated_fraction {report['saturated_fraction']:.3f}")
    fell = [e for e in report["episodes"] if e["fell_at_step"] is not None]
    print(f"  fell               {len(fell)}/{len(report['episodes'])} episodes\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
