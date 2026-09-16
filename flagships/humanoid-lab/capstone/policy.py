"""Your capstone submission starts here.

This starter is F15-L05's BASELINE_GAIT, unchanged. Measured under this capstone's harness,
not asserted:

    nominal model, 8 s      walks 1.208 m, then falls at step 978 (2.9 simulated seconds)
    24 development conditions   survives 12 of 24, mean 1.532 m over survivors, CoT 3.525

It is not a solved gait, and that is deliberate. F15-L05 validated it over 800 steps (2.4 s)
against a 1.0 rad pitch tolerance; the capstone runs 8 s and calls 0.6 rad a fall, so the
lesson's baseline is a genuine starting point here rather than an answer.

It is open-loop: `act` ignores the observation entirely and plays a fixed rhythm. That is its
weakness and your opportunity. A heavier torso, joint friction, control latency or sensor noise
will topple it, and the capstone scores you mostly on surviving conditions you never saw
(CAPSTONE.md section 3). Closing that gap means using the observation.

Run it:
    .venv/bin/python flagships/humanoid-lab/capstone/measure_walk.py \
        --policy flagships/humanoid-lab/capstone/policy.py --seed dev --episodes 24
"""
import math

import numpy as np

# The legs swing half a cycle apart. This is F15-L05's LEG_PHASE.
LEG_PHASE = math.pi


def build_policy():
    """Called once per grading run, before any episode.

    These are F15-L05's BASELINE_GAIT values. Tune them, replace them with a feedback
    controller, or carry a small learned table here — whatever you carry between steps must
    live inside this object.
    """
    return {
        "freq": 0.90,
        "hip_amp": 0.25,
        "knee_amp": 0.60,
        "knee_lag": 4.00,
        "hip_bias": 0.00,
    }


def act(policy, observation, t):
    """Return the four joint targets in actuator order: hip_r, knee_r, hip_l, knee_l.

    `observation` is the 14-number state (7 qpos then 7 qvel). This starter ignores it.
    `t` is elapsed SIMULATED seconds, so a periodic gait can use it directly as phase.
    """
    phase = 2.0 * math.pi * policy["freq"] * t
    hip_amp, knee_amp = policy["hip_amp"], policy["knee_amp"]
    knee_lag, hip_bias = policy["knee_lag"], policy["hip_bias"]

    hip_r = hip_amp * math.sin(phase) + hip_bias
    hip_l = hip_amp * math.sin(phase + LEG_PHASE) + hip_bias
    knee_r = -knee_amp * (1.0 - math.cos(phase + knee_lag)) / 2.0
    knee_l = -knee_amp * (1.0 - math.cos(phase + LEG_PHASE + knee_lag)) / 2.0
    return np.array([hip_r, knee_r, hip_l, knee_l])
