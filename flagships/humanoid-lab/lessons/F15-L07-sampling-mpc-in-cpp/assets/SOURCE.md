# Asset provenance

## `cartpole.xml`

| | |
|---|---|
| Origin | Authored for this lesson. Not derived from, or copied out of, any third-party model. |
| Download | None. The file ships in this directory and is the only asset the lesson reads. |
| Format | MJCF, the MuJoCo XML schema documented at <https://mujoco.readthedocs.io/en/stable/XMLreference.html> |
| Engine licence | MuJoCo is Apache License, Version 2.0 — <https://raw.githubusercontent.com/google-deepmind/mujoco/main/LICENSE> |
| Gated? | No. Nothing is fetched, so there is nothing to register for, agree to, or pay for. |

### Why an authored model rather than a downloaded one

Gate 11 asks for data that is real, free and ungated, and gate 9 asks the lesson to run with
no network. The cheapest way to satisfy both at once is to need no data. A cart-pole is four
rigid-body parameters and one actuator; writing it out in full is shorter than the code that
would be needed to fetch someone else's copy and prove it had not changed.

The obvious alternative, `dm_control`'s suite cart-pole, pulls in the suite's shared
`common/` includes, so it is not self-contained and would have to be vendored with its
dependencies to run offline. MuJoCo's own `model/` tree ships no cart-pole.

### What the lesson depends on in this file

- **`<compiler angle="radian"/>`** — MJCF reads angle-valued attributes as degrees by
  default, but a hinge's `qpos` is always radians. Pinning the units means the keyframe can
  say `3.14159265358979` and the `hinge` joint reads back the same number.
- **`<keyframe name="hanging">`** — pole straight down, cart centred, at rest. Both the C++
  planner and the Python reference reset to this keyframe by index, so the two languages are
  measured from an identical initial state rather than from two hand-assigned poses that
  might differ in the last bit.
- **`timestep="0.01"`** — an override of MuJoCo's documented default of 0.002 s (see
  `claims.yaml`). The lesson reads `model->opt.timestep` everywhere rather than assuming it,
  which is the habit that survives the next model.
- **A ±2 m slider range** — long enough that the swing-up does not become a study of the
  rail limit. Earlier drafts used ±1.1 m; the cart reached the stop on most seeds and the
  planner's success rate became a property of the track rather than of the algorithm.
- **Contact disabled** — nothing in this task touches anything. The flag makes `mj_step`
  cheaper and the rollouts-per-second figure easier to reason about; re-enabling it changes
  no line of the student's code.
