# Asset provenance

Two MJCF files, both authored for this lesson. Nothing in this directory was downloaded, so
there is nothing to register for, agree to, or pay for, and the lesson runs with the network
switched off (gates 9 and 11).

| | |
|---|---|
| Origin | Authored for this lesson. Not derived from, or copied out of, any third-party model. |
| Licence | CC0-1.0 — dedicated to the public domain by the Synapsa Commons project |
| Download | None. Both files ship in this directory and are the only assets the lesson reads. |
| Format | MJCF, the MuJoCo XML schema documented at <https://mujoco.readthedocs.io/en/stable/XMLreference.html> |
| Engine licence | MuJoCo itself is Apache License, Version 2.0 — <https://raw.githubusercontent.com/google-deepmind/mujoco/main/LICENSE> |
| Gated? | No. |

## `arm2.xml` — the arm the control loop drives

A shoulder and an elbow, two capsules, two direct-drive motors, pinned to the world. `nq`,
`nv` and `nu` are all 2. This is the model the C control loop steps, and the one both the C
and the Python loops are timed on.

What the lesson depends on in this file:

- **`<compiler angle="radian"/>`** — MJCF reads angle-valued attributes as degrees by
  default, but a hinge's `qpos` is always radians. Pinning the units means the joint ranges
  and the numbers the C loop reads out of `d->qpos` are in the same units.
- **`<key name="hanging">`** — both languages reset to this keyframe *by index*, so the C
  loop and the Python loop provably start from the same bits rather than from two
  hand-assigned poses. Without it the trajectory comparison would be measuring the pose
  assignment, not the loop.
- **Tight `ctrlrange`** — 8 N m at the shoulder and 3 at the elbow, against a gravitational
  load of a few N m and a PD law that asks for tens during the transient. This is a
  deliberate pedagogical choice: with a generous actuator the per-actuator clamp in
  `pd_ctrl` would never fire, and a student could delete it and still pass. The lesson
  measures the saturated fraction of the run, so the clamp has to be right.
- **`timestep="0.002"`** — MuJoCo's documented default, written out rather than left
  implicit (`claims.yaml`, `mujoco-default-timestep`). The C code reads `m->opt.timestep`
  anyway, which is the habit that survives the next model.
- **Contact disabled** — nothing in this task touches anything. A contact-free `mj_step` is
  the cheapest honest unit of work for the steps-per-second measurement. Re-enabling the
  flag changes no line of the student's code, only the throughput number.

## `arm2_floating.xml` — the model that proves the address arithmetic

The same arm with one line added: the base carries a `<freejoint/>`. It is never stepped.
The lesson loads it, reads addresses out of `mjModel`, and deletes it.

It exists because `arm2.xml` cannot, on its own, tell a right answer from a lucky one. With
two hinges and nothing else, `nq == nv == 2` and `qpos[i]` happens to pair with `qvel[i]`,
so code that indexes velocity with a position address is correct by accident. Add a free
joint and the coincidence ends: the free joint takes seven `qpos` entries (three of
translation plus a four-entry quaternion) but only six rows of `qvel`, so from the first
hinge onward the two addresses differ by one. Measured from the shipped file:

| | `nq` | `nv` | actuator 0 | actuator 1 |
|---|---|---|---|---|
| `arm2.xml` | 2 | 2 | qpos 0, dof 0 | qpos 1, dof 1 |
| `arm2_floating.xml` | 9 | 8 | qpos 7, dof 6 | qpos 8, dof 7 |

That `nq > nv` whenever a model contains ball or free joints is MuJoCo's own documented
behaviour, not a quirk of this file (`claims.yaml`, `mujoco-nq-exceeds-nv`). The numbers in
the table above are not typed into the lesson: the student's own `actuator_addresses` prints
them, and the autograder checks them against both models.
