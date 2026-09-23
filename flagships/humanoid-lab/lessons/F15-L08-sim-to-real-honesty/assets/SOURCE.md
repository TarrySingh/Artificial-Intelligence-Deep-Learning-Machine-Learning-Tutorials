# Asset provenance

## `balancer.xml`

| | |
|---|---|
| Source | Authored for this lesson by the Synapsa Commons project. Not a download. |
| Licence | CC0-1.0 — <https://creativecommons.org/publicdomain/zero/1.0/> |
| Retrieved | 2026-09-16 (written, not fetched) |
| Gated? | No. Nothing to register for, nothing to agree to, no network access required. |

A planar ankle-and-hip balancer: two capsules, two hinge joints, two direct-torque motors.
The file is the complete model — no `<mesh>`, no `file=` attribute, no `<include>`, no
procedural asset that has to be downloaded. `lesson.py` resolves it through
`balancer_xml_path()`, which searches the lesson directory and its parent and never reaches
for the network at all. There is no download branch to fail.

### Why this model rather than a humanoid

This lesson runs four separate studies — control latency, actuator saturation, sensor noise,
and domain randomisation over mass and dry friction — and each needs hundreds of independent
rollouts before it can say anything honest. The model is therefore the smallest machine that
still has a humanoid's defining property: it is an inverted pendulum, and it falls over
unless something actively holds it up.

Contacts are disabled and the ankle is a pin joint rather than a foot resting on a floor.
That is a real simplification and the lesson says so out loud — it is named in section 6 as
one of the gaps the course cannot close, alongside the ones the student measures. The payoff
is a contact-free `mj_step` costing a few microseconds, which is what puts a full randomised
study inside a laptop CPU budget instead of inside a paragraph describing one.

### The calibration is load-bearing

The keyframe disturbance (0.15 rad of lean, 0.5 rad/s forward) and the ankle torque limit
(40 N·m) are chosen together, and the comment block in the XML records the arithmetic. Holding
this body statically at a lean of θ costs roughly 133·sin(θ) N·m at the ankle, so a 40 N·m
ankle stalls near 0.31 rad. Starting at 0.15 rad and still falling forwards puts recovery
squarely in the region where the torque limit is a live constraint.

That margin is the lesson. Make the disturbance much smaller and every reality gap becomes
survivable, so the student measures nothing. Make it much larger and nothing survives, so the
student measures nothing. Both failure modes were observed while the model was being written,
which is why the numbers are commented rather than merely present.
