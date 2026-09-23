# Asset provenance

## `planar_walker.xml`

| | |
|---|---|
| Source | Authored for this lesson by the Synapsa Commons project. Not a download. |
| Licence | CC0-1.0 — <https://creativecommons.org/publicdomain/zero/1.0/> |
| Legal code | <https://creativecommons.org/publicdomain/zero/1.0/legalcode> (HTTP 200, checked 2026-09-16) |
| Retrieved | 2026-09-16 (written, not fetched) |
| Size | 4,444 bytes |
| Gated? | No. Nothing to register for, nothing to agree to, no network access required. |

A seven-degree-of-freedom planar biped: a torso on three root degrees of freedom (fore-aft
slide, vertical slide, pitch hinge) and two legs of hip-and-knee hinges. Every collision and
visual shape is a primitive capsule or an infinite plane. There is no `<mesh>`, no `file=`
attribute and no `<include>` anywhere in the file, so nothing else has to be downloaded for it
to compile. `lesson.py` resolves it through `walker_xml_path()`, which searches the lesson
directory and its parent and never reaches for the network. There is no download branch to
fail.

### Why a planar biped rather than a 3D humanoid

This lesson is about rhythm, phase and energy bookkeeping, and it pays for those with
rollouts: a gait search is worthless if it evaluates one candidate. Confining the machine to
the x-z plane means a student can reason about the whole gait from four numbers per step and
still run a hundred-odd rollouts inside a few seconds of laptop CPU.

Measured on the authoring machine with MuJoCo 3.13.0: **6.4 µs per `mj_step`**, about 157,000
steps per second, so the lesson's 2.4-second horizon costs roughly 5 ms per rollout. The
whole lesson — every demonstration plus a 47-rollout coordinate search — runs in about a
second. That headroom is the reason the search is a real search and not a description of one.

### The root joints are the instrument

The torso body sits at the origin, so a slide joint's coordinate **is** a world coordinate.
`qpos[0]` is the hip's fore-aft position in metres and `qpos[1]` is the hip height in metres.
Distance walked is therefore read directly out of `qpos[0]` rather than reconstructed from a
body transform, which keeps the cost-of-transport arithmetic legible: the student can see the
numerator and the denominator on the same screen.

### The actuators are position servos, and that is deliberate

All four actuators are `<position>` elements, so `data.ctrl[i]` is a **target angle in
radians**, not a torque. MuJoCo expands a `position` actuator to `gaintype="fixed"` with
`gainprm="kp 0 0"` and `biastype="affine"` with `biasprm="0 -kp -kv"` — the servo law is in
the engine, not in the notebook. That is what makes an open-loop gait possible at all: the
notebook supplies a rhythm of setpoints and the servo does the tracking.

It also sets up the lesson's central measurement. The torque that servo actually produced is
`data.actuator_force`, and the speed the joint actually moved at is `data.actuator_velocity`.
Their product is mechanical power, and its integral is the energy that the cost of transport
divides by weight and distance. None of those three quantities is typed into the notebook;
all three are read back out of `mjData`.

### The keyframe is a mid-stride pose, not a standing one

`qpos="0 0.80 0 0.30 -0.05 -0.30 -0.35"` puts one leg forward and one trailing with a bent
knee, with the leading foot a few millimetres above the floor at a hip height of 0.80 m. A
gait that starts from a symmetric standing pose has to break symmetry before it can walk,
which wastes a large part of a short horizon and makes the search surface much flatter. This
calibration was chosen while the model was written, and it is why the file carries a keyframe
at all.

### Friction and torque limits are load-bearing

`friction="1.0 0.05 0.05"` on both the feet and the floor, with a `forcerange` of ±150 N·m and
`kp="180"`, `kv="8"` on every servo. Weaken the friction and no gait finds purchase, so every
candidate scores the same and the search measures nothing. Raise the torque limit far enough
and the servo brute-forces any rhythm, so the energy term stops discriminating between gaits
and the cost of transport goes flat. Both failure modes were seen while the model was being
calibrated, which is why these numbers are commented in the XML rather than merely present.
