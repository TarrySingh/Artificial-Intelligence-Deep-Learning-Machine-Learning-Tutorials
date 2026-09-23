# Asset provenance

## `stander.xml`

| | |
|---|---|
| Source | Authored for this lesson by the Synapsa Commons project. Not a download. |
| Licence | CC0-1.0 — <https://creativecommons.org/publicdomain/zero/1.0/> |
| Retrieved | 2026-09-16 (written, not fetched) |
| Gated? | No. Nothing to register for, nothing to agree to, no network access required. |

A planar ankle-balancer: a box foot on a plane, three unactuated joints giving the foot the
freedom of a planar rigid body, and one actuated ankle carrying a capsule torso. The file is
the complete model — no `<mesh>`, no `file=` attribute, no `<include>`, no procedural asset
that has to be downloaded. `lesson.py` resolves it through `stander_xml_path()`, which
searches the lesson directory and its parent and never reaches for the network. There is no
download branch to fail.

### Why this model rather than F15-L08's balancer

F15-L08 disables contacts and pins the ankle, because its subject is the reality gap and it
needs thousands of cheap rollouts. This lesson's subject is the **support polygon**, which
does not exist without contacts: the polygon is the patch of ground the machine can press on,
and the whole lesson is about what happens when the centre of mass leaves it. So contacts stay
on, the foot is a real box on a real plane, and nothing pins it — it can tip, skid or leave the
ground, and the notebook measures which of those actually happens.

The simplification is spent elsewhere: one leg, one actuated joint, motion in the x-z plane.
A flat box on a plane settles onto four corner contacts, so the support polygon is a rectangle
the notebook computes from the foot geom rather than a number anyone typed.

### The foot is deliberately not centred on the ankle

The footprint runs from −0.08 m (heel) to +0.16 m (toe) in the foot frame, so the toe margin is
about twice the heel margin. This is the model's load-bearing asymmetry, and it is what makes
section 10 worth running: the notebook measures the largest push each of three controllers
survives forwards and backwards, and the two directions behave completely differently. Forwards,
where there is room, the controllers separate. Backwards, where there is not, all three stop at
the same rung however they are built.

That asymmetry is the result the lesson is built on — **not** a numerical agreement between the
push ratio and the margin ratio. The notebook does not claim one and does not compute one: the
ladder is coarse, and the three controllers' forward-to-backward ratios straddle the margin
ratio rather than reproducing it. What the geometry predicts here is the *shape* of the result —
that control quality cashes in on one side and not the other — and that is what gets measured.

Making the foot symmetric would delete that result and cost the lesson its sharpest measurement.

### The torso does not collide, on purpose

`torso_geom` carries `contype="0" conaffinity="0"`. Two consequences, both declared in the
notebook rather than hidden:

1. The support polygon is exactly the footprint, so it can be computed from one geom.
2. A fallen machine passes through the floor instead of landing on it, so a fall must be
   detected from **state** — which is one of the lesson's exercises, and a better habit than
   waiting for a collision that a real robot only gets to experience once.

The first draft omitted the flag. The torso capsule's lower cap reaches 0.01 m below the ankle,
so it rested on the plane and carried the entire body weight on one contact point while the
foot floated a centimetre in the air. The lesson's whole premise — that the foot's footprint is
the support polygon — was quietly false until that was found by printing `ncon` and the contact
positions instead of assuming them.

### The torque limit is not the real limit

`ctrlrange` is ±60 N·m. The machine weighs ≈19.5 kg and therefore presses on the floor with
≈191 N; because ground pressure can only act inside the footprint, the centre of pressure can
be shifted at most ≈0.083 m heelward or ≈0.157 m toeward of the mass, worth roughly 16 N·m and
30 N·m of ankle torque. Beyond that the foot rotates off the floor instead of the body rotating
upright.

That gap — a motor offering 60 N·m to a foot that will accept a small fraction of it — is the
section the model exists to make measurable. The notebook sweeps a constant ankle torque and
watches where the measured centre of pressure stops moving, rather than repeating the arithmetic
above, and it comes back with a ceiling *below* the 16 N·m that arithmetic predicts. The
prediction assumes the mass holds still; the body actually leans as the torque goes on, carrying
the centre of mass toward the edge it is about to tip over. The measured number is the one the
lesson uses, and the shortfall against the prediction is itself one of the things the student is
asked to explain.
