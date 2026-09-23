# F15 · Humanoid Lab

A flagship about making a simulated humanoid do something, on hardware you already own, and
then being honest about the distance between that and a robot.

Every lesson is a notebook or a compiled program you run and fill in. There is no lesson that
is only reading. The flagship ends by measuring its own limits rather than claiming not to
have any.

## What you build

An instrument at a time. You start by learning to read state out of a physics engine instead
of assuming it, and you finish holding a bench that can inject control latency, actuator
saturation, sensor noise and parameter error into a controller and tell you, in numbers, which
one breaks it first.

The through-line is a habit, not a library: **measure the thing rather than repeat what is
said about it.** It applies to a centre of mass, to a real-time factor, to a control loop
written in C, and — in the last lesson — to a claim about what hardware a piece of NVIDIA
software requires.

## The lessons, in order

| # | id | what you do |
|---|---|---|
| 1 | `F15-L01-first-contact` | Load MuJoCo's humanoid, step its physics, and read centre-of-mass height and this machine's real-time factor out of state. |
| 2 | `F15-L02-kinematics-jacobians` | Build a translational Jacobian, verify it against finite differences of forward kinematics, and drive a 9-DoF upper body's palm onto a Cartesian target with damped least squares. |
| 3 | `F15-L03-pd-control` | State feedback and gravity compensation: recover the bias force from `qfrc_bias`, and find where a high gain stops being free. |
| 4 | `F15-L04` | **Standing balance under disturbance** — centre of mass, the support polygon, and why a static posture controller falls over. Apply an impulse, detect the fall, then close a CoM-feedback loop and measure the largest push it survives. |
| 5 | `F15-L05-gait-and-energy` | Generate a gait for a planar biped and account for its energy honestly, in cost-of-transport terms. |
| 6 | `F15-L06-control-loop-in-c` | Write the control loop in C against MuJoCo's own API, built with `clang` and `make`, graded by a test binary. |
| 7 | `F15-L07-sampling-mpc-in-cpp` | Sampling-based MPC in C++ on a cart-pole, measured in rollouts per second. |
| 8 | `F15-L08-sim-to-real-honesty` | Turn each reality gap into a number, build a domain-randomisation wrapper and show it narrows the held-out gap, then decide from published requirements what this course cannot give you. |

Lessons 6 and 7 are the systems lessons and are C and C++ by design: the point is the engine's
real API and the real cost of a step, which a Python wrapper hides. Everything else is a
jupytext `py:percent` notebook.

## Prerequisites

- Python, to the level of writing a function and a loop and reading a traceback.
- Enough linear algebra to know what a matrix–vector product is. Jacobians are built in
  lesson 2 from finite differences rather than assumed.
- For lessons 6 and 7: enough C to compile a file and read a segfault. No build-system
  knowledge is assumed; the `Makefile` discovers MuJoCo's headers and dylib itself.
- No machine-learning background. No reinforcement-learning background. Nothing in this
  flagship trains a neural network, and that is a deliberate choice explained in lesson 8.

## The hardware and compute truth

**This entire flagship runs on a laptop CPU.** Every lesson is tier `cpu8`: at most 8 GiB of
RAM, no GPU, and under ten minutes of wall clock, with the measured numbers written back into
each `meta.yaml` by the execution gate rather than typed by an author. No lesson downloads a
dataset on its required path. Nothing needs an API key.

That is a real constraint and it buys something real — you can do the whole thing on the
machine you already have, offline, for nothing. It also costs something real, and the course
says so rather than glossing it:

- **Isaac Sim and Isaac Lab are out of reach here.** NVIDIA's own requirements page names a
  GeForce RTX 4080 with 16 GB of VRAM as the minimum, states that GPUs without RT Cores are
  not supported, and supports Ubuntu and Windows only. Isaac Lab is built on top of Isaac Sim
  and inherits all of it. No amount of course design routes around a published hardware floor.
- **A free notebook tier does not rescue you, but not for the reason usually given.** The
  claim that the T4 "lacks ray-tracing cores" is false — NVIDIA's own product page says the T4
  has RT Cores, and it meets the 16 GB floor exactly. It fails because it is not a GPU NVIDIA
  names in the requirements table, and because Google's own FAQ says the GPU types available
  vary over time and are not guaranteed. Lesson 8 makes you compute that verdict rather than
  repeat either version of the folklore.
- **No GPU reinforcement learning.** Policies in this flagship are small parameter vectors
  found by search, not networks trained on a GPU cluster. That keeps every result reproducible
  on a CPU in seconds, and it is honest about what it is: the sim-to-real lesson transfers,
  the sample efficiency of modern RL does not get taught here.
- **No contact-rich manipulation, and no real robot.** Lesson 8's model deliberately disables
  contacts and pins the ankle. That simplification is named in the lesson and in the model
  file rather than hidden.

Where the ladder goes next — real arms, open-source humanoids, rented RTX hours — is priced in
lesson 8 from the sellers' own pages, with every URL and access date recorded in that lesson's
`claims.yaml`. One rung is deliberately left without a price, because no primary source for it
could be verified on the day.

## Running any lesson

```bash
# from the repository root

# the execution gate: runs the reference solution, measures wall time and peak RSS
.venv/bin/python tools/execute.py flagships/humanoid-lab/lessons/<lesson-id> --write-back

# the autograder, against your own work
.venv/bin/python tools/grade.py flagships/humanoid-lab/lessons/<lesson-id>

# regenerate the student notebook from lesson.py (never run jupytext directly: its random
# cell ids make the committed notebook look stale to tools/notebooks.py --check)
.venv/bin/python tools/notebooks.py --build flagships/humanoid-lab/lessons/<lesson-id>
```

`lesson.py` is the source of truth; the `.ipynb` is generated and never hand-edited.
`solutions/` and `tests/` are excluded from the student bundle by the build, not by
`.gitignore`.

## The contract

Every lesson here is bound by [`QUALITY.md`](../../QUALITY.md) — twelve gates covering
measurable objectives, interleaving, scaffolded stubs, partial-credit autograding, worked
solutions, real and free data, and sourced claims. Gate 12 is the one this flagship leans on
hardest: any statement about the world carries a primary source URL and an access date in the
lesson's `claims.yaml`, and any number a student sees in output is computed by the code they
ran rather than typed into prose.

## The capstone

[`CAPSTONE.md`](./CAPSTONE.md) — make the humanoid walk a measured distance without falling,
graded under randomised parameters rather than the nominal model.
