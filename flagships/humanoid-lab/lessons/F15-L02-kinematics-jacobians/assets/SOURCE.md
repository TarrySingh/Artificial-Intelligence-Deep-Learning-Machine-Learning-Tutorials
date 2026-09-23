# Asset provenance

## `arm.xml`

| | |
|---|---|
| Source | Authored for this lesson by the Synapsa Commons project. Not a download. |
| Licence | CC0-1.0 — <https://creativecommons.org/publicdomain/zero/1.0/> |
| Retrieved | 2026-09-16 (written, not fetched) |
| Gated? | No. Nothing to register for, nothing to agree to, no network access required. |

A humanoid upper body welded to the world: a pelvis, a torso on two waist hinges, a
six-joint right arm ending in a `palm` site, and a two-joint left arm. The file is the
complete model — no `<mesh>`, no `file=` attribute, no `<include>`, no procedural asset that
has to be fetched. `lesson.py` resolves it through `arm_xml_path()`, which searches the
lesson directory and its parent and never reaches for the network. There is no download
branch to fail.

### Three properties, each chosen to make one teaching point measurable

**Every joint is a hinge, and the pelvis is welded.** So `nq == nv == 10`, and `qpos` index
*i* is dof index *i* for every *i*. That 1:1 mapping is what lets the finite-difference
exercise perturb `qpos[i]` directly. It is also a privilege the lesson is explicit about
losing: on a model with a free or ball joint, `qpos` carries a four-number quaternion where
`qvel` carries a three-number angular velocity, and `qpos[i] += eps` no longer means "move
dof *i*". Section 9 loads a short MJCF string with a ball joint and prints its `nq` and
`nv` rather than asking anyone to take that on trust.

**The arms are asymmetric.** Eight degrees of freedom move the right palm; the two left-arm
joints move it not at all. Their Jacobian columns come back *exactly* zero — `np.array_equal`
against `zeros`, not `allclose` — which is the cleanest available demonstration that a
Jacobian column answers "if this one dof moved, where would this point go?". It also gives
the finite-difference exercise an honest reason to difference a handful of dofs rather than
all ten, which is what keeps the sweep cheap.

**The home pose is singular on purpose.** At `qpos = 0` the right arm hangs straight down,
fully extended: the shoulder origin sits at z = 1.38 and the palm at z = 0.745, exactly
0.635 m below it — the sum of the three link offsets, 0.28 + 0.25 + 0.105. A fully extended
arm cannot move its palm along its own axis at any joint velocity, so the translational
Jacobian loses rank there. Measured on the shipped model, its singular values at home are
approximately (0.807, 0.644, 0.000): the third is not small, it is zero.

That is what makes the damping term in sections 10 and 11 load-bearing rather than decorative. The
inverse-kinematics solver starts every solve from the home pose, so with `lam = 0` the normal
equations are singular and NumPy raises `LinAlgError` on the first iteration — the notebook
runs that and prints the exception. Nudge the elbow a ten-thousandth of a radian off full
extension and the undamped step instead asks for a joint velocity of order 10² rad, while the
damped step stays near zero. Both numbers are computed by the code the student runs.

### The palm site is offset on purpose

`palm` sits 0.105 m beyond the `hand_right` body's frame origin, along the hand's own −z. The
frames section rests on that offset being real: the body frame origin, the body centre of
mass and the site are three different points rigidly attached to the same body, so
`mj_jacBody`, `mj_jacBodyCom` and `mj_jacSite` return three different translational Jacobians
even though the body — and therefore every joint axis in the chain — is identical. MuJoCo's
own API reference states that the variants "call mj_jac internally, with the center of the
body, geom or site", and the notebook measures the resulting gap rather than describing it.

### What was deliberately left out

There are no actuators and the lesson never calls `mj_step`. Everything here is the position
stage of the pipeline: set `qpos`, run forward kinematics, read a position or a Jacobian. Add
motors and a student would step the model, watch the arm sag under gravity, and spend their
attention on a control problem that belongs to F15-L03. Contacts are disabled for the same
reason, plus one practical one: a contact-free `mj_forward` on ten dofs costs a few
microseconds, which is what puts a damping sweep, a ten-point step-size sweep and several
hundred inverse-kinematics iterations inside a laptop CPU budget.
