// F15-L06 — the control loop in C, against MuJoCo's own C API.
//
// Three functions are stubs. Fill them in, then run:
//
//     make test
//
// The self-test reports each exercise separately, so you can finish them one at a time and
// watch the TODOs turn into PASSes. Everything below the three exercises is the harness; it
// is given, and it is worth reading, because it is the shape every control loop has.

/* clock_gettime and CLOCK_MONOTONIC are POSIX, not ISO C. Under -std=c11, glibc (Linux)
   hides them unless asked before the first header; macOS shows them regardless. */
#define _DEFAULT_SOURCE
#include <mujoco/mujoco.h>

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------------------
// Defaults. Every one is overridable on the command line, because the point of the lesson is
// to let a student move a gain and watch what it costs.
// ---------------------------------------------------------------------------------------
#define MAXU 16          // most actuators this teaching binary will handle
#define MAXTRACE 256     // rows of trajectory kept for the Python comparison

// Three helpers below are used by some builds of this file and not others: todo() disappears
// once you have finished all three exercises, and clamp()/stats_tick() are unreferenced until
// you start calling them. Marking them keeps -Wunused-function meaningful for YOUR code
// instead of drowning it in warnings about the scaffolding.
#define COMMONS_MAYBE_UNUSED __attribute__((unused))

static const double kDefaultKp = 40.0;
static const double kDefaultKd = 3.0;
static const int    kDefaultTicks = 1500;
static const double kDefaultTarget[2] = {0.8, -1.2};

// A joint counts as "settled" when it is inside this many radians of its target AND stays
// there. Leaving the band resets the clock, so a settle tick is a time after which the arm
// never left again — not merely the first time it brushed past.
static const double kSettleTol = 0.05;

// =======================================================================================
// Plumbing for the self-test. Read it once, then ignore it.
// =======================================================================================
//
// C has no exceptions, so an unfinished exercise reports itself with setjmp/longjmp: todo()
// records a message and jumps back to the check that called it. main() turns an unfinished
// exercise into exit code 2, which the lesson's Python wrapper translates into
// NotImplementedError so the grader prints TODO rather than a stack trace. A real failure
// jumps the same way with a different code and becomes exit 1.
static jmp_buf g_jmp;
static int g_jmp_active = 0;
static char g_msg[1024];

static COMMONS_MAYBE_UNUSED void todo(const char* what, const char* hint) {
  snprintf(g_msg, sizeof g_msg, "%s() is still a stub — %s", what, hint);
  if (g_jmp_active) longjmp(g_jmp, 1);
  fprintf(stderr, "NOT IMPLEMENTED: %s\n", g_msg);
  exit(2);
}

static void require(int ok, const char* fmt, ...) {
  if (ok) return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_msg, sizeof g_msg, fmt, ap);
  va_end(ap);
  if (g_jmp_active) longjmp(g_jmp, 2);
  fprintf(stderr, "error: %s\n", g_msg);
  exit(1);
}

static COMMONS_MAYBE_UNUSED double clamp(double v, double lo, double hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// =======================================================================================
// EXERCISE 1 — actuator_addresses()
// =======================================================================================
//
// Given an actuator index, find the joint it drives and that joint's two addresses: where
// its position lives in d->qpos, and where its velocity lives in d->qvel.
//
// mjModel carries three tables that answer this, and the whole exercise is looking up the
// right one:
//
//     m->actuator_trnid[2*actuator]   the id of the joint this actuator transmits to
//     m->jnt_qposadr[joint]           that joint's first slot in qpos
//     m->jnt_dofadr[joint]            that joint's first slot in qvel
//
// Write the joint id into the return value, the qpos address through *qadr, and the qvel
// address through *vadr.
//
// It is tempting to skip all three and write qpos[actuator] / qvel[actuator]. On the pinned
// arm that is RIGHT, which is exactly what makes it dangerous: with two hinges and nothing
// else, nq == nv == 2 and the addresses happen to agree. Load assets/arm2_floating.xml,
// where the base carries a free joint, and the coincidence ends — the free joint takes seven
// qpos slots but only six qvel rows, so every hinge after it sits at a different address in
// the two arrays. The self-test checks BOTH models for this reason.
//
// Worked example, on assets/arm2_floating.xml: actuator 0 drives joint 1, whose qpos address
// is 7 and whose qvel address is 6. On assets/arm2.xml the same actuator drives joint 0, at
// qpos address 0 and qvel address 0.
static int actuator_addresses(const mjModel* m, int actuator, int* qadr, int* vadr) {
  // YOUR CODE HERE
  todo("actuator_addresses",
       "read the joint id from m->actuator_trnid[2*actuator], then write "
       "m->jnt_qposadr[joint] through qadr and m->jnt_dofadr[joint] through vadr, and return "
       "the joint id");
  return -1;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 2 — pd_ctrl()
// =======================================================================================
//
// Compute one command per actuator and write them into ctrl_out. For actuator i:
//
//     tau = kp * (target[i] - qpos[qadr]) - kd * qvel[vadr]
//     ctrl_out[i] = clamp(tau, ctrlrange_lo, ctrlrange_hi)   when the actuator is limited
//
// Three things to get right, each of which the self-test checks separately:
//
//  * target is indexed by ACTUATOR (target[i]), while qpos and qvel are indexed by the
//    ADDRESSES exercise 1 hands you. They are different index spaces and only agree by
//    accident on simple models.
//  * the limits are per actuator, and they are not symmetric in general:
//    m->actuator_ctrlrange[2*i] is the low bound, [2*i + 1] the high one.
//  * only clamp when m->actuator_ctrllimited[i] is true. An unlimited actuator has a
//    meaningless ctrlrange and clamping to it would silently throttle the controller.
//
// Worked example, on assets/arm2.xml with kp = 40, kd = 3, target = {0.8, -1.2} from the
// hanging keyframe (qpos = {0, 0}, qvel = {0, 0}): the raw commands are 40*0.8 = 32 and
// 40*(-1.2) = -48, and the ctrlranges are [-8, 8] and [-3, 3], so ctrl_out comes back
// {8, -3} — both actuators pinned to their limits on the very first tick.
static void pd_ctrl(const mjModel* m, const mjData* d, const double* target, double kp,
                    double kd, double* ctrl_out) {
  // YOUR CODE HERE
  todo("pd_ctrl",
       "loop i over m->nu: get the addresses from actuator_addresses, compute "
       "kp*(target[i] - d->qpos[qadr]) - kd*d->qvel[vadr], clamp it to "
       "m->actuator_ctrlrange[2*i]..[2*i+1] when m->actuator_ctrllimited[i], and store it in "
       "ctrl_out[i]");
}

// ---------------------------------------------------------------------------------------
// Bookkeeping for the loop. GIVEN — not an exercise. It is here so that exercise 3 is the
// loop and nothing but the loop.
// ---------------------------------------------------------------------------------------
typedef struct {
  long long ticks_done;
  long long saturated_ticks;
  int settle_tick;
  double peak_abs_err;
  int n_trace;
  int trace_every;
  double trace[MAXTRACE][8];  // k, time, qpos.., qvel.., ctrl..  (2 actuators assumed)
} LoopStats;

// The largest distance any actuated joint is from its target, right now.
static double max_abs_error(const mjModel* m, const mjData* d, const double* target) {
  double worst = 0.0;
  for (int i = 0; i < (int)m->nu; ++i) {
    int qadr = 0, vadr = 0;
    actuator_addresses(m, i, &qadr, &vadr);
    const double e = fabs(target[i] - d->qpos[qadr]);
    if (e > worst) worst = e;
  }
  return worst;
}

// Did any command come back sitting on its limit?
static int at_limit(const mjModel* m, const double* ctrl) {
  for (int i = 0; i < (int)m->nu; ++i) {
    if (!m->actuator_ctrllimited[i]) continue;
    const double lo = m->actuator_ctrlrange[2 * i], hi = m->actuator_ctrlrange[2 * i + 1];
    if (ctrl[i] >= hi - 1e-12 || ctrl[i] <= lo + 1e-12) return 1;
  }
  return 0;
}

// Everything that happens after a step: the counters, the settle clock and the trace.
// Call it once per tick, AFTER mj_step, with the ctrl you actually applied.
static COMMONS_MAYBE_UNUSED void stats_tick(const mjModel* m, const mjData* d, const double* ctrl,
                                          const double* target, int k, LoopStats* s) {
  s->ticks_done += 1;
  s->saturated_ticks += at_limit(m, ctrl);
  const double err = max_abs_error(m, d, target);
  if (err > s->peak_abs_err) s->peak_abs_err = err;
  // Leaving the band restarts the clock, so settle_tick ends up being the tick after which
  // the arm never left again.
  if (err < kSettleTol) {
    if (s->settle_tick < 0) s->settle_tick = k;
  } else {
    s->settle_tick = -1;
  }
  if (s->trace_every > 0 && k % s->trace_every == 0 && s->n_trace < MAXTRACE) {
    double* row = s->trace[s->n_trace++];
    row[0] = (double)k;
    row[1] = d->time;
    for (int i = 0; i < 2 && i < (int)m->nu; ++i) {
      int qadr = 0, vadr = 0;
      actuator_addresses(m, i, &qadr, &vadr);
      row[2 + i] = d->qpos[qadr];
      row[4 + i] = d->qvel[vadr];
      row[6 + i] = ctrl[i];
    }
  }
}

// =======================================================================================
// EXERCISE 3 — run_loop()
// =======================================================================================
//
// The fixed-step control loop. This is the shape every simulated controller has, and it is
// four lines. For each of `ticks` ticks, in this order:
//
//   1. pd_ctrl(m, d, target, kp, kd, ctrl)   compute the commands from the CURRENT state
//   2. copy ctrl into d->ctrl                hand them to the engine
//   3. mj_step(m, d)                         advance the physics exactly once
//   4. stats_tick(m, d, ctrl, target, k, s)  record what just happened
//
// The order is the lesson. Step first and you have applied last tick's command to this
// tick's state — a one-tick delay you did not ask for and will not see until the gains go
// up and the loop rings. Call mj_step twice and simulated time runs at double the rate your
// controller thinks it does. Forget step 2 and the arm never moves, because computing a
// command and giving it to the engine are two different acts: d->ctrl is the only channel
// the engine reads.
//
// `ctrl` is a scratch buffer of MAXU doubles, already allocated for you. Write into it.
//
// Worked example: 1500 ticks at kp = 40, kd = 3 on assets/arm2.xml leaves d->time at exactly
// 1500 * 0.002 = 3.0 s, and s->ticks_done at 1500.
static void run_loop(const mjModel* m, mjData* d, const double* target, double kp, double kd,
                     int ticks, double* ctrl, LoopStats* s) {
  // YOUR CODE HERE
  todo("run_loop",
       "loop k from 0 to ticks: call pd_ctrl into the ctrl buffer, copy ctrl[i] into "
       "d->ctrl[i] for every actuator, call mj_step(m, d) exactly once, then call "
       "stats_tick(m, d, ctrl, target, k, s) — in that order");
}

// =======================================================================================
// Everything below here is the harness.
// =======================================================================================

typedef struct {
  const char* model;
  double kp, kd;
  int ticks;
  int trace_every;
  double target[MAXU];
  int n_target;
  double qpos[MAXU], qvel[MAXU];
  int n_qpos, n_qvel;
  int unlimited;   // clear every actuator's ctrllimited flag before computing (ctrldump)
} Options;

static void emit(const char* key, double v) { printf("@ %s=%.17g\n", key, v); }
static void emit_i(const char* key, long long v) { printf("@ %s=%lld\n", key, v); }

static mjModel* load_model(const char* path) {
  char err[1000] = "";
  mjModel* m = mj_loadXML(path, NULL, err, sizeof err);
  require(m != NULL, "could not load model '%s': %s", path, err);
  require((int)m->nu <= MAXU, "model has %lld actuators; this teaching binary handles %d",
          (long long)m->nu, MAXU);
  return m;
}

static void reset_start(const mjModel* m, mjData* d) {
  if (m->nkey > 0) {
    mj_resetDataKeyframe(m, d, 0);
  } else {
    mj_resetData(m, d);
  }
  mj_forward(m, d);
}

static void stats_init(LoopStats* s, int trace_every) {
  memset(s, 0, sizeof *s);
  s->settle_tick = -1;
  s->trace_every = trace_every;
}

// ---------------------------------------------------------------------------------------
// The PD run, with a printed summary. This is the command the lesson's payoff section calls.
// ---------------------------------------------------------------------------------------
static int cmd_run(const Options* o) {
  mjModel* m = load_model(o->model);
  mjData* d = mj_makeData(m);
  reset_start(m, d);

  double ctrl[MAXU] = {0};
  LoopStats s;
  stats_init(&s, o->trace_every);

  const double t0 = now_seconds();
  run_loop(m, d, o->target, o->kp, o->kd, o->ticks, ctrl, &s);
  const double wall = now_seconds() - t0;

  emit_i("nq", (long long)m->nq);
  emit_i("nv", (long long)m->nv);
  emit_i("nu", (long long)m->nu);
  emit_i("ticks", o->ticks);
  emit_i("ticks_done", s.ticks_done);
  emit("timestep", m->opt.timestep);
  emit("sim_seconds", d->time);
  emit("kp", o->kp);
  emit("kd", o->kd);
  emit_i("saturated_ticks", s.saturated_ticks);
  emit("saturated_fraction", o->ticks ? (double)s.saturated_ticks / o->ticks : 0.0);
  emit_i("settle_tick", s.settle_tick);
  emit("settle_seconds", s.settle_tick >= 0 ? s.settle_tick * m->opt.timestep : -1.0);
  emit("peak_abs_err", s.peak_abs_err);
  emit("final_abs_err", max_abs_error(m, d, o->target));
  emit("wall_seconds", wall);
  emit("ticks_per_second", wall > 0 ? o->ticks / wall : 0.0);

  // Per-actuator final error, and the gravitational load that explains it. qfrc_bias is the
  // force the arm needs just to stand still; a PD law with no integral term can only supply
  // it by sitting at an error of bias/kp, which is the droop the lesson measures.
  for (int i = 0; i < (int)m->nu; ++i) {
    int qadr = 0, vadr = 0;
    actuator_addresses(m, i, &qadr, &vadr);
    printf("err %d %.17g %.17g %.17g %.17g\n", i, o->target[i] - d->qpos[qadr],
           d->qfrc_bias[vadr], d->qfrc_bias[vadr] / o->kp, d->ctrl[i]);
  }
  for (int r = 0; r < s.n_trace; ++r) {
    printf("trace");
    for (int c = 0; c < 8; ++c) printf(" %.17g", s.trace[r][c]);
    printf("\n");
  }

  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}

// ---------------------------------------------------------------------------------------
// Throughput of the same loop, with the trace switched off. This is the number the language
// comparison rests on, so it must do exactly the work the Python loop does: no more, no less.
// ---------------------------------------------------------------------------------------
static int cmd_bench(const Options* o) {
  mjModel* m = load_model(o->model);
  mjData* d = mj_makeData(m);
  reset_start(m, d);

  double ctrl[MAXU] = {0};
  LoopStats s;
  stats_init(&s, 0);  // no trace: the Python loop does not build one either

  const double t0 = now_seconds();
  run_loop(m, d, o->target, o->kp, o->kd, o->ticks, ctrl, &s);
  const double wall = now_seconds() - t0;

  emit_i("ticks", o->ticks);
  emit_i("ticks_done", s.ticks_done);
  emit("wall_seconds", wall);
  emit("ticks_per_second", wall > 0 ? o->ticks / wall : 0.0);
  emit("us_per_tick", o->ticks ? wall * 1e6 / o->ticks : 0.0);
  emit("sim_seconds", d->time);
  // A checksum the optimiser cannot discard the loop without changing.
  emit("final_abs_err", max_abs_error(m, d, o->target));

  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}

// ---------------------------------------------------------------------------------------
// Print the address table for every actuator. Depends on exercise 1 alone.
// ---------------------------------------------------------------------------------------
static int cmd_addrdump(const Options* o) {
  mjModel* m = load_model(o->model);
  emit_i("nq", (long long)m->nq);
  emit_i("nv", (long long)m->nv);
  emit_i("nu", (long long)m->nu);
  emit_i("njnt", (long long)m->njnt);
  for (int i = 0; i < (int)m->nu; ++i) {
    int qadr = -1, vadr = -1;
    const int joint = actuator_addresses(m, i, &qadr, &vadr);
    printf("addr %d %d %d %d %d\n", i, joint, qadr, vadr,
           joint >= 0 && joint < (int)m->njnt ? m->jnt_type[joint] : -1);
  }
  mj_deleteModel(m);
  return 0;
}

// ---------------------------------------------------------------------------------------
// Compute one control vector from a state supplied on the command line. Depends on
// exercises 1 and 2, and on no stepping at all, so the rubric can check the control law in
// isolation from the loop.
// ---------------------------------------------------------------------------------------
static int cmd_ctrldump(const Options* o) {
  mjModel* m = load_model(o->model);
  // --unlimited turns every actuator on this loaded model into an unlimited one. Both shipped
  // models limit both actuators, so without this the `if (m->actuator_ctrllimited[i])` in
  // pd_ctrl can never be observed: clamping unconditionally would behave identically and no
  // check could tell the two apart.
  if (o->unlimited) {
    for (int i = 0; i < (int)m->nu; ++i) m->actuator_ctrllimited[i] = 0;
  }
  mjData* d = mj_makeData(m);
  reset_start(m, d);
  for (int i = 0; i < o->n_qpos && i < (int)m->nq; ++i) d->qpos[i] = o->qpos[i];
  for (int i = 0; i < o->n_qvel && i < (int)m->nv; ++i) d->qvel[i] = o->qvel[i];
  mj_forward(m, d);

  double ctrl[MAXU] = {0};
  pd_ctrl(m, d, o->target, o->kp, o->kd, ctrl);

  emit_i("nu", (long long)m->nu);
  emit("kp", o->kp);
  emit("kd", o->kd);
  for (int i = 0; i < (int)m->nu; ++i) {
    printf("ctrl %d %.17g %.17g %.17g\n", i, ctrl[i], m->actuator_ctrlrange[2 * i],
           m->actuator_ctrlrange[2 * i + 1]);
  }
  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}

// ---------------------------------------------------------------------------------------
// The self-test: your instant feedback. Each check runs independently, so an unfinished
// exercise is reported as TODO without stopping the others.
// ---------------------------------------------------------------------------------------
typedef struct {
  const char* name;
  int status;  // 0 pass, 1 fail, 2 todo
  char msg[1024];
} Check;

static const mjModel* g_m;          // the pinned arm, for the check bodies
static mjModel* g_m_rw;             // the same model, writable, so a check can clear a flag
static const mjModel* g_mfloat;     // the floating-base arm
static mjData* g_d;

static void check_addresses(void) {
  // On the pinned arm every joint is a hinge, so the addresses are the identity.
  for (int i = 0; i < (int)g_m->nu; ++i) {
    int qadr = -1, vadr = -1;
    const int joint = actuator_addresses(g_m, i, &qadr, &vadr);
    require(joint == g_m->actuator_trnid[2 * i],
            "actuator %d drives joint %d, but you returned %d — the joint id is "
            "m->actuator_trnid[2*actuator]", i, g_m->actuator_trnid[2 * i], joint);
    require(qadr == g_m->jnt_qposadr[joint],
            "actuator %d: qpos address came back %d, expected %d (m->jnt_qposadr[joint])",
            i, qadr, g_m->jnt_qposadr[joint]);
    require(vadr == g_m->jnt_dofadr[joint],
            "actuator %d: qvel address came back %d, expected %d (m->jnt_dofadr[joint])",
            i, vadr, g_m->jnt_dofadr[joint]);
  }
  // And now the model that can tell a lookup from a lucky guess. The base carries a free
  // joint: 7 qpos slots, 6 qvel rows, so the hinges after it sit at addresses that differ.
  if (!g_mfloat) return;
  for (int i = 0; i < (int)g_mfloat->nu; ++i) {
    int qadr = -1, vadr = -1;
    const int joint = actuator_addresses(g_mfloat, i, &qadr, &vadr);
    const int want_q = g_mfloat->jnt_qposadr[g_mfloat->actuator_trnid[2 * i]];
    const int want_v = g_mfloat->jnt_dofadr[g_mfloat->actuator_trnid[2 * i]];
    require(qadr == want_q && vadr == want_v,
            "on the floating-base arm, actuator %d sits at qpos address %d and qvel address "
            "%d, but you returned qpos %d / qvel %d. The free joint takes 7 qpos slots and "
            "only 6 qvel rows, so the two are NOT the same number — look each one up in its "
            "own table (jnt_qposadr for qpos, jnt_dofadr for qvel) instead of reusing one.",
            i, want_q, want_v, qadr, vadr);
    require(qadr != vadr || joint == 0,
            "actuator %d returned the same address for qpos and qvel on a model with a free "
            "joint; that cannot be right", i);
  }
}

static void check_pd_ctrl(void) {
  const double target[2] = {0.8, -1.2};
  double ctrl[MAXU] = {0};

  // From the hanging keyframe the raw commands are far beyond both limits, so both must come
  // back sitting exactly on their bounds.
  reset_start(g_m, g_d);
  pd_ctrl(g_m, g_d, target, 40.0, 3.0, ctrl);
  for (int i = 0; i < (int)g_m->nu; ++i) {
    const double lo = g_m->actuator_ctrlrange[2 * i], hi = g_m->actuator_ctrlrange[2 * i + 1];
    require(ctrl[i] >= lo - 1e-12 && ctrl[i] <= hi + 1e-12,
            "actuator %d came back at %g, outside its ctrlrange [%g, %g] — clamp each command "
            "to ITS OWN limits, from m->actuator_ctrlrange[2*i] and [2*i + 1]",
            i, ctrl[i], lo, hi);
  }
  require(fabs(ctrl[0] - g_m->actuator_ctrlrange[1]) < 1e-12,
          "at kp = 40 the shoulder is asked for 40*0.8 = 32 against a limit of %g, so it must "
          "clamp to exactly %g; you returned %g",
          g_m->actuator_ctrlrange[1], g_m->actuator_ctrlrange[1], ctrl[0]);
  require(fabs(ctrl[1] - g_m->actuator_ctrlrange[2]) < 1e-12,
          "the elbow is asked for 40*(-1.2) = -48 against a low limit of %g, so it must clamp "
          "to exactly %g; you returned %g",
          g_m->actuator_ctrlrange[2], g_m->actuator_ctrlrange[2], ctrl[1]);

  // Sitting exactly on target with no velocity: every command must be exactly zero. This is
  // the check a sign error cannot survive.
  reset_start(g_m, g_d);
  for (int i = 0; i < (int)g_m->nu; ++i) {
    int qadr = 0, vadr = 0;
    actuator_addresses(g_m, i, &qadr, &vadr);
    g_d->qpos[qadr] = target[i];
    g_d->qvel[vadr] = 0.0;
  }
  mj_forward(g_m, g_d);
  pd_ctrl(g_m, g_d, target, 40.0, 3.0, ctrl);
  for (int i = 0; i < (int)g_m->nu; ++i) {
    require(fabs(ctrl[i]) < 1e-12,
            "with the arm exactly on target and at rest, actuator %d still asks for %g. The "
            "error term is (target - qpos), so it is zero here; a non-zero command means the "
            "error is the wrong way round or gravity has crept into your law.", i, ctrl[i]);
  }

  // A small error, well inside the limits, so the arithmetic itself is checked rather than
  // the clamp. Also pins down the sign of the damping term.
  reset_start(g_m, g_d);
  {
    int qadr = 0, vadr = 0;
    actuator_addresses(g_m, 0, &qadr, &vadr);
    g_d->qpos[qadr] = target[0] - 0.1;   // 0.1 rad below target
    g_d->qvel[vadr] = 0.2;               // and moving up at 0.2 rad/s
    mj_forward(g_m, g_d);
  }
  pd_ctrl(g_m, g_d, target, 10.0, 2.0, ctrl);
  const double want = 10.0 * 0.1 - 2.0 * 0.2;  // = 0.6
  require(fabs(ctrl[0] - want) < 1e-9,
          "with an error of +0.1 rad and a velocity of +0.2 rad/s at kp = 10, kd = 2, the "
          "command must be 10*0.1 - 2*0.2 = %g; you returned %g. The damping term is "
          "SUBTRACTED, and it uses qvel, not the change in error.", want, ctrl[0]);

  // An unlimited actuator must not be clamped to a meaningless range. The shipped model
  // limits both actuators — that is the premise of every check above — so the flag is cleared
  // here for the length of one call and put straight back. A law that clamps unconditionally
  // comes back pinned to a range that no longer applies, and nothing else in this file can
  // see that.
  for (int i = 0; i < (int)g_m->nu; ++i) {
    require(g_m->actuator_ctrllimited[i],
            "this check assumes the teaching model limits every actuator; actuator %d is "
            "unlimited, so the model has changed underneath the lesson", i);
  }
  require(g_m_rw != NULL, "the self-test needs a writable handle on the model");
  {
    mjtByte saved[MAXU];
    for (int i = 0; i < (int)g_m->nu; ++i) {
      saved[i] = g_m_rw->actuator_ctrllimited[i];
      g_m_rw->actuator_ctrllimited[i] = 0;
    }
    reset_start(g_m, g_d);
    pd_ctrl(g_m, g_d, target, 40.0, 3.0, ctrl);
    for (int i = 0; i < (int)g_m->nu; ++i) g_m_rw->actuator_ctrllimited[i] = saved[i];
    require(fabs(ctrl[0] - 32.0) < 1e-9 && fabs(ctrl[1] - (-48.0)) < 1e-9,
            "with ctrllimited cleared, the law must hand back its RAW commands 40*0.8 = 32 "
            "and 40*(-1.2) = -48; you returned %g and %g. Clamp only when "
            "m->actuator_ctrllimited[i] is set — an unlimited actuator's ctrlrange is "
            "meaningless, and clamping to it throttles the controller silently.",
            ctrl[0], ctrl[1]);
  }
}

static void check_run_loop(void) {
  const double target[2] = {0.8, -1.2};
  double ctrl[MAXU] = {0};
  LoopStats s;

  // Exactly one mj_step per tick, and simulated time to prove it.
  reset_start(g_m, g_d);
  stats_init(&s, 0);
  const int ticks = 200;
  run_loop(g_m, g_d, target, 40.0, 3.0, ticks, ctrl, &s);
  const double want_time = ticks * g_m->opt.timestep;
  require(fabs(g_d->time - want_time) < 1e-12,
          "after %d ticks the simulation clock reads %.6f s but one mj_step per tick at a "
          "timestep of %g s is %.6f s. %s",
          ticks, g_d->time, g_m->opt.timestep, want_time,
          g_d->time > want_time ? "You are stepping more than once per tick."
                                : "You are stepping fewer times than you have ticks.");
  require(s.ticks_done == ticks,
          "stats_tick ran %lld times over %d ticks — call it exactly once per tick, after "
          "mj_step", s.ticks_done, ticks);

  // The arm must actually have moved toward the target. If d->ctrl never receives the
  // commands, the arm hangs where it started and the error stays at the full target.
  const double err = max_abs_error(g_m, g_d, target);
  require(err < 1.2,
          "after %d ticks the arm is still %.3f rad from target, which is about where it "
          "started. Computing a command is not the same as applying one: copy ctrl[i] into "
          "d->ctrl[i] before calling mj_step — d->ctrl is the only channel the engine reads.",
          ticks, err);

  // Order matters: control computed from the CURRENT state, then the step. A loop that steps
  // first applies a stale command, and its trajectory is measurably different.
  reset_start(g_m, g_d);
  stats_init(&s, 0);
  run_loop(g_m, g_d, target, 40.0, 3.0, 1, ctrl, &s);
  {
    // One tick from the keyframe: the command is the clamped one computed at qpos = 0, and
    // after the step the shoulder must have risen.
    int qadr = 0, vadr = 0;
    actuator_addresses(g_m, 0, &qadr, &vadr);
    require(g_d->qpos[qadr] > 0.0,
            "after one tick the shoulder is at %g; a positive command should have started it "
            "moving up. If it is exactly 0 you stepped before applying the control.",
            g_d->qpos[qadr]);
  }

  // Determinism: the same run twice must give the same answer.
  reset_start(g_m, g_d);
  stats_init(&s, 0);
  run_loop(g_m, g_d, target, 40.0, 3.0, 300, ctrl, &s);
  const double first = max_abs_error(g_m, g_d, target);
  reset_start(g_m, g_d);
  LoopStats s2;
  stats_init(&s2, 0);
  run_loop(g_m, g_d, target, 40.0, 3.0, 300, ctrl, &s2);
  const double second = max_abs_error(g_m, g_d, target);
  require(first == second,
          "the same 300-tick run gave %.17g and then %.17g — something is carrying state "
          "between runs that should not be", first, second);
  require(s.saturated_ticks == s2.saturated_ticks,
          "the saturated-tick count changed between two identical runs (%lld then %lld)",
          s.saturated_ticks, s2.saturated_ticks);
}

static Check run_check(const char* name, void (*body)(void)) {
  Check c;
  c.name = name;
  c.msg[0] = '\0';
  g_msg[0] = '\0';
  g_jmp_active = 1;
  const int jumped = setjmp(g_jmp);
  if (jumped == 0) {
    body();
    c.status = 0;
  } else {
    c.status = (jumped == 1) ? 2 : 1;
    snprintf(c.msg, sizeof c.msg, "%s", g_msg);
  }
  g_jmp_active = 0;
  return c;
}

static int cmd_selftest(const Options* o) {
  mjModel* m = load_model(o->model);
  mjData* d = mj_makeData(m);
  g_m = m;
  g_m_rw = m;
  g_d = d;

  // The floating-base companion model, if it is next to the one we were given. It is never
  // stepped; it exists only so the address check has a model where the two address spaces
  // genuinely differ.
  char alt[1024];
  snprintf(alt, sizeof alt, "%s", o->model);
  char* slash = strrchr(alt, '/');
  if (slash) {
    snprintf(slash + 1, sizeof alt - (size_t)(slash + 1 - alt), "arm2_floating.xml");
  } else {
    snprintf(alt, sizeof alt, "arm2_floating.xml");
  }
  char err[1000] = "";
  g_mfloat = mj_loadXML(alt, NULL, err, sizeof err);

  Check results[3];
  results[0] = run_check("actuator_addresses", check_addresses);
  results[1] = run_check("pd_ctrl", check_pd_ctrl);
  results[2] = run_check("run_loop", check_run_loop);

  int failed = 0, todos = 0;
  printf("\n  F15-L06 self-test (%s%s)\n\n", o->model,
         g_mfloat ? " + arm2_floating.xml" : " — floating model not found, address check is partial");
  for (int i = 0; i < 3; ++i) {
    const char* tag = results[i].status == 0 ? "PASS " : (results[i].status == 2 ? "TODO " : "FAIL ");
    printf("  [%s] %-20s %s\n", tag, results[i].name, results[i].msg);
    if (results[i].status == 1) ++failed;
    if (results[i].status == 2) ++todos;
  }
  printf("\n  3 checks, %d failed, %d not implemented\n\n", failed, todos);

  if (g_mfloat) mj_deleteModel((mjModel*)g_mfloat);
  mj_deleteData(d);
  mj_deleteModel(m);
  if (failed) return 1;
  if (todos) return 2;
  return 0;
}

static void usage(void) {
  printf(
      "usage: lesson_bin <command> [options]\n\n"
      "commands:\n"
      "  selftest    run the built-in checks for the three exercises\n"
      "  run         the PD control loop, with a printed summary and a trace\n"
      "  bench       the same loop, timed, with the trace switched off\n"
      "  addrdump    the qpos/qvel address table for every actuator\n"
      "  ctrldump    one control vector, computed from a state given on the command line\n\n"
      "options:\n"
      "  --model F --kp K --kd D --ticks N --trace-every N\n"
      "  --target a,b --qpos a,b --qvel a,b\n"
      "  --unlimited     (ctrldump) clear every actuator's ctrllimited flag first\n");
}

// Parse "0.8,-1.2" into out[]; returns how many numbers were read.
static int parse_list(const char* text, double* out, int cap) {
  int n = 0;
  const char* p = text;
  while (*p && n < cap) {
    char* end = NULL;
    const double v = strtod(p, &end);
    if (end == p) break;
    out[n++] = v;
    p = end;
    while (*p == ',' || *p == ' ') ++p;
  }
  return n;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 64;
  }
  const char* cmd = argv[1];

  Options o;
  memset(&o, 0, sizeof o);
  o.model = "assets/arm2.xml";
  o.kp = kDefaultKp;
  o.kd = kDefaultKd;
  o.ticks = kDefaultTicks;
  o.trace_every = 25;
  o.target[0] = kDefaultTarget[0];
  o.target[1] = kDefaultTarget[1];
  o.n_target = 2;

  for (int i = 2; i < argc; ++i) {
    const char* k = argv[i];
    const char* v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define NEEDVAL()                                                       \
  do {                                                                  \
    if (!v) {                                                           \
      fprintf(stderr, "option %s needs a value\n", k);                   \
      return 64;                                                        \
    }                                                                   \
    ++i;                                                                \
  } while (0)
    if (!strcmp(k, "--model")) { NEEDVAL(); o.model = v; }
    else if (!strcmp(k, "--kp")) { NEEDVAL(); o.kp = atof(v); }
    else if (!strcmp(k, "--kd")) { NEEDVAL(); o.kd = atof(v); }
    else if (!strcmp(k, "--ticks")) { NEEDVAL(); o.ticks = atoi(v); }
    else if (!strcmp(k, "--trace-every")) { NEEDVAL(); o.trace_every = atoi(v); }
    else if (!strcmp(k, "--target")) { NEEDVAL(); o.n_target = parse_list(v, o.target, MAXU); }
    else if (!strcmp(k, "--qpos")) { NEEDVAL(); o.n_qpos = parse_list(v, o.qpos, MAXU); }
    else if (!strcmp(k, "--qvel")) { NEEDVAL(); o.n_qvel = parse_list(v, o.qvel, MAXU); }
    else if (!strcmp(k, "--unlimited")) { o.unlimited = 1; }
    else if (!strcmp(k, "--help") || !strcmp(k, "-h")) { usage(); return 0; }
    else {
      fprintf(stderr, "unknown option %s\n", k);
      return 64;
    }
#undef NEEDVAL
  }
  if (o.ticks < 1) {
    fprintf(stderr, "--ticks must be >= 1\n");
    return 64;
  }

  if (!strcmp(cmd, "selftest")) return cmd_selftest(&o);
  if (!strcmp(cmd, "run")) return cmd_run(&o);
  if (!strcmp(cmd, "bench")) return cmd_bench(&o);
  if (!strcmp(cmd, "addrdump")) return cmd_addrdump(&o);
  if (!strcmp(cmd, "ctrldump")) return cmd_ctrldump(&o);
  fprintf(stderr, "unknown command '%s'\n", cmd);
  usage();
  return 64;
}
