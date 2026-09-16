// F15-L07 — sampling-based model-predictive control (MPPI), in C++.
//
// Three functions are stubs. Fill them in, then run:
//
//     make test
//
// The self-test reports each exercise separately, so you can finish them one at a time and
// watch the TODOs turn into PASSes. Everything below the three exercises is the harness; it
// is given, and it is worth reading, because it is the shape every sampling planner has.

#include <mujoco/mujoco.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------------------
// Defaults. Every one of these is overridable on the command line, because the whole point
// of the lesson is to let a student move the sample count and the horizon and watch what it
// costs. These particular values were chosen by measurement, not taste: they are the
// cheapest setting that swung the pole up on every seed tried.
// ---------------------------------------------------------------------------------------
constexpr int kDefaultSamples = 200;
constexpr int kDefaultHorizon = 40;
constexpr double kDefaultSigma = 0.5;
constexpr double kDefaultLambda = 30.0;
constexpr int kDefaultTicks = 400;
constexpr unsigned long long kDefaultSeed = 7;
constexpr int kBenchSamples = 4000;

// Cost weights. GIVEN, not an exercise: everyone must optimise the same objective or no two
// runs are comparable. (1 - cos theta) is 0 upright and 2 hanging, and unlike theta^2 it does
// not care which way round the pole spun to get there.
constexpr double kWAngle = 10.0;
constexpr double kWCart = 1.0;
constexpr double kWThetaDot = 0.05;
constexpr double kWXDot = 0.05;
constexpr double kWCtrl = 0.01;

// A tick counts as "upright" when the wrapped pole angle is inside this many radians.
constexpr double kUprightTol = 0.2;
// Success is judged over the last second of the run, not over the whole of it: a swing-up
// spends its first second legitimately far from upright.
constexpr double kScoreWindowSeconds = 1.0;

// ---------------------------------------------------------------------------------------
// Plumbing
// ---------------------------------------------------------------------------------------

// Thrown by the stub bodies below. main() turns it into exit code 2, which the lesson's
// Python wrapper translates into NotImplementedError so the grader prints TODO rather than a
// stack trace. Delete nothing here — just replace the todo() call in each exercise.
struct NotImplemented : std::logic_error {
  using std::logic_error::logic_error;
};

[[maybe_unused]] [[noreturn]] void todo(const char* what, const char* hint) {
  throw NotImplemented(std::string(what) + "() is still a stub — " + hint);
}

// theta wrapped into (-pi, pi]. A swing-up can leave the hinge at 2pi, which is upright.
double wrap_angle(double a) { return std::atan2(std::sin(a), std::cos(a)); }

// The objective, evaluated on the state AFTER a step, with the control that produced it.
[[maybe_unused]] double stage_cost(const mjData* d, double u) {
  const double x = d->qpos[0];
  const double th = d->qpos[1];
  const double xd = d->qvel[0];
  const double thd = d->qvel[1];
  return kWAngle * (1.0 - std::cos(th)) + kWCart * x * x + kWThetaDot * thd * thd +
         kWXDot * xd * xd + kWCtrl * u * u;
}

// Branch a scratch mjData off the live one. This is the line that makes a rollout a
// *prediction* instead of an action: mj_copyData duplicates the whole state, so the stepping
// that follows happens to the copy and the robot's real state never moves.
[[maybe_unused]] void load_state(const mjModel* m, mjData* dst, const mjData* src) {
  mj_copyData(dst, m, src);
}

// =======================================================================================
// EXERCISE 1 — sample_controls()
// =======================================================================================
//
// Fill `out` with `samples` independent control sequences, each `horizon` long, laid out
// row-major: the control for sample i at time t lives at out[i * horizon + t].
//
// Each value is the current plan plus noise, clipped to what the actuator can accept:
//
//     out[i*horizon + t] = clamp(mean[t] + sigma * z, lo, hi),  z ~ Normal(0, 1)
//
// A FRESH z for every (i, t). One draw reused down a row gives you a constant offset; one
// draw reused across rows gives you `samples` identical copies of the same plan, and the
// refit that follows then has nothing to choose between.
//
// Worked example: samples = 2, horizon = 3, mean = {0, 0, 0}, sigma = 0.5, lo/hi = -1/+1.
// `out` ends up 6 long. out[0..2] is the first candidate plan, out[3..5] the second, every
// entry inside [-1, 1], and the two rows differ.
//
// Useful: std::normal_distribution<double> gauss(0.0, 1.0);  then gauss(rng).
//         out.assign(n, 0.0) sizes the buffer. std::clamp(v, lo, hi) is in <algorithm>.
void sample_controls(std::mt19937_64& rng, const std::vector<double>& mean, int samples,
                     int horizon, double sigma, double lo, double hi,
                     std::vector<double>& out) {
  // YOUR CODE HERE
  todo("sample_controls",
       "size `out` to samples*horizon, then for every (i, t) write "
       "clamp(mean[t] + sigma * gauss(rng), lo, hi) into out[i*horizon + t], drawing a fresh "
       "standard normal each time");
}

// =======================================================================================
// EXERCISE 2 — rollout_cost()
// =======================================================================================
//
// Score one candidate plan by simulating it. Branch `scratch` off `start`, then for each
// t in [0, horizon): write controls[t] into scratch->ctrl[0], call mj_step once, and add
// stage_cost(scratch, controls[t]) to a running total. Return the total.
//
// `start` is const for a reason. If you step `start` instead of `scratch` you have not
// predicted what the robot might do, you have *moved* it — and every later sample then
// starts from a different state, so the costs are not comparable and the planner's own
// rollouts drive the machine. The self-test checks that `start` is untouched.
//
// Worked example: horizon = 2 with controls {0, 0} from the hanging keyframe returns
// stage_cost after one step plus stage_cost after two — a positive number, since the pole
// is nowhere near upright and every weight is non-negative.
//
// Useful: load_state(m, scratch, start) is the branch. mj_step(m, scratch) is one step.
double rollout_cost(const mjModel* m, const mjData* start, mjData* scratch,
                    const double* controls, int horizon) {
  // YOUR CODE HERE
  todo("rollout_cost",
       "branch scratch off start with load_state, then loop t in [0, horizon): set "
       "scratch->ctrl[0] = controls[t], call mj_step(m, scratch), and accumulate "
       "stage_cost(scratch, controls[t]); return the sum");
}

// =======================================================================================
// EXERCISE 3 — refit_mean()
// =======================================================================================
//
// The MPPI update. Turn the costs into weights, then set the new plan to the weighted
// average of the sampled plans:
//
//     w_i  = exp(-(cost_i - min_j cost_j) / lambda)
//     mean[t] = sum_i (w_i / sum_j w_j) * samples_buf[i*horizon + t]
//
// Subtracting the minimum cost before the exponential is not a nicety, it is the whole
// difference between working and not. Costs here run into the hundreds; exp(-300/30) is
// 4.5e-5, but exp(-30000/30) is exactly 0 in double precision. Shift the costs down so the
// best sample always has weight exp(0) = 1 and the sum can never be zero. Subtracting a
// constant from every cost cannot change the answer — the constant cancels between the
// numerator and the denominator — so you lose nothing by doing it.
//
// Worked example: three samples with equal costs give weights 1, 1, 1, so the new mean is
// the plain arithmetic mean of the three sampled plans. One sample far cheaper than the
// others gives weights near 1, 0, 0, so the new mean is essentially that sample.
//
// Useful: std::min_element(costs.begin(), costs.begin() + samples) finds the cheapest.
//         mean.assign(horizon, 0.0) clears the output before you accumulate into it.
void refit_mean(const std::vector<double>& samples_buf, const std::vector<double>& costs,
                int samples, int horizon, double lambda, std::vector<double>& mean) {
  // YOUR CODE HERE
  todo("refit_mean",
       "find the smallest cost, build w_i = exp(-(cost_i - best) / lambda) and their sum, "
       "then set mean[t] to the sum over i of (w_i / wsum) * samples_buf[i*horizon + t]");
}

// =======================================================================================
// Everything below here is the harness. It is given, and it is worth reading: it is the
// shape every sampling planner has.
// =======================================================================================

struct Options {
  std::string model = "assets/cartpole.xml";
  std::string controls;
  std::string costs = "equal";
  int samples = kDefaultSamples;
  int horizon = kDefaultHorizon;
  int ticks = kDefaultTicks;
  double sigma = kDefaultSigma;
  double lambda = kDefaultLambda;
  double offset = 0.0;
  unsigned long long seed = kDefaultSeed;
};

void emit(const char* key, double v) { std::printf("@ %s=%.17g\n", key, v); }
void emit_i(const char* key, long long v) { std::printf("@ %s=%lld\n", key, v); }

mjModel* load_model(const std::string& path) {
  char err[1000] = "";
  mjModel* m = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
  if (!m) throw std::runtime_error("could not load model '" + path + "': " + err);
  if (m->nkey < 1) throw std::runtime_error("model has no keyframe; expected one named 'hanging'");
  return m;
}

double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Deterministic, RNG-free control sequence. Used by `bench` and `refitdump` so that those
// commands depend only on the one exercise they are measuring.
double scripted_control(int i, int t, double lo, double hi) {
  return std::clamp(0.5 * std::sin(0.37 * t + 1.1 * i), lo, hi);
}

// ---------------------------------------------------------------------------------------
// The closed loop: plan, apply one control, shift the plan, repeat.
// ---------------------------------------------------------------------------------------
int cmd_mpc(const Options& o) {
  mjModel* m = load_model(o.model);
  mjData* d = mj_makeData(m);
  mjData* scratch = mj_makeData(m);
  mj_resetDataKeyframe(m, d, 0);
  mj_forward(m, d);

  const double lo = m->actuator_ctrlrange[0];
  const double hi = m->actuator_ctrlrange[1];
  const double dt = m->opt.timestep;

  std::mt19937_64 rng(o.seed);
  std::vector<double> mean(static_cast<size_t>(o.horizon), 0.0);
  std::vector<double> buf;
  std::vector<double> costs(static_cast<size_t>(o.samples), 0.0);
  std::vector<double> trace_t, trace_x, trace_th;
  trace_t.reserve(static_cast<size_t>(o.ticks));

  long long steps = 0;
  const double t0 = now_seconds();
  for (int k = 0; k < o.ticks; ++k) {
    sample_controls(rng, mean, o.samples, o.horizon, o.sigma, lo, hi, buf);
    for (int i = 0; i < o.samples; ++i) {
      costs[static_cast<size_t>(i)] =
          rollout_cost(m, d, scratch, &buf[static_cast<size_t>(i) * o.horizon], o.horizon);
      steps += o.horizon;
    }
    refit_mean(buf, costs, o.samples, o.horizon, o.lambda, mean);

    // Apply only the first control of the freshly refitted plan. That is what makes this
    // *model-predictive* control rather than open-loop trajectory optimisation: the other
    // horizon-1 controls are thrown away and recomputed from the new state next tick.
    d->ctrl[0] = std::clamp(mean[0], lo, hi);
    mj_step(m, d);
    trace_t.push_back(d->time);
    trace_x.push_back(d->qpos[0]);
    trace_th.push_back(d->qpos[1]);

    // Warm start: this tick's plan for t+1 is next tick's plan for t. Shift left and leave
    // a zero at the end. Re-planning from scratch every tick throws away the search.
    for (int t = 0; t + 1 < o.horizon; ++t) mean[static_cast<size_t>(t)] = mean[static_cast<size_t>(t) + 1];
    mean[static_cast<size_t>(o.horizon) - 1] = 0.0;
  }
  const double wall = now_seconds() - t0;

  const double end_time = trace_t.empty() ? 0.0 : trace_t.back();
  int in_window = 0, upright = 0;
  double cos_sum = 0.0, max_abs_x = 0.0;
  for (size_t i = 0; i < trace_t.size(); ++i) {
    max_abs_x = std::max(max_abs_x, std::fabs(trace_x[i]));
    if (trace_t[i] >= end_time - kScoreWindowSeconds) {
      ++in_window;
      cos_sum += std::cos(trace_th[i]);
      if (std::fabs(wrap_angle(trace_th[i])) < kUprightTol) ++upright;
    }
  }
  const double upright_fraction = in_window ? static_cast<double>(upright) / in_window : 0.0;
  const double final_abs_theta = trace_th.empty() ? 0.0 : std::fabs(wrap_angle(trace_th.back()));

  emit_i("samples", o.samples);
  emit_i("horizon", o.horizon);
  emit_i("ticks", o.ticks);
  emit("sigma", o.sigma);
  emit("lambda", o.lambda);
  emit_i("seed", static_cast<long long>(o.seed));
  emit("timestep", dt);
  emit("horizon_seconds", o.horizon * dt);
  emit("control_hz", 1.0 / dt);
  emit_i("planning_steps", steps);
  emit("wall_seconds", wall);
  emit("steps_per_second", wall > 0 ? steps / wall : 0.0);
  emit("rollouts_per_second", wall > 0 ? static_cast<double>(o.ticks) * o.samples / wall : 0.0);
  emit("plan_seconds_per_tick", o.ticks ? wall / o.ticks : 0.0);
  emit("sim_seconds", end_time);
  emit("final_abs_theta", final_abs_theta);
  emit("upright_fraction", upright_fraction);
  emit("mean_cos_theta", in_window ? cos_sum / in_window : 0.0);
  emit("max_abs_cart", max_abs_x);
  emit_i("swung_up", (upright_fraction >= 0.8 && final_abs_theta < kUprightTol) ? 1 : 0);

  mj_deleteData(scratch);
  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}

// ---------------------------------------------------------------------------------------
// Throughput of the rollout loop alone: no sampling, no refit, no closed loop. Depends only
// on rollout_cost, so it measures the one thing it claims to measure.
// ---------------------------------------------------------------------------------------
int cmd_bench(const Options& o) {
  mjModel* m = load_model(o.model);
  mjData* d = mj_makeData(m);
  mjData* scratch = mj_makeData(m);
  mj_resetDataKeyframe(m, d, 0);
  mj_forward(m, d);
  const double lo = m->actuator_ctrlrange[0];
  const double hi = m->actuator_ctrlrange[1];

  std::vector<double> buf(static_cast<size_t>(o.samples) * o.horizon);
  for (int i = 0; i < o.samples; ++i)
    for (int t = 0; t < o.horizon; ++t)
      buf[static_cast<size_t>(i) * o.horizon + t] = scripted_control(i, t, lo, hi);

  double checksum = 0.0;
  const double t0 = now_seconds();
  for (int i = 0; i < o.samples; ++i)
    checksum += rollout_cost(m, d, scratch, &buf[static_cast<size_t>(i) * o.horizon], o.horizon);
  const double wall = now_seconds() - t0;
  const long long steps = static_cast<long long>(o.samples) * o.horizon;

  emit_i("rollouts", o.samples);
  emit_i("horizon", o.horizon);
  emit_i("steps", steps);
  emit("wall_seconds", wall);
  emit("steps_per_second", wall > 0 ? steps / wall : 0.0);
  emit("rollouts_per_second", wall > 0 ? o.samples / wall : 0.0);
  emit("checksum", checksum);

  mj_deleteData(scratch);
  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}

// ---------------------------------------------------------------------------------------
// Score one control sequence read from a file. The bridge used to prove that the C++ rollout
// and the Python rollout are the same computation, not merely similar ones.
// ---------------------------------------------------------------------------------------
int cmd_costof(const Options& o) {
  if (o.controls.empty()) throw std::runtime_error("costof needs --controls <file>");
  std::ifstream in(o.controls);
  if (!in) throw std::runtime_error("cannot read controls file '" + o.controls + "'");
  std::vector<double> u;
  double v;
  while (in >> v) u.push_back(v);
  if (u.empty()) throw std::runtime_error("controls file '" + o.controls + "' held no numbers");

  mjModel* m = load_model(o.model);
  mjData* d = mj_makeData(m);
  mjData* scratch = mj_makeData(m);
  mj_resetDataKeyframe(m, d, 0);
  mj_forward(m, d);

  const double cost = rollout_cost(m, d, scratch, u.data(), static_cast<int>(u.size()));

  emit_i("horizon", static_cast<long long>(u.size()));
  emit("cost", cost);
  emit("final_theta", scratch->qpos[1]);
  emit("final_x", scratch->qpos[0]);
  // The live state must be exactly where it started.
  emit("start_theta_after", d->qpos[1]);
  emit("start_time_after", d->time);

  mj_deleteData(scratch);
  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}

// ---------------------------------------------------------------------------------------
// Dump a sampled control buffer so the statistics can be checked from outside.
// ---------------------------------------------------------------------------------------
int cmd_sampledump(const Options& o) {
  mjModel* m = load_model(o.model);
  const double lo = m->actuator_ctrlrange[0];
  const double hi = m->actuator_ctrlrange[1];
  std::mt19937_64 rng(o.seed);
  std::vector<double> mean(static_cast<size_t>(o.horizon), o.offset);
  std::vector<double> buf;
  sample_controls(rng, mean, o.samples, o.horizon, o.sigma, lo, hi, buf);

  emit_i("samples", o.samples);
  emit_i("horizon", o.horizon);
  emit("lo", lo);
  emit("hi", hi);
  emit_i("size", static_cast<long long>(buf.size()));
  for (int i = 0; i < o.samples; ++i) {
    std::printf("row");
    for (int t = 0; t < o.horizon; ++t)
      std::printf(" %.17g", buf[static_cast<size_t>(i) * o.horizon + t]);
    std::printf("\n");
  }
  mj_deleteModel(m);
  return 0;
}

// ---------------------------------------------------------------------------------------
// Run refit_mean on a fixed, RNG-free sample set so its invariants can be checked directly.
// --costs equal | spike | graded, each shifted up by --offset.
// ---------------------------------------------------------------------------------------
int cmd_refitdump(const Options& o) {
  mjModel* m = load_model(o.model);
  const double lo = m->actuator_ctrlrange[0];
  const double hi = m->actuator_ctrlrange[1];

  std::vector<double> buf(static_cast<size_t>(o.samples) * o.horizon);
  for (int i = 0; i < o.samples; ++i)
    for (int t = 0; t < o.horizon; ++t)
      buf[static_cast<size_t>(i) * o.horizon + t] = scripted_control(i, t, lo, hi);

  std::vector<double> costs(static_cast<size_t>(o.samples), 0.0);
  for (int i = 0; i < o.samples; ++i) {
    double c = 0.0;
    if (o.costs == "equal") c = 1.0;
    else if (o.costs == "spike") c = (i == 0) ? 0.0 : 1.0e9;
    else if (o.costs == "graded") c = static_cast<double>(i);
    else throw std::runtime_error("--costs must be equal, spike or graded");
    costs[static_cast<size_t>(i)] = c + o.offset;
  }

  std::vector<double> mean;
  refit_mean(buf, costs, o.samples, o.horizon, o.lambda, mean);

  emit_i("samples", o.samples);
  emit_i("horizon", o.horizon);
  emit("lo", lo);
  emit("hi", hi);
  std::printf("mean");
  for (int t = 0; t < o.horizon; ++t) std::printf(" %.17g", mean[static_cast<size_t>(t)]);
  std::printf("\n");
  // The arithmetic (unweighted) mean of the same sample set, for the equal-cost comparison.
  std::printf("arithmetic");
  for (int t = 0; t < o.horizon; ++t) {
    double s = 0.0;
    for (int i = 0; i < o.samples; ++i) s += buf[static_cast<size_t>(i) * o.horizon + t];
    std::printf(" %.17g", s / o.samples);
  }
  std::printf("\n");
  std::printf("first");
  for (int t = 0; t < o.horizon; ++t) std::printf(" %.17g", buf[static_cast<size_t>(t)]);
  std::printf("\n");
  mj_deleteModel(m);
  return 0;
}

// ---------------------------------------------------------------------------------------
// Your instant feedback. Each check runs independently; an unfinished exercise is reported
// as TODO and does not stop the others from running.
// ---------------------------------------------------------------------------------------
struct Check {
  const char* name;
  int status;  // 0 pass, 1 fail, 2 todo
  std::string msg;
};

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}

template <typename F>
Check run_check(const char* name, F body) {
  try {
    body();
    return {name, 0, ""};
  } catch (const NotImplemented& e) {
    return {name, 2, e.what()};
  } catch (const std::exception& e) {
    return {name, 1, e.what()};
  }
}

int cmd_selftest(const Options& o) {
  mjModel* m = load_model(o.model);
  const double lo = m->actuator_ctrlrange[0];
  const double hi = m->actuator_ctrlrange[1];
  std::vector<Check> results;

  // ---- exercise 1 -------------------------------------------------------------------
  results.push_back(run_check("sample_controls", [&] {
    const int S = 400, H = 8;
    const double sigma = 0.4, mu = 0.25;
    std::mt19937_64 rng(1234);
    std::vector<double> mean(static_cast<size_t>(H), mu), buf;
    sample_controls(rng, mean, S, H, sigma, lo, hi, buf);

    require(buf.size() == static_cast<size_t>(S) * H,
            "out has " + std::to_string(buf.size()) + " entries, expected samples*horizon = " +
                std::to_string(S * H) + " — size the buffer to the whole block, not one row");

    for (double v : buf)
      require(v >= lo - 1e-12 && v <= hi + 1e-12,
              "a sampled control landed outside the actuator's ctrlrange — clamp each value "
              "to [lo, hi] after adding the noise, not before");

    bool rows_differ = false;
    for (int t = 0; t < H && !rows_differ; ++t)
      if (buf[static_cast<size_t>(t)] != buf[static_cast<size_t>(H) + t]) rows_differ = true;
    require(rows_differ,
            "every sampled plan is identical — you drew one noise value and reused it for all "
            "samples; draw a fresh standard-normal for each (i, t)");

    bool varies_in_time = false;
    for (int t = 1; t < H && !varies_in_time; ++t)
      if (buf[static_cast<size_t>(t)] != buf[0]) varies_in_time = true;
    require(varies_in_time,
            "every entry within one plan is identical — you drew one noise value per sample "
            "and held it across the horizon; the draw is per (i, t)");

    double sum = 0.0, sq = 0.0;
    for (double v : buf) { sum += v; sq += v * v; }
    const double n = static_cast<double>(buf.size());
    const double emp_mean = sum / n;
    const double emp_sd = std::sqrt(sq / n - emp_mean * emp_mean);
    require(std::fabs(emp_mean - mu) < 0.05,
            "the sample mean came out at " + std::to_string(emp_mean) + " but the plan being "
            "perturbed was " + std::to_string(mu) + " — add mean[t]; noise alone is not a plan");
    require(std::fabs(emp_sd - sigma) < 0.08,
            "the sample spread came out at " + std::to_string(emp_sd) + " against sigma = " +
                std::to_string(sigma) + " — multiply a standard normal by sigma (a normal_"
                "distribution's second argument is a standard deviation, and a uniform draw "
                "has the wrong shape entirely)");

    std::mt19937_64 rng2(1234);
    std::vector<double> buf2;
    sample_controls(rng2, mean, S, H, sigma, lo, hi, buf2);
    require(buf == buf2,
            "the same seed produced a different buffer — draw from the rng you were handed, "
            "not from a fresh one seeded off the clock");
  }));

  // ---- exercise 2 -------------------------------------------------------------------
  results.push_back(run_check("rollout_cost", [&] {
    mjData* d = mj_makeData(m);
    mjData* scratch = mj_makeData(m);
    mj_resetDataKeyframe(m, d, 0);
    mj_forward(m, d);

    const int H = 20;
    std::vector<double> zero(static_cast<size_t>(H) * 2, 0.0);
    std::vector<double> hard(static_cast<size_t>(H), 1.0);

    const double th_before = d->qpos[1];
    const double t_before = d->time;
    const double c_short = rollout_cost(m, d, scratch, zero.data(), H);

    require(d->qpos[1] == th_before && d->time == t_before,
            "the live mjData moved during scoring — you stepped `start` instead of the "
            "scratch copy, so planning is now driving the robot");
    require(std::isfinite(c_short), "the cost is not a finite number");
    require(c_short > 0.0,
            "a zero-control rollout from the hanging keyframe scored " +
                std::to_string(c_short) + " — every weight is non-negative and the pole is "
                "hanging, so the total must be positive; did you return before accumulating?");

    const double again = rollout_cost(m, d, scratch, zero.data(), H);
    require(again == c_short,
            "scoring the same plan twice gave two different answers — the scratch state is "
            "leaking between rollouts; branch it from `start` at the top of every call");

    const double c_long = rollout_cost(m, d, scratch, zero.data(), 2 * H);
    require(c_long > c_short,
            "doubling the horizon did not increase the cost — you are returning only the last "
            "stage's cost instead of the sum over the horizon");

    const double c_hard = rollout_cost(m, d, scratch, hard.data(), H);
    require(c_hard != c_short,
            "driving the cart at full throttle scored exactly the same as doing nothing — the "
            "controls argument is being ignored; write controls[t] into scratch->ctrl[0]");

    mj_deleteData(scratch);
    mj_deleteData(d);
  }));

  // ---- exercise 3 -------------------------------------------------------------------
  results.push_back(run_check("refit_mean", [&] {
    const int S = 6, H = 5;
    std::vector<double> buf(static_cast<size_t>(S) * H);
    for (int i = 0; i < S; ++i)
      for (int t = 0; t < H; ++t)
        buf[static_cast<size_t>(i) * H + t] = scripted_control(i, t, lo, hi);

    std::vector<double> arithmetic(static_cast<size_t>(H), 0.0);
    for (int t = 0; t < H; ++t) {
      double s = 0.0;
      for (int i = 0; i < S; ++i) s += buf[static_cast<size_t>(i) * H + t];
      arithmetic[static_cast<size_t>(t)] = s / S;
    }

    // equal costs -> plain average
    std::vector<double> equal(static_cast<size_t>(S), 3.0), mean;
    refit_mean(buf, equal, S, H, kDefaultLambda, mean);
    require(mean.size() == static_cast<size_t>(H),
            "the refitted plan has " + std::to_string(mean.size()) + " entries, expected " +
                std::to_string(H));
    for (int t = 0; t < H; ++t)
      require(std::fabs(mean[static_cast<size_t>(t)] - arithmetic[static_cast<size_t>(t)]) < 1e-9,
              "with every cost equal, all weights are equal, so the refit must be the plain "
              "arithmetic mean of the samples — yours is not; check that you divide by the "
              "sum of the weights");

    // one dominant sample -> that sample
    std::vector<double> spike(static_cast<size_t>(S), 1.0e9);
    spike[0] = 0.0;
    refit_mean(buf, spike, S, H, kDefaultLambda, mean);
    for (int t = 0; t < H; ++t)
      require(std::fabs(mean[static_cast<size_t>(t)] - buf[static_cast<size_t>(t)]) < 1e-6,
              "one sample was astronomically cheaper than the rest, so the refit should be "
              "that sample almost exactly — yours is not, so the weights are not falling off "
              "with cost");

    // the stability property: adding a constant to every cost must change nothing
    std::vector<double> graded(static_cast<size_t>(S)), shifted(static_cast<size_t>(S));
    for (int i = 0; i < S; ++i) {
      graded[static_cast<size_t>(i)] = 10.0 * i;
      shifted[static_cast<size_t>(i)] = 10.0 * i + 1.0e6;
    }
    std::vector<double> m_small, m_big;
    refit_mean(buf, graded, S, H, kDefaultLambda, m_small);
    refit_mean(buf, shifted, S, H, kDefaultLambda, m_big);
    for (int t = 0; t < H; ++t) {
      require(std::isfinite(m_big[static_cast<size_t>(t)]),
              "shifting every cost up by 1e6 produced a non-finite plan — exp(-1e6/lambda) is "
              "exactly 0 in double precision, so the weights all vanished and you divided by "
              "zero; subtract the minimum cost before exponentiating");
      require(std::fabs(m_big[static_cast<size_t>(t)] - m_small[static_cast<size_t>(t)]) < 1e-9,
              "adding the same constant to every cost changed the answer — it must cancel "
              "between the weights and their sum; subtract the minimum cost first");
    }
    for (int t = 0; t < H; ++t)
      require(mean[static_cast<size_t>(t)] >= lo - 1e-12 && mean[static_cast<size_t>(t)] <= hi + 1e-12,
              "the refitted plan left the actuator range, which a weighted average of in-range "
              "samples cannot do — the weights are not normalised to sum to 1");
  }));

  // ---- all three together ------------------------------------------------------------
  results.push_back(run_check("closed_loop", [&] {
    Options s = o;
    s.ticks = 40;
    mjData* d = mj_makeData(m);
    mjData* scratch = mj_makeData(m);
    mj_resetDataKeyframe(m, d, 0);
    mj_forward(m, d);
    std::mt19937_64 rng(s.seed);
    std::vector<double> mean(static_cast<size_t>(s.horizon), 0.0), buf;
    std::vector<double> costs(static_cast<size_t>(s.samples), 0.0);
    const double th0 = d->qpos[1];
    for (int k = 0; k < s.ticks; ++k) {
      sample_controls(rng, mean, s.samples, s.horizon, s.sigma, lo, hi, buf);
      for (int i = 0; i < s.samples; ++i)
        costs[static_cast<size_t>(i)] =
            rollout_cost(m, d, scratch, &buf[static_cast<size_t>(i) * s.horizon], s.horizon);
      refit_mean(buf, costs, s.samples, s.horizon, s.lambda, mean);
      require(std::isfinite(mean[0]), "the refitted plan went non-finite during a closed loop");
      d->ctrl[0] = std::clamp(mean[0], lo, hi);
      mj_step(m, d);
      for (int t = 0; t + 1 < s.horizon; ++t)
        mean[static_cast<size_t>(t)] = mean[static_cast<size_t>(t) + 1];
      mean[static_cast<size_t>(s.horizon) - 1] = 0.0;
    }
    require(std::fabs(d->qpos[1] - th0) > 1e-3,
            "40 control ticks moved the pole by less than a milliradian — the planner is "
            "emitting zero control; check that refit_mean writes into `mean`");
    mj_deleteData(scratch);
    mj_deleteData(d);
  }));

  int failed = 0, todos = 0;
  std::printf("\n  F15-L07 self-test (%s)\n\n", o.model.c_str());
  for (const Check& c : results) {
    const char* tag = c.status == 0 ? "PASS " : (c.status == 2 ? "TODO " : "FAIL ");
    std::printf("  [%s] %-16s %s\n", tag, c.name, c.msg.c_str());
    if (c.status == 1) ++failed;
    if (c.status == 2) ++todos;
  }
  std::printf("\n  %zu checks, %d failed, %d not implemented\n\n", results.size(), failed, todos);
  mj_deleteModel(m);
  if (failed) return 1;
  if (todos) return 2;
  return 0;
}

void usage() {
  std::printf(
      "usage: lesson_bin <command> [options]\n\n"
      "commands:\n"
      "  selftest    run the built-in checks for the three exercises\n"
      "  mpc         closed-loop sampling MPC from the 'hanging' keyframe\n"
      "  bench       rollout throughput only (depends on rollout_cost alone)\n"
      "  costof      score one control sequence read from --controls <file>\n"
      "  sampledump  print a sampled control buffer\n"
      "  refitdump   print refit_mean's output on a fixed sample set\n\n"
      "options:\n"
      "  --model F --samples N --horizon H --ticks K --sigma S --lambda L\n"
      "  --seed Z --controls F --costs equal|spike|graded --offset X\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 64;
  }
  const std::string cmd = argv[1];
  Options o;
  if (cmd == "bench") o.samples = kBenchSamples;
  for (int i = 2; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("option " + k + " needs a value");
      return argv[++i];
    };
    if (k == "--model") o.model = next();
    else if (k == "--controls") o.controls = next();
    else if (k == "--costs") o.costs = next();
    else if (k == "--samples") o.samples = std::stoi(next());
    else if (k == "--horizon") o.horizon = std::stoi(next());
    else if (k == "--ticks") o.ticks = std::stoi(next());
    else if (k == "--sigma") o.sigma = std::stod(next());
    else if (k == "--lambda") o.lambda = std::stod(next());
    else if (k == "--offset") o.offset = std::stod(next());
    else if (k == "--seed") o.seed = std::stoull(next());
    else if (k == "--help" || k == "-h") { usage(); return 0; }
    else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 64; }
  }
  if (o.samples < 1 || o.horizon < 1) {
    std::fprintf(stderr, "--samples and --horizon must be >= 1\n");
    return 64;
  }

  try {
    if (cmd == "selftest") return cmd_selftest(o);
    if (cmd == "mpc") return cmd_mpc(o);
    if (cmd == "bench") return cmd_bench(o);
    if (cmd == "costof") return cmd_costof(o);
    if (cmd == "sampledump") return cmd_sampledump(o);
    if (cmd == "refitdump") return cmd_refitdump(o);
    std::fprintf(stderr, "unknown command '%s'\n", cmd.c_str());
    usage();
    return 64;
  } catch (const NotImplemented& e) {
    std::fprintf(stderr, "NOT IMPLEMENTED: %s\n", e.what());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
