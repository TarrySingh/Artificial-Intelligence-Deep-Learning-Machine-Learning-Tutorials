// P04-L06 — Reproducing the first line's number, exactly.
//
// Four functions are stubs. Fill them in, then run:
//
//     make test
//
// The self-test reports each exercise separately, so you can finish them one at a time and
// watch the TODOs turn into PASSes. Everything below the four exercises is the harness; it
// is given, and it is worth reading, because it is the shape of every reconciliation you
// will ever be asked to do.
//
// Moved or copied this checkout? Run `make clean` first. See the common-mistakes section of
// the notebook for why.

/* clock_gettime and CLOCK_MONOTONIC are POSIX, not ISO C. Under -std=c11, glibc (Linux)
   hides them unless asked before the first header; macOS shows them regardless. */
#define _DEFAULT_SOURCE
#include <float.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------------------
// The one constant the whole lesson turns on. Pairwise summation is only reproducible if
// everybody agrees where the recursion stops, so the block size is fixed here and stated in
// the exercise. Change it and your answer stops matching the reference — which is the lesson.
// ---------------------------------------------------------------------------------------
#define PAIRWISE_BLOCK 128

static const uint64_t kSeed = 20260916ULL;
static const uint64_t kGamma = 0x9E3779B97F4A7C15ULL;
static const long kDefaultN = 2000000;
static const long kDefaultDump = 2000;
static const int kExpSpan = 41;    // exponent codes 0..40 ...
static const int kExpShift = 20;   // ... mapped onto 2^-20 .. 2^+20

#define COMMONS_MAYBE_UNUSED __attribute__((unused))

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

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Distance between two doubles counted in representable steps. Used by the self-test to say
// "you are 412 numbers away from the right answer" rather than "not equal". The same
// function is exercise 5, in Python.
static double ulps_between(double a, double b) {
  if (isnan(a) || isnan(b)) return INFINITY;
  if (a == b) return 0.0;  // catches +0.0 vs -0.0, which are equal but differ in bits
  uint64_t ia, ib;
  memcpy(&ia, &a, sizeof ia);
  memcpy(&ib, &b, sizeof ib);
  int64_t ka = (ia & 0x8000000000000000ULL) ? (int64_t)(0x8000000000000000ULL - (ia & 0x7FFFFFFFFFFFFFFFULL))
                                            : (int64_t)ia;
  int64_t kb = (ib & 0x8000000000000000ULL) ? (int64_t)(0x8000000000000000ULL - (ib & 0x7FFFFFFFFFFFFFFFULL))
                                            : (int64_t)ib;
  // Subtract as integers and convert AFTERWARDS. An ordering key for a number of the size
  // this lesson sums is around 4.7e18, which needs 63 bits; converting the two keys to
  // double first rounds each to a multiple of 1024 and a difference of a couple of hundred
  // steps disappears entirely. A measuring instrument made of the thing being measured has
  // to be built carefully, and this is the lesson's own instrument.
  int64_t d = ka - kb;
  return (double)(d < 0 ? -d : d);
}

// =======================================================================================
// The portfolio. SYNTHETIC, generated here, identical in C and in numpy.
// =======================================================================================
//
// Two teams cannot argue about a number unless they are summing the same array, so the array
// is not a file: it is a counter-based pseudo-random stream, and every value is built out of
// exact operations (an integer to a double, then a scaling by a power of two) so that C and
// numpy produce the SAME 64 bits for every element. No file, no download, no rounding
// difference between the two languages to hide behind.
//
// Record i gets three draws: a 53-bit mantissa, an exponent code and a sign from the second,
// and a probability of default from the third.
static uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static uint64_t draw(long k) {
  return mix64(kSeed + (uint64_t)(k + 1) * kGamma);
}

// exposure: a signed position value scaled by a power of two from 2^-20 to 2^+20, with a
// uniform 53-bit mantissa on top, so the smallest magnitudes land well below 2^-20 — the
// notebook prints the range actually generated. A portfolio where a rounding error in the
// big positions can swallow a small position whole.
// pd: a probability of default in [0, 1).
static void generate(long n, double* ex, double* pd) {
  for (long i = 0; i < n; i++) {
    uint64_t u0 = draw(3 * i), u1 = draw(3 * i + 1), u2 = draw(3 * i + 2);
    double mant = (double)(u0 >> 11);                      // exact: 53 bits into a double
    int e = (int)(u1 % (uint64_t)kExpSpan) - kExpShift;
    double sign = (u1 >> 63) ? -1.0 : 1.0;
    ex[i] = sign * ldexp(mant, e - 53);                    // exact: scaling by a power of 2
    pd[i] = ldexp((double)(u2 >> 11), -53);                // exact
  }
}

// =======================================================================================
// EXERCISE 1 — naive_sum()
// =======================================================================================
//
// Add the elements left to right into one accumulator, exactly as a spreadsheet, a for loop
// and most first-line model code do it:
//
//     s = 0; for i in 0..n-1: s = s + x[i]
//
// Nothing clever. This is the baseline the rest of the lesson is measured against, and it is
// what the first line almost certainly ran.
//
// Requirements, all graded:
//   * start the accumulator at 0.0, and return 0.0 for n <= 0;
//   * add in index order, one element at a time, with no reordering and no partial sums;
//   * return the accumulator itself — no compensation, no correction.
//
// Worked example: naive_sum({1e16, 1.0, -1e16}, 3) returns 0.0, because 1e16 + 1.0 rounds
// back to 1e16 (the gap between neighbouring doubles up there is 2.0) and the 1.0 is gone
// before the third element is ever seen. That lost 1.0 is the whole lesson.
static double naive_sum(const double* x, long n) {
  // YOUR CODE HERE
  todo("naive_sum",
       "start an accumulator at 0.0, add x[0], x[1], ... into it in index order, and return "
       "the accumulator");
  return 0.0;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 2 — pairwise_sum()
// =======================================================================================
//
// Split the array in half, sum each half, add the two results. Recurse until a block is
// small enough to sum left to right. Errors then accumulate along a tree of depth log2(n)
// instead of along a chain of length n.
//
// The convention is FIXED, because the point of this lesson is reproducibility and two
// pairwise implementations that split differently do not reproduce each other:
//
//     if n <= PAIRWISE_BLOCK (128):  sum left to right, exactly as naive_sum does
//     otherwise:                     half = n / 2   (integer division, so the LEFT half is
//                                                    the short one when n is odd)
//                                    return pairwise_sum(x, half)
//                                         + pairwise_sum(x + half, n - half)
//
// Return 0.0 for n <= 0.
//
// Worked example: with n = 300 and every element 1.0, the recursion splits 300 into 150 and
// 150, each of those into 75 and 75, and each 75 is a leaf (75 <= 128) summed left to right.
// The answer is 300.0. With n = 129 the split is 64 and 65, and both are leaves.
//
// This one is graded BIT FOR BIT against a reference that follows exactly the rule above.
// A pairwise sum that splits at a power of two, or stops at 64, is a perfectly good
// algorithm and will still be marked wrong here — because a validator who cannot say which
// convention produced a number cannot reproduce it.
static double pairwise_sum(const double* x, long n) {
  // YOUR CODE HERE
  todo("pairwise_sum",
       "return 0.0 for n <= 0; sum left to right when n <= PAIRWISE_BLOCK; otherwise take "
       "half = n / 2 and return pairwise_sum(x, half) + pairwise_sum(x + half, n - half)");
  return 0.0;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 3 — kahan_sum()
// =======================================================================================
//
// Compensated summation: keep a second accumulator holding the part of each addition that
// the first accumulator could not represent, and add it back at the end.
//
// Implement the Kahan-Babuska-Neumaier form, which is the one that survives an addend LARGER
// than the running sum:
//
//     s = 0; c = 0
//     for each x:
//         t = s + x
//         if |s| >= |x|:  c += (s - t) + x        // s is the big one: x lost its low bits
//         else:           c += (x - t) + s        // x is the big one: s lost its low bits
//         s = t
//     return s + c
//
// Both branches are exact in IEEE arithmetic: when one operand dominates, the difference
// between the rounded total and the larger operand recovers the discarded part of the
// smaller one with no rounding of its own.
//
// Requirements, all graded:
//   * both branches, chosen on |s| >= |x| — the original 1965 form without the branch loses
//     the compensation whenever an element is larger than the sum so far;
//   * the compensation is ACCUMULATED across iterations (c += ...), not overwritten;
//   * the return value is s + c, not s;
//   * return 0.0 for n <= 0.
//
// Worked example: kahan_sum({1e16, 1.0, -1e16}, 3) returns exactly 1.0, where naive_sum
// returns 0.0. Walk it: after the first element s = 1e16, c = 0. The second gives
// t = 1e16 (the 1.0 rounds away), and since |s| >= |x| the compensation picks up
// (1e16 - 1e16) + 1.0 = 1.0. The third cancels s to 0.0 and adds nothing to c. The answer
// is s + c = 0.0 + 1.0.
static double kahan_sum(const double* x, long n) {
  // YOUR CODE HERE
  todo("kahan_sum",
       "keep two accumulators s and c; for each element take t = s + x, add (s - t) + x to c "
       "when fabs(s) >= fabs(x) and (x - t) + s to c otherwise, then set s = t; return s + c");
  return 0.0;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 4 — reproduce()
// =======================================================================================
//
// The validator's decision. Two teams report a and b for the same portfolio aggregate over
// n values whose absolute values sum to sum_abs. Is the difference explained by
// floating-point accumulation, or is it a finding?
//
// The only bound you can defend in a report is the worst case for left-to-right summation:
// every one of the n-1 additions can round by up to half an ulp of its own partial sum, and
// no partial sum can exceed sum_abs. That gives
//
//     bound = (n - 1) * u * sum_abs        with u = 2^-53, the unit roundoff
//
// Compute it in exactly that order — (double)(n - 1), then times u, then times
// fabs(sum_abs) — so that your bound is itself reproducible.
//
// The fabs is load-bearing. sum_abs is a magnitude, and a caller who hands you the signed net
// total by mistake — which is the figure sitting next to it in every reconciliation pack —
// would otherwise get a NEGATIVE bound, and `gap <= bound` is false for every gap there. The
// function would report a finding on every reconciliation it was ever asked about, including
// the ones that reconcile, and it would look like it was working.
//
// Fill in the Reproduction struct:
//   * gap     = fabs(a - b)
//   * bound   = the expression above; 0.0 when n <= 1 (no additions, nothing to blame)
//   * verdict = kReproduced (0) when a == b exactly — bit for bit, the only answer that
//                            actually closes a validation finding;
//               kExplained  (1) when gap <= bound — inclusive, a gap sitting exactly on the
//                            bound is still inside it;
//               kFinding    (2) otherwise.
//
// A NaN on either side falls through to kFinding on its own: a NaN is equal to nothing, and
// no comparison against the bound succeeds. Do not special-case it, and do NOT reach for
// fabs(a - b) < 1e-9 or any other constant — a tolerance with no n and no scale in it is a
// number somebody made up.
//
// Worked example: a = 1.0, b = 1.0 + 2^-52, n = 1000, sum_abs = 1.0. The gap is 2.22e-16,
// the bound is 999 * 2^-53 * 1.0 = 1.109e-13, so the verdict is kExplained (1).
enum { kReproduced = 0, kExplained = 1, kFinding = 2 };

typedef struct {
  int verdict;
  double gap;
  double bound;
} Reproduction;

static Reproduction reproduce(double a, double b, long n, double sum_abs) {
  Reproduction r = {kFinding, 0.0, 0.0};
  // YOUR CODE HERE
  todo("reproduce",
       "set r.gap = fabs(a - b); set r.bound to (double)(n - 1) * 0x1p-53 * fabs(sum_abs), "
       "or 0.0 when n <= 1; set r.verdict to kReproduced when a == b, kExplained when "
       "r.gap <= r.bound, and kFinding otherwise");
  return r;  // unreachable; todo() never returns
}

// =======================================================================================
// Given from here down: the harness that drives your four functions.
// =======================================================================================

typedef enum { kSumNaive = 0, kSumPairwise = 1, kSumKahan = 2 } SumMode;

static const char* mode_name(SumMode m) {
  return m == kSumNaive ? "naive" : (m == kSumPairwise ? "pairwise" : "kahan");
}

static const char* verdict_name(int v) {
  return v == kReproduced ? "REPRODUCED" : (v == kExplained ? "EXPLAINED" : "FINDING");
}

static double sum_with(SumMode m, const double* x, long n) {
  if (m == kSumNaive) return naive_sum(x, n);
  if (m == kSumPairwise) return pairwise_sum(x, n);
  return kahan_sum(x, n);
}

// The module-1 shape: a support-weighted mean, numerator and denominator each a sum.
// The products are materialised into their own array first, deliberately: written as
// `s += v[i] * w[i]` the compiler is free to fuse the multiply and the add into one FMA
// instruction with a single rounding, and the answer would then depend on the chip. The
// Makefile also passes -ffp-contract=off. Reproducibility is a build flag as well as an
// algorithm.
static double weighted_mean(SumMode m, const double* values, const double* weights, long n) {
  double* prod = (double*)malloc((size_t)n * sizeof(double));
  require(prod != NULL, "out of memory allocating %ld products", n);
  for (long i = 0; i < n; i++) prod[i] = values[i] * weights[i];
  double num = sum_with(m, prod, n);
  double den = sum_with(m, weights, n);
  free(prod);
  return num / den;
}

static int cmp_by_magnitude(const void* pa, const void* pb) {
  double a = *(const double*)pa, b = *(const double*)pb;
  double fa = fabs(a), fb = fabs(b);
  if (fa < fb) return -1;
  if (fa > fb) return 1;
  // Exact magnitude ties are broken by sign so the order is total and the sum is the same
  // on every libc, whatever qsort does with equal keys.
  if (signbit(a) && !signbit(b)) return -1;
  if (!signbit(a) && signbit(b)) return 1;
  return 0;
}

typedef struct {
  long n;
  long dump;
  double a, b, sum_abs;
} Options;

// ---------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------

static int cmd_facts(const Options* o) {
  (void)o;
  printf("@ mant_dig=%d\n", DBL_MANT_DIG);
  printf("@ dbl_epsilon=%.17g\n", DBL_EPSILON);
  printf("@ unit_roundoff=%.17g\n", DBL_EPSILON / 2.0);
  printf("@ sizeof_double=%zu\n", sizeof(double));
  // The obvious fix for a summation that loses digits is "accumulate in long double".
  // Whether that buys anything at all is a property of the machine, so the binary
  // reports it rather than the lesson asserting it.
  printf("@ ldbl_mant_dig=%d\n", LDBL_MANT_DIG);
  printf("@ sizeof_long_double=%zu\n", sizeof(long double));
  printf("@ pairwise_block=%d\n", PAIRWISE_BLOCK);
  printf("@ seed=%llu\n", (unsigned long long)kSeed);
  printf("@ default_n=%ld\n", kDefaultN);
  printf("@ exp_lo=%d\n", -kExpShift);
  printf("@ exp_hi=%d\n", kExpSpan - 1 - kExpShift);
  return 0;
}

static int cmd_dump(const Options* o) {
  long k = o->dump;
  double* ex = (double*)malloc((size_t)k * sizeof(double));
  double* pd = (double*)malloc((size_t)k * sizeof(double));
  require(ex && pd, "out of memory");
  generate(k, ex, pd);
  printf("@ dumped=%ld\n", k);
  for (long i = 0; i < k; i++) printf("x %ld %.17g %.17g\n", i, ex[i], pd[i]);
  free(ex);
  free(pd);
  return 0;
}

static int cmd_sums(const Options* o) {
  long n = o->n;
  double* ex = (double*)malloc((size_t)n * sizeof(double));
  double* pd = (double*)malloc((size_t)n * sizeof(double));
  double* mag = (double*)malloc((size_t)n * sizeof(double));
  require(ex && pd && mag, "out of memory for %ld records", n);
  generate(n, ex, pd);
  for (long i = 0; i < n; i++) mag[i] = fabs(ex[i]);

  double t0 = now_seconds();
  double nv = naive_sum(ex, n);
  double t1 = now_seconds();
  double pw = pairwise_sum(ex, n);
  double t2 = now_seconds();
  double kh = kahan_sum(ex, n);
  double t3 = now_seconds();

  printf("@ n=%ld\n", n);
  printf("@ naive=%.17g\n", nv);
  printf("@ pairwise=%.17g\n", pw);
  printf("@ kahan=%.17g\n", kh);
  printf("@ gross_naive=%.17g\n", naive_sum(mag, n));
  printf("@ gross_kahan=%.17g\n", kahan_sum(mag, n));
  printf("@ naive_seconds=%.6f\n", t1 - t0);
  printf("@ pairwise_seconds=%.6f\n", t2 - t1);
  printf("@ kahan_seconds=%.6f\n", t3 - t2);
  free(ex);
  free(pd);
  free(mag);
  return 0;
}

// Same numbers, four orders. Codes: 0 as generated, 1 reversed, 2 ascending by magnitude,
// 3 descending by magnitude.
static int cmd_order(const Options* o) {
  long n = o->n;
  double* ex = (double*)malloc((size_t)n * sizeof(double));
  double* pd = (double*)malloc((size_t)n * sizeof(double));
  double* tmp = (double*)malloc((size_t)n * sizeof(double));
  require(ex && pd && tmp, "out of memory for %ld records", n);
  generate(n, ex, pd);

  printf("@ n=%ld\n", n);
  printf("legend 0=as_generated 1=reversed 2=ascending_magnitude 3=descending_magnitude\n");

  memcpy(tmp, ex, (size_t)n * sizeof(double));
  printf("order 0 %.17g %.17g\n", naive_sum(tmp, n), kahan_sum(tmp, n));

  for (long i = 0; i < n; i++) tmp[i] = ex[n - 1 - i];
  printf("order 1 %.17g %.17g\n", naive_sum(tmp, n), kahan_sum(tmp, n));

  memcpy(tmp, ex, (size_t)n * sizeof(double));
  qsort(tmp, (size_t)n, sizeof(double), cmp_by_magnitude);
  printf("order 2 %.17g %.17g\n", naive_sum(tmp, n), kahan_sum(tmp, n));

  for (long i = 0; i < n / 2; i++) {
    double s = tmp[i];
    tmp[i] = tmp[n - 1 - i];
    tmp[n - 1 - i] = s;
  }
  printf("order 3 %.17g %.17g\n", naive_sum(tmp, n), kahan_sum(tmp, n));

  free(ex);
  free(pd);
  free(tmp);
  return 0;
}

static int cmd_wmean(const Options* o) {
  long n = o->n;
  double* ex = (double*)malloc((size_t)n * sizeof(double));
  double* pd = (double*)malloc((size_t)n * sizeof(double));
  double* mag = (double*)malloc((size_t)n * sizeof(double));
  require(ex && pd && mag, "out of memory for %ld records", n);
  generate(n, ex, pd);
  for (long i = 0; i < n; i++) mag[i] = fabs(ex[i]);
  printf("@ n=%ld\n", n);
  printf("legend 0=%s 1=%s 2=%s\n", mode_name(kSumNaive), mode_name(kSumPairwise),
         mode_name(kSumKahan));
  for (int m = 0; m <= 2; m++)
    printf("wmean %d %.17g\n", m, weighted_mean((SumMode)m, pd, mag, n));
  free(ex);
  free(pd);
  free(mag);
  return 0;
}

static int cmd_reproduce(const Options* o) {
  Reproduction r = reproduce(o->a, o->b, o->n, o->sum_abs);
  printf("@ verdict=%d\n", r.verdict);
  printf("@ gap=%.17g\n", r.gap);
  printf("@ bound=%.17g\n", r.bound);
  printf("verdict_name %s\n", verdict_name(r.verdict));
  return 0;
}

// ---------------------------------------------------------------------------------------
// The self-test: your feedback loop. Every case below is hand-workable.
// ---------------------------------------------------------------------------------------

static void check_naive(void) {
  double easy[3] = {1.0, 2.0, 3.0};
  require(naive_sum(easy, 3) == 6.0, "naive_sum({1,2,3}) gave %.17g, expected 6",
          naive_sum(easy, 3));
  require(naive_sum(easy, 0) == 0.0, "naive_sum with n = 0 must return 0.0, gave %.17g",
          naive_sum(easy, 0));
  double lost[3] = {1e16, 1.0, -1e16};
  double got = naive_sum(lost, 3);
  require(got == 0.0,
          "naive_sum({1e16, 1, -1e16}) gave %.17g, expected 0 — add left to right, one "
          "element at a time, with no compensation: the 1.0 must be lost", got);
  double reordered[3] = {1e16, -1e16, 1.0};
  require(naive_sum(reordered, 3) == 1.0,
          "naive_sum({1e16, -1e16, 1}) gave %.17g, expected 1 — the SAME three numbers in a "
          "different order keep the 1.0, because the big pair cancels before the small one "
          "arrives. Two teams, two orders, two answers: that is the whole lesson.",
          naive_sum(reordered, 3));
}

static void check_pairwise(void) {
  double ones[300];
  for (int i = 0; i < 300; i++) ones[i] = 1.0;
  require(pairwise_sum(ones, 300) == 300.0, "pairwise_sum of 300 ones gave %.17g",
          pairwise_sum(ones, 300));
  require(pairwise_sum(ones, 0) == 0.0, "pairwise_sum with n = 0 must return 0.0");

  // The base case: at or below the block size the result must be the naive one, bit for bit.
  long small = PAIRWISE_BLOCK;
  double* ex = (double*)malloc((size_t)(4 * PAIRWISE_BLOCK) * sizeof(double));
  double* pd = (double*)malloc((size_t)(4 * PAIRWISE_BLOCK) * sizeof(double));
  require(ex && pd, "out of memory");
  generate(4 * PAIRWISE_BLOCK, ex, pd);
  require(pairwise_sum(ex, small) == naive_sum(ex, small),
          "with n = %ld (<= PAIRWISE_BLOCK) pairwise_sum must sum left to right and match "
          "naive_sum exactly; you returned %.17g against naive's %.17g", small,
          pairwise_sum(ex, small), naive_sum(ex, small));

  // The first split: n = 2*BLOCK must be exactly naive(left half) + naive(right half).
  long two = 2 * PAIRWISE_BLOCK;
  double want = naive_sum(ex, two / 2) + naive_sum(ex + two / 2, two / 2);
  require(pairwise_sum(ex, two) == want,
          "with n = %ld pairwise_sum must split at n/2 and return naive(first half) + "
          "naive(second half) = %.17g; you returned %.17g (%.0f representable steps away)",
          two, want, pairwise_sum(ex, two), ulps_between(pairwise_sum(ex, two), want));

  // An odd length: half = n/2 puts the SHORT half on the left. 255 splits into 127 and 128,
  // and both are leaves, so the whole answer is two naive sums.
  long odd = 2 * PAIRWISE_BLOCK - 1;
  double want_odd = naive_sum(ex, odd / 2) + naive_sum(ex + odd / 2, odd - odd / 2);
  require(pairwise_sum(ex, odd) == want_odd,
          "with n = %ld the split is half = n/2 = %ld on the left and %ld on the right, so "
          "the answer is %.17g; you returned %.17g", odd, odd / 2, odd - odd / 2, want_odd,
          pairwise_sum(ex, odd));
  free(ex);
  free(pd);
}

static void check_kahan(void) {
  double lost[3] = {1e16, 1.0, -1e16};
  double got = kahan_sum(lost, 3);
  require(got == 1.0,
          "kahan_sum({1e16, 1, -1e16}) gave %.17g, expected exactly 1 — this is the case the "
          "Neumaier branch exists for: on the second element the running sum is the larger "
          "operand, on the third it is not", got);
  require(kahan_sum(lost, 0) == 0.0, "kahan_sum with n = 0 must return 0.0");
  double big_first[3] = {1.0, 1e16, -1e16};
  require(kahan_sum(big_first, 3) == 1.0,
          "kahan_sum({1, 1e16, -1e16}) gave %.17g, expected 1 — here the ADDEND is larger "
          "than the running sum, which is exactly the case the 1965 form drops. Branch on "
          "|s| >= |x| and compensate with (x - t) + s in the other direction.",
          kahan_sum(big_first, 3));

  double ones[1000];
  for (int i = 0; i < 1000; i++) ones[i] = 1.0;
  require(kahan_sum(ones, 1000) == 1000.0, "kahan_sum of 1000 ones gave %.17g",
          kahan_sum(ones, 1000));
}

static void check_summation_on_the_portfolio(void) {
  long n = 2000000;
  double* ex = (double*)malloc((size_t)n * sizeof(double));
  double* pd = (double*)malloc((size_t)n * sizeof(double));
  double* rev = (double*)malloc((size_t)n * sizeof(double));
  require(ex && pd && rev, "out of memory");
  generate(n, ex, pd);
  for (long i = 0; i < n; i++) rev[i] = ex[n - 1 - i];

  double nv = naive_sum(ex, n), nv_r = naive_sum(rev, n);
  double pw = pairwise_sum(ex, n);
  double kh = kahan_sum(ex, n), kh_r = kahan_sum(rev, n);

  require(ulps_between(nv, nv_r) > 0.0,
          "summing this portfolio forwards and backwards gave the SAME naive answer "
          "(%.17g). On %ld values whose scales span 2^-20 to 2^20 that cannot happen unless "
          "the sum is not really left to right.", nv, n);
  require(ulps_between(kh, kh_r) <= 4.0,
          "compensated summation moved %.0f representable steps when the array was reversed "
          "(%.17g against %.17g); it should move almost nothing. Check that the "
          "compensation accumulates and that you return s + c.",
          ulps_between(kh, kh_r), kh, kh_r);
  require(fabs(pw - kh) <= fabs(nv - kh),
          "pairwise summation (%.17g) is further from the compensated answer (%.17g) than "
          "the naive one is (%.17g). A log-depth tree should beat a length-n chain.",
          pw, kh, nv);
  free(ex);
  free(pd);
  free(rev);
}

static void check_reproduce(void) {
  Reproduction same = reproduce(1.25, 1.25, 1000, 1.0);
  require(same.verdict == kReproduced,
          "identical figures must return kReproduced (0), you returned %d (%s)",
          same.verdict, verdict_name(same.verdict));
  require(same.gap == 0.0, "identical figures must give gap 0.0, you returned %.17g",
          same.gap);

  double one_ulp = 1.0 + 0x1p-52;
  Reproduction close = reproduce(1.0, one_ulp, 1000, 1.0);
  require(close.verdict == kExplained,
          "a one-ulp gap over 1000 additions of values summing to 1.0 is inside the bound "
          "999 * 2^-53 * 1.0 = %.17g, so the verdict is kExplained (1); you returned %d (%s) "
          "with bound %.17g", 999.0 * 0x1p-53, close.verdict, verdict_name(close.verdict),
          close.bound);
  require(fabs(close.bound - 999.0 * 0x1p-53) == 0.0,
          "bound was %.17g, expected (n-1) * 2^-53 * sum_abs = %.17g", close.bound,
          999.0 * 0x1p-53);

  Reproduction far = reproduce(1.0, 1.5, 1000, 1.0);
  require(far.verdict == kFinding,
          "half a unit apart on a portfolio whose absolute values sum to 1.0 is not "
          "floating point; expected kFinding (2), you returned %d (%s)", far.verdict,
          verdict_name(far.verdict));

  Reproduction single = reproduce(1.0, 1.0 + 0x1p-52, 1, 1e300);
  require(single.bound == 0.0 && single.verdict == kFinding,
          "with n = 1 there are no additions, so the bound is 0.0 and any difference is a "
          "finding; you returned bound %.17g and verdict %d (%s)", single.bound,
          single.verdict, verdict_name(single.verdict));

  // Exactly on the bound: (3-1) * 2^-53 * 2^53 = 2.0, with no rounding anywhere.
  Reproduction edge = reproduce(0.0, 2.0, 3, 0x1p53);
  require(edge.bound == 2.0, "bound was %.17g, expected exactly 2.0", edge.bound);
  require(edge.verdict == kExplained,
          "a gap sitting exactly ON the bound is inside it — the comparison is <=, not <. "
          "You returned %d (%s).", edge.verdict, verdict_name(edge.verdict));
  Reproduction over = reproduce(0.0, nextafter(2.0, 3.0), 3, 0x1p53);
  require(over.verdict == kFinding,
          "one representable step past the bound is a finding; you returned %d (%s)",
          over.verdict, verdict_name(over.verdict));

  Reproduction neg = reproduce(1.0, 1.0 + 0x1p-52, 1000, -1.0);
  require(neg.bound == close.bound && neg.verdict == kExplained,
          "a negative sum_abs must give the same bound as its magnitude (%.17g) and the same "
          "verdict; you returned bound %.17g and verdict %d (%s). A bound is a size, and a "
          "negative one makes gap <= bound false for every gap — every reconciliation would "
          "come back a finding.", close.bound, neg.bound, neg.verdict,
          verdict_name(neg.verdict));

  Reproduction nan_case = reproduce(NAN, NAN, 1000, 1.0);
  require(nan_case.verdict == kFinding,
          "a NaN never reproduces anything: NaN == NaN is false and no comparison against "
          "the bound succeeds, so the verdict is kFinding (2). You returned %d (%s).",
          nan_case.verdict, verdict_name(nan_case.verdict));
}

static int run_check(const char* name, void (*fn)(void)) {
  g_jmp_active = 1;
  int rc = setjmp(g_jmp);
  if (rc == 0) {
    fn();
    g_jmp_active = 0;
    printf("  PASS  %s\n", name);
    return 0;
  }
  g_jmp_active = 0;
  if (rc == 1) {
    printf("  TODO  %s: %s\n", name, g_msg);
    return 2;
  }
  printf("  FAIL  %s: %s\n", name, g_msg);
  return 1;
}

static int cmd_selftest(const Options* o) {
  (void)o;
  int todos = 0, fails = 0;
  struct { const char* name; void (*fn)(void); } checks[] = {
      {"exercise 1  naive_sum", check_naive},
      {"exercise 2  pairwise_sum", check_pairwise},
      {"exercise 3  kahan_sum", check_kahan},
      {"exercises 1-3 on the portfolio", check_summation_on_the_portfolio},
      {"exercise 4  reproduce", check_reproduce},
  };
  for (size_t i = 0; i < sizeof checks / sizeof checks[0]; i++) {
    int rc = run_check(checks[i].name, checks[i].fn);
    if (rc == 1) fails++;
    if (rc == 2) todos++;
  }
  printf("\n  %d failed, %d still a stub\n", fails, todos);
  if (fails) return 1;
  if (todos) return 2;
  return 0;
}

// ---------------------------------------------------------------------------------------

static int usage(void) {
  fprintf(stderr,
          "usage: lesson_bin <command> [options]\n"
          "  selftest    run the built-in checks for the four exercises\n"
          "  facts       print what this machine's doubles are made of\n"
          "  dump        print the first --dump records at full precision\n"
          "  sums        naive, pairwise and compensated totals over --n records\n"
          "  order       the same values summed in four different orders\n"
          "  wmean       the exposure-weighted mean PD under each summation\n"
          "  reproduce   --a A --b B --n N --sum-abs S\n"
          "options: --n --dump --a --b --sum-abs\n");
  return 3;
}

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  Options o = {kDefaultN, kDefaultDump, 0.0, 0.0, 0.0};
  const char* cmd = argv[1];
  for (int i = 2; i < argc; i++) {
    const char* k = argv[i];
    if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", k); return 3; }
    const char* v = argv[++i];
    if (!strcmp(k, "--n")) o.n = strtol(v, NULL, 10);
    else if (!strcmp(k, "--dump")) o.dump = strtol(v, NULL, 10);
    else if (!strcmp(k, "--a")) o.a = strtod(v, NULL);
    else if (!strcmp(k, "--b")) o.b = strtod(v, NULL);
    else if (!strcmp(k, "--sum-abs")) o.sum_abs = strtod(v, NULL);
    else { fprintf(stderr, "error: unknown option %s\n", k); return 3; }
  }
  if (!strcmp(cmd, "selftest")) return cmd_selftest(&o);
  if (!strcmp(cmd, "facts")) return cmd_facts(&o);
  if (!strcmp(cmd, "dump")) return cmd_dump(&o);
  if (!strcmp(cmd, "sums")) return cmd_sums(&o);
  if (!strcmp(cmd, "order")) return cmd_order(&o);
  if (!strcmp(cmd, "wmean")) return cmd_wmean(&o);
  if (!strcmp(cmd, "reproduce")) return cmd_reproduce(&o);
  return usage();
}
