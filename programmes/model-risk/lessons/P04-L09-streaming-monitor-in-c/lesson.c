// P04-L09 — Monitoring a portfolio that does not fit in memory.
//
// Five functions are stubs. Fill them in, then run:
//
//     make test
//
// The self-test reports each exercise separately, so you can finish them one at a time and
// watch the TODOs turn into PASSes. Everything below the five exercises is the harness; it
// is given, and it is worth reading, because the way it MEASURES your monitor is half of what
// this lesson teaches.
//
// Moved or copied this checkout? Run `make clean` first. See the common-mistakes section of
// the notebook for why.

// Both of these only switch declarations ON. Under -std=c11 the C library hides everything
// that is not ISO C — fork, wait4, getrusage, fstat — and these two ask glibc (Linux) and the
// macOS headers respectively to show the POSIX and BSD calls this harness needs.
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

// ---------------------------------------------------------------------------------------
// The contract. Every constant here is mirrored in the notebook, which checks that the two
// agree before it trusts a single number from this binary.
// ---------------------------------------------------------------------------------------
#define STREAM_BUFFER_BYTES 65536  // the monitor's whole allowance for data in flight
#define RECORD_BYTES 16            // one record: uint64 account id, then a float64 score
#define MAX_BINS 64                // the most bins a baseline profile may declare
#define RESERVOIR_MAX 4096         // the largest tail sample the harness will hold
#define CHECKPOINT_EVERY 262144    // records read between two running-PSI checkpoints

static const double kPsiFloor = 1e-6;      // module 1's PSI_FLOOR, unchanged
static const double kPsiThreshold = 0.25;  // module 1's PSI_THRESHOLD, unchanged
static const uint64_t kGamma = 0x9E3779B97F4A7C15ULL;

#define LESSON_MAYBE_UNUSED __attribute__((unused))

// =======================================================================================
// Plumbing for the self-test. Read it once, then ignore it.
// =======================================================================================
//
// C has no exceptions, so an unfinished exercise reports itself with setjmp/longjmp: todo()
// records a message and jumps back to the check that called it. Outside the self-test it
// prints the message and exits 2, which the notebook's Python wrapper turns into
// NotImplementedError, so the grader prints TODO rather than a stack trace. A real failure
// jumps with a different code and becomes exit 1.
static jmp_buf g_jmp;
static int g_jmp_active = 0;
static char g_msg[1024];

static LESSON_MAYBE_UNUSED void todo(const char* what, const char* hint) {
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

// =======================================================================================
// The data structures. Given.
// =======================================================================================

// One record of the month's extract, as it sits in the file: 16 bytes, little-endian.
typedef struct {
  uint64_t account_id;
  double score;
} Record;

// Decode one record from 16 bytes. memcpy rather than a pointer cast, because a record that
// starts at an arbitrary offset in a byte buffer is not guaranteed to be aligned for a double.
static LESSON_MAYBE_UNUSED Record decode_record(const unsigned char* p) {
  Record r;
  memcpy(&r.account_id, p, 8);
  memcpy(&r.score, p + 8, 8);
  return r;
}

// The baseline profile: module 1's edges, cut ONCE on the development sample, and the number
// of development records that fell in each bin. It travels to every monthly run as a file;
// the monitor never re-cuts it.
typedef struct {
  int n_bins;
  double edges[MAX_BINS + 1];  // bin i is [edges[i], edges[i+1]); the last bin is closed
  long long counts[MAX_BINS];  // development records per bin
  long long n;                 // development records in total
  int tail_bin;                // the tail is every valid record in this bin or above
} Profile;

// The reservoir: a uniform random sample of k tail records, kept in O(k) memory however long
// the tail turns out to be. The generator is splitmix64 (see splitmix64_next below), seeded
// once, so the sample is reproducible bit for bit — in C and in the notebook's Python.
typedef struct {
  int k;                          // capacity, fixed before the stream starts
  long long seen;                 // tail records offered so far
  uint64_t state;                 // splitmix64 state
  uint64_t ids[RESERVOIR_MAX];    // slot j holds one sampled record's account id ...
  double scores[RESERVOIR_MAX];   // ... and its score
} Reservoir;

// Everything the monitor keeps. Its size does not depend on the length of the stream: that
// is the whole point, and the harness measures it.
typedef struct {
  long long records_read;      // every COMPLETE record taken off the stream
  long long valid;             // records whose score is a probability, 0 <= s <= 1
  long long rejected;          // every other record — NaN included
  long long counts[MAX_BINS];  // valid records per baseline bin
  long long truncated_bytes;   // bytes at the end of the stream that made no whole record
  Reservoir tail;              // the tail sample
  // Maintained by monitor_checkpoint(), which is given:
  int checkpoints;
  long long first_breach_at;   // records_read at the first breaching checkpoint, -1 if none
  int first_breach_bin;        // the bin that breach named, -1 if none
} Monitor;

// The ONLY place a record may sit while the monitor reads it. Static, fixed, and small: the
// month's extract is many times this size, and it goes through here a buffer at a time.
static LESSON_MAYBE_UNUSED unsigned char g_buffer[STREAM_BUFFER_BYTES];

// splitmix64 — Sebastiano Vigna's public-domain generator, the same four lines as his
// splitmix64.c (claims.yaml). Given, in C and in the notebook, so that both sides draw the
// SAME sequence and the reservoir can be compared bit for bit.
static uint64_t splitmix64_next(uint64_t* state) {
  uint64_t z = (*state += kGamma);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static void reservoir_init(Reservoir* r, int k, uint64_t seed) {
  memset(r, 0, sizeof *r);
  r->k = k;
  r->seen = 0;
  r->state = seed;
}

static void monitor_init(Monitor* m, int k, uint64_t seed) {
  memset(m, 0, sizeof *m);
  reservoir_init(&m->tail, k, seed);
  m->first_breach_at = -1;
  m->first_breach_bin = -1;
}

// =======================================================================================
// EXERCISE 2 — bin_index()
// =======================================================================================
//
// Which baseline bin does a score fall in? Module 1 answered this in numpy, and a monitor
// that answers it differently is not monitoring the same thing:
//
//     idx = clip(searchsorted(edges, x, side="right") - 1, 0, n_bins - 1)
//
// In words: count the edges that are <= x, subtract one, and clip into [0, n_bins - 1]. So
//   * bin i is the half-open interval [edges[i], edges[i+1]);
//   * a score sitting EXACTLY on an interior edge belongs to the bin ABOVE that edge;
//   * a score below edges[0] lands in bin 0, and one at or above the last edge lands in the
//     last bin — which is how the last bin comes to be closed on the right.
//
// p->edges holds p->n_bins + 1 strictly increasing values; the outer two may be -inf and
// +inf, which is what module 1's quantile_edges() writes. x is never NaN here: the caller
// rejects anything that is not a probability before it asks which bin it is in.
//
// Worked example: edges {-inf, 0.25, 0.5, 0.75, +inf} (four bins).
//   bin_index(p, 0.1)  = 0     bin_index(p, 0.25) = 1   (on an edge: the bin above)
//   bin_index(p, 0.74) = 2     bin_index(p, 1.0)  = 3
// With FINITE outer edges {0.1, 0.5, 0.9}: bin_index(p, 0.0) = 0 (clipped up) and
// bin_index(p, 0.9) = 1 (the last edge itself: the last bin is closed on the right).
//
// Returns the bin number, 0 <= b < p->n_bins.
static int bin_index(const Profile* p, double x) {
  // YOUR CODE HERE
  todo("bin_index", "the comment above it has module 1's rule and a worked example");
  return 0;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 3 — psi_from_counts()
// =======================================================================================
//
// The population stability index from two vectors of bin COUNTS, floored exactly as module 1
// floors it — the monitor never holds the samples, only their counts, so this is module 1's
// arithmetic moved from samples to counts:
//
//     e_i = expected[i] / n_expected          a_i = actual[i] / n_actual      (shares)
//     e_i = max(e_i, floor)                   a_i = max(a_i, floor)          (BOTH sides)
//     contributions[i] = (a_i - e_i) * log(a_i / e_i)                        (natural log)
//     psi = contributions[0] + contributions[1] + ... , added left to right
//
// Requirements, all graded:
//   * shares, not counts: each side divided by its OWN total, as doubles. `expected[i] /
//     n_expected` on two long longs is INTEGER division, and every share becomes 0;
//   * the floor on BOTH sides, before the logarithm. A bin that emptied out, or appeared
//     from nothing, is the loudest finding PSI has; unfloored it scores inf, and a bin empty
//     on both sides scores 0 * log(0/0) = NaN, which erases the whole index;
//   * fill contributions[0 .. n_bins-1] as well as returning psi — the breach report names a
//     bin, and it can only name one it can see;
//   * if n_expected <= 0 or n_actual <= 0 there are no shares at all: set every contribution
//     to NAN and return NAN.
//
// Worked example: expected {50, 50}, actual {100, 0}, floor 1e-6. Shares e = {0.5, 0.5},
// a = {1.0, 0.0}; the floor lifts a_1 to 1e-6. Contributions:
//     bin 0: (1.0 - 0.5)  * log(1.0 / 0.5)  =  0.34657...
//     bin 1: (1e-6 - 0.5) * log(1e-6 / 0.5) =  6.56116...
// psi = 6.90774... — finite, huge, and pinned on bin 1, which is where the month went wrong.
//
// Returns psi (a double, NaN when either side is empty).
static double psi_from_counts(const long long* expected, long long n_expected,
                              const long long* actual, long long n_actual, int n_bins,
                              double floor, double* contributions) {
  // YOUR CODE HERE
  todo("psi_from_counts", "the comment above it has module 1's arithmetic and a worked "
       "example");
  return 0.0;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 4 — breach_bin()
// =======================================================================================
//
// A breach flag that cannot say WHERE is a flag somebody has to investigate from scratch.
// Decide whether psi breaches the threshold and, if it does, name the triggering bin:
//
//   * a breach is psi STRICTLY greater than threshold. Module 1's rule: a PSI exactly on the
//     threshold is within it;
//   * no breach — including psi being NaN, which is greater than nothing — returns -1;
//   * on a breach, return the bin with the largest contribution; if several bins tie for the
//     largest, return the LOWEST-numbered of them, so two monitors never disagree about
//     which bin to blame.
//
// Worked example: psi = 0.30, contributions {0.05, 0.12, 0.12, 0.01}, threshold 0.25.
// 0.30 > 0.25, so it is a breach; bins 1 and 2 tie for the largest; the answer is 1.
// With psi = 0.25 exactly the answer is -1, whatever the contributions say.
//
// Returns the triggering bin, or -1 when there is no breach.
static int breach_bin(double psi, const double* contributions, int n_bins, double threshold) {
  // YOUR CODE HERE
  todo("breach_bin", "the comment above it has the rule and a worked example");
  return -1;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 5 — reservoir_offer()
// =======================================================================================
//
// A uniform sample of k tail records from a tail whose length nobody knows in advance, in
// O(k) memory. This is Algorithm R (Vitter, 1985 — claims.yaml). The sample is part of the
// monitoring evidence — the analyst pulls those accounts for review — so it must be
// REPRODUCIBLE: rerun the month, get the same accounts. That fixes every detail below.
//
//     let t = r->seen, the number of tail records offered BEFORE this one
//     if t < k:   store this record in slot t              (no draw from the generator)
//     else:       j = splitmix64_next(&r->state) % (t + 1)  (one draw, as an unsigned 64-bit)
//                 if j < k: store this record in slot j, replacing what was there
//     then r->seen = t + 1
//
// "Store in slot j" means r->ids[j] = id and r->scores[j] = score.
//
// Requirements, all graded bit for bit against the same algorithm in Python:
//   * the first k records fill slots 0..k-1 in order and consume NO draws — draw for them and
//     every later draw shifts by k, and the sample silently becomes a different sample;
//   * the modulus is t + 1, the number of records seen INCLUDING this one. `% t` is the
//     classic off-by-one, and it biases the sample towards the end of the month;
//   * replace only when j < k (strictly).
//
// Worked example: k = 2, seed s. Offers of accounts 10, 11, 12 in that order. Accounts 10 and
// 11 go to slots 0 and 1 with no draw. For account 12, t = 2: draw z = splitmix64_next once,
// j = z % 3; if j is 0 or 1, account 12 replaces that slot, if j is 2 it is not kept.
// Either way r->seen ends at 3 and exactly one draw has been used.
//
// Returns nothing; it updates *r.
static void reservoir_offer(Reservoir* r, uint64_t id, double score) {
  // YOUR CODE HERE
  todo("reservoir_offer", "the comment above it has Algorithm R, step by step, and a "
       "worked example");
}

// Given: the running PSI. Called by YOUR monitor_stream() every CHECKPOINT_EVERY records read.
// It scores the month so far against the baseline, reports the checkpoint as a line on
// stdout while the stream is still flowing, and remembers the first checkpoint that breached.
static int g_with_psi = 1;   // off with `monitor --no-psi` or `--counts-only`; see cmd_monitor
static int g_with_tail = 1;  // off with `monitor --no-tail` or `--counts-only`
static LESSON_MAYBE_UNUSED void monitor_checkpoint(const Profile* p, Monitor* m);

// =======================================================================================
// EXERCISE 6 — monitor_stream()
// =======================================================================================
//
// The monthly run, in a single pass, in fixed memory. Read `in` to the end through g_buffer
// and nothing else, and for every complete 16-byte record:
//
//     m->records_read += 1
//     if 0.0 <= score <= 1.0:      it is a probability — bin it with bin_index(), add one to
//                                  m->counts[bin] and to m->valid, and if bin >= p->tail_bin
//                                  offer (account_id, score) to reservoir_offer(&m->tail, ...)
//     else:                        m->rejected += 1       (never binned, never sampled)
//     if m->records_read is a multiple of CHECKPOINT_EVERY: call monitor_checkpoint(p, m)
//
// Reading: fread(g_buffer, 1, STREAM_BUFFER_BYTES, in) returns the number of BYTES read, and
// keeps reading until the buffer is full or the stream ends — so a short read means the end.
// Decode each whole record with decode_record(g_buffer + offset). Bytes left over after the
// last whole record of the LAST read are a record the upstream job never finished writing:
// add their number to m->truncated_bytes rather than dropping them without a word.
//
// Requirements, all graded:
//   * the validity test must reject NaN. `score < 0.0 || score > 1.0` does NOT: every
//     ordered comparison with NaN is false, so NaN sails through it and into a bin. Write
//     the test so that it is true only for a real probability, and let everything else fall
//     to rejected.
//     -0.0 is a probability (it equals 0.0); 1.0 is a probability; 1.0000000000000002 is not;
//   * one pass: the stream is a PIPE and can be read only once — fseek and rewind fail on it;
//   * fixed memory: nothing you keep may grow with the length of the stream. No array of
//     scores, no copy of the tail, no buffer that doubles. The harness measures this;
//   * fread with an element size of 1, as above. fread(g_buffer, RECORD_BYTES, n, in) reads
//     whole records only, and SWALLOWS a partial record at the end without telling you.
//
// Worked example: a stream of five records with scores {0.30, NaN, 1.0, -0.5, 0.9}, then 3
// stray bytes. records_read = 5, valid = 3, rejected = 2 (the NaN and the -0.5),
// truncated_bytes = 3, and the three valid scores are counted in their bins.
//
// Returns 0, or -1 if the stream reported a read error (ferror).
static int monitor_stream(FILE* in, const Profile* p, Monitor* m) {
  // YOUR CODE HERE
  todo("monitor_stream", "the comment above it has the loop, the validity rule and a "
       "worked example");
  return 0;  // unreachable; todo() never returns
}

// =======================================================================================
// Given from here down: the harness that drives your five functions and measures them.
// =======================================================================================

static LESSON_MAYBE_UNUSED void monitor_checkpoint(const Profile* p, Monitor* m) {
  m->checkpoints++;
  if (!g_with_psi) {
    printf("checkpoint %lld %lld nan -1\n", m->records_read, m->valid);
    return;
  }
  double contrib[MAX_BINS];
  double psi = psi_from_counts(p->counts, p->n, m->counts, m->valid, p->n_bins, kPsiFloor,
                               contrib);
  int bin = breach_bin(psi, contrib, p->n_bins, kPsiThreshold);
  if (bin >= 0 && m->first_breach_at < 0) {
    m->first_breach_at = m->records_read;
    m->first_breach_bin = bin;
  }
  printf("checkpoint %lld %lld %.17g %d\n", m->records_read, m->valid, psi, bin);
}

// The final report, printed once the stream has ended.
static void monitor_finish(const Profile* p, Monitor* m, int stream_rc) {
  printf("@ records_read=%lld\n", m->records_read);
  printf("@ valid=%lld\n", m->valid);
  printf("@ rejected=%lld\n", m->rejected);
  printf("@ truncated_bytes=%lld\n", m->truncated_bytes);
  printf("@ stream_error=%d\n", stream_rc != 0);
  printf("@ checkpoints=%d\n", m->checkpoints);
  for (int i = 0; i < p->n_bins; i++) printf("count %d %lld\n", i, m->counts[i]);
  if (g_with_psi) {
    double contrib[MAX_BINS];
    double psi = psi_from_counts(p->counts, p->n, m->counts, m->valid, p->n_bins, kPsiFloor,
                                 contrib);
    int bin = breach_bin(psi, contrib, p->n_bins, kPsiThreshold);
    // The end of the stream counts as a checkpoint for "when did it first breach?", so a
    // month that ends between two checkpoints still dates its breach.
    if (bin >= 0 && m->first_breach_at < 0) {
      m->first_breach_at = m->records_read;
      m->first_breach_bin = bin;
    }
    printf("@ psi=%.17g\n", psi);
    for (int i = 0; i < p->n_bins; i++) printf("contrib %d %.17g\n", i, contrib[i]);
    printf("@ breach_bin=%d\n", bin);
    printf("@ first_breach_at=%lld\n", m->first_breach_at);
    printf("@ first_breach_bin=%d\n", m->first_breach_bin);
  }
  if (g_with_tail) {
    printf("@ tail_seen=%lld\n", m->tail.seen);
    long long kept = m->tail.seen < m->tail.k ? m->tail.seen : m->tail.k;
    for (long long j = 0; j < kept; j++)
      printf("sample %lld %" PRIu64 " %.17g\n", j, m->tail.ids[j], m->tail.scores[j]);
  }
}

// ---------------------------------------------------------------------------------------
// The baseline profile file. A small text file the notebook writes from module 1's edges:
//
//     p04l09-profile 1
//     n_bins 10
//     tail_bin 8
//     edges -inf 0.0206... ... inf
//     counts 20000 20000 ...
//
// Loaded once, before the stream, and validated: a profile with unsorted edges is refused
// rather than monitored against.
// ---------------------------------------------------------------------------------------
static int load_profile(const char* path, Profile* p) {
  FILE* f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open profile %s: %s\n", path, strerror(errno));
    return 0;
  }
  char word[64];
  int version = 0, ok = 1;
  memset(p, 0, sizeof *p);
  p->tail_bin = -1;
  if (fscanf(f, "%63s %d", word, &version) != 2 || strcmp(word, "p04l09-profile") != 0 ||
      version != 1)
    ok = 0;
  while (ok && fscanf(f, "%63s", word) == 1) {
    if (!strcmp(word, "n_bins")) {
      ok = fscanf(f, "%d", &p->n_bins) == 1 && p->n_bins >= 1 && p->n_bins <= MAX_BINS;
    } else if (!strcmp(word, "tail_bin")) {
      ok = fscanf(f, "%d", &p->tail_bin) == 1;
    } else if (!strcmp(word, "edges")) {
      for (int i = 0; ok && i <= p->n_bins; i++) {
        char tok[64];
        char* end = NULL;
        ok = fscanf(f, "%63s", tok) == 1;
        if (ok) p->edges[i] = strtod(tok, &end);
        ok = ok && end && *end == '\0';
      }
    } else if (!strcmp(word, "counts")) {
      p->n = 0;
      for (int i = 0; ok && i < p->n_bins; i++) {
        ok = fscanf(f, "%lld", &p->counts[i]) == 1 && p->counts[i] >= 0;
        if (ok) p->n += p->counts[i];
      }
    } else {
      ok = 0;
    }
  }
  fclose(f);
  for (int i = 0; ok && i < p->n_bins; i++) ok = p->edges[i] < p->edges[i + 1];  // NaN fails
  ok = ok && p->n > 0 && p->tail_bin >= 0 && p->tail_bin <= p->n_bins;
  if (!ok) fprintf(stderr, "error: %s is not a valid profile (bad header, unsorted edges, "
                           "missing counts or a tail_bin out of range)\n", path);
  return ok;
}

// ---------------------------------------------------------------------------------------
// The measurement. Why a worker process, and not simply this process's own getrusage()?
//
// ru_maxrss is a HIGH-WATER MARK, and on Linux the kernel carries a process's mark across
// exec: fs/exec.c folds the address space being replaced into it (claims.yaml quotes the
// line). This binary is started by a Python process holding the whole month in numpy, so its
// OWN mark would start at Python's size before it had allocated a byte. A process that is
// forked AFTER exec has a fresh mark and a copy of this small C image, and nothing else. So
// every measured command runs in a forked worker, and the launcher reports the worker's
// ru_maxrss as wait4() returns it. The unit is the platform's: the notebook normalises it.
// ---------------------------------------------------------------------------------------
typedef int (*WorkerFn)(void* arg);

static int run_in_worker(WorkerFn fn, void* arg) {
  fflush(stdout);  // or the worker's exit() would flush the launcher's buffer a second time
  fflush(stderr);
  pid_t pid = fork();
  if (pid < 0) {
    fprintf(stderr, "error: fork failed: %s\n", strerror(errno));
    return 1;
  }
  if (pid == 0) {
    int rc = fn(arg);
    fflush(stdout);
    exit(rc);
  }
  int status = 0;
  struct rusage worker;
  memset(&worker, 0, sizeof worker);
  if (wait4(pid, &status, 0, &worker) < 0) {
    fprintf(stderr, "error: wait4 failed: %s\n", strerror(errno));
    return 1;
  }
  struct rusage self;
  getrusage(RUSAGE_SELF, &self);
  printf("@ worker_maxrss_raw=%ld\n", (long)worker.ru_maxrss);
  printf("@ launcher_maxrss_raw=%ld\n", (long)self.ru_maxrss);
  if (WIFSIGNALED(status)) {
    fprintf(stderr, "error: the worker was killed by signal %d\n", WTERMSIG(status));
    return 1;
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

static int stdin_is_a_pipe(void) {
  struct stat st;
  return fstat(STDIN_FILENO, &st) == 0 && S_ISFIFO(st.st_mode);
}

static int refuse_seekable_stdin(void) {
  fprintf(stderr,
          "error: the monitor reads its stream from a PIPE on stdin, and stdin here is not "
          "one. A pipe can be read exactly once and cannot seek, which is what makes the "
          "single pass a fact rather than a promise. Feed it like this:\n"
          "    cat month.bin | ./lesson_bin monitor --profile profile.txt\n");
  return 4;
}

// ---------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------

static Profile g_profile;
static Monitor g_monitor;

typedef struct {
  const char* profile;
  int k;
  uint64_t seed;
  long long n;
  long long mib;
  double floor;
  double threshold;
  double psi;
  const char* expected;
  const char* actual;
  const char* contrib;
  int first_value;  // argv index of the first positional value (bins)
} Options;

static int worker_monitor(void* arg) {
  (void)arg;
  int rc = monitor_stream(stdin, &g_profile, &g_monitor);
  monitor_finish(&g_profile, &g_monitor, rc);
  return rc == 0 ? 0 : 1;
}

static int cmd_monitor(const Options* o) {
  if (!stdin_is_a_pipe()) return refuse_seekable_stdin();
  if (!load_profile(o->profile, &g_profile)) return 3;
  // --no-tail switches the tail sample off (no bin is at or above tail_bin), --no-psi the
  // running PSI, --counts-only both. They let each part of exercise 6 be judged on its own,
  // before the exercises it would otherwise wait on exist. The monthly run uses neither.
  if (!g_with_tail) g_profile.tail_bin = g_profile.n_bins;
  monitor_init(&g_monitor, o->k, o->seed);
  return run_in_worker(worker_monitor, NULL);
}

// The process's own baseline: everything the monitor does before the first byte arrives —
// profile loaded, monitor initialised — and then nothing. Your monitor is charged for every
// byte of peak memory above this.
static int worker_baseline(void* arg) {
  (void)arg;
  printf("@ baseline_records=%lld\n", g_monitor.records_read);
  return 0;
}

static int cmd_baseline(const Options* o) {
  if (!load_profile(o->profile, &g_profile)) return 3;
  monitor_init(&g_monitor, o->k, o->seed);
  return run_in_worker(worker_baseline, NULL);
}

// Touch a known number of bytes, so the notebook can find out what unit ru_maxrss is in on
// this machine rather than take anybody's word for it. Every page is written AND read back
// into a printed checksum: an untouched page is never charged to the process, and a buffer
// nothing reads is one -O2 may delete along with the malloc that made it.
static int worker_touch(void* arg) {
  long long bytes = *(const long long*)arg;
  volatile unsigned char* block = (volatile unsigned char*)malloc((size_t)bytes);
  if (!block) return 1;
  long long sum = 0;
  for (long long i = 0; i < bytes; i += 4096) block[i] = (unsigned char)((i >> 12) & 0x7f);
  for (long long i = 0; i < bytes; i += 4096) sum += block[i];
  printf("@ touched_bytes=%lld\n", bytes);
  printf("@ touch_checksum=%lld\n", sum);
  free((void*)block);
  return 0;
}

static int cmd_touch(const Options* o) {
  long long bytes = o->mib * 1048576LL;
  return run_in_worker(worker_touch, &bytes);
}

// What the gate exists to catch: a "monitor" that holds the whole stream before it looks at
// it. Reads stdin into a buffer that doubles as it fills, then reports what it held.
static int worker_hold(void* arg) {
  (void)arg;
  size_t cap = STREAM_BUFFER_BYTES, len = 0;
  unsigned char* all = (unsigned char*)malloc(cap);
  if (!all) return 1;
  for (;;) {
    if (len == cap) {
      unsigned char* bigger = (unsigned char*)realloc(all, cap * 2);
      if (!bigger) {
        free(all);
        return 1;
      }
      all = bigger;
      cap *= 2;
    }
    size_t got = fread(all + len, 1, cap - len, stdin);
    if (got == 0) break;
    len += got;
  }
  unsigned long long sum = 0;
  for (size_t i = 0; i < len; i += 4096) sum += all[i];
  printf("@ held_bytes=%zu\n", len);
  printf("@ held_checksum=%llu\n", sum);
  free(all);
  return 0;
}

static int cmd_hold(const Options* o) {
  (void)o;
  if (!stdin_is_a_pipe()) return refuse_seekable_stdin();
  return run_in_worker(worker_hold, NULL);
}

static int cmd_bins(int argc, char** argv, const Options* o) {
  if (!load_profile(o->profile, &g_profile)) return 3;
  for (int i = o->first_value; i < argc; i++) {
    double x = strtod(argv[i], NULL);
    printf("bin %s %d\n", argv[i], bin_index(&g_profile, x));
  }
  return 0;
}

static int parse_ll_list(const char* s, long long* out, int cap) {
  int n = 0;
  char* end = NULL;
  while (s && *s && n < cap) {
    out[n++] = strtoll(s, &end, 10);
    if (end == s) return -1;
    s = (*end == ',') ? end + 1 : end;
    if (*end != ',' && *end != '\0') return -1;
  }
  return n;
}

static int parse_double_list(const char* s, double* out, int cap) {
  int n = 0;
  char* end = NULL;
  while (s && *s && n < cap) {
    out[n++] = strtod(s, &end);
    if (end == s) return -1;
    s = (*end == ',') ? end + 1 : end;
    if (*end != ',' && *end != '\0') return -1;
  }
  return n;
}

static int cmd_psi(const Options* o) {
  long long e[MAX_BINS], a[MAX_BINS];
  int ne = parse_ll_list(o->expected, e, MAX_BINS), na = parse_ll_list(o->actual, a, MAX_BINS);
  if (ne < 1 || ne != na) {
    fprintf(stderr, "error: --expected and --actual need the same number of counts\n");
    return 3;
  }
  long long n_e = 0, n_a = 0;
  for (int i = 0; i < ne; i++) {
    n_e += e[i];
    n_a += a[i];
  }
  double contrib[MAX_BINS];
  double psi = psi_from_counts(e, n_e, a, n_a, ne, o->floor, contrib);
  printf("@ psi=%.17g\n", psi);
  for (int i = 0; i < ne; i++) printf("contrib %d %.17g\n", i, contrib[i]);
  return 0;
}

static int cmd_breach(const Options* o) {
  double c[MAX_BINS];
  int n = parse_double_list(o->contrib, c, MAX_BINS);
  if (n < 1) {
    fprintf(stderr, "error: --contrib needs at least one value\n");
    return 3;
  }
  printf("@ bin=%d\n", breach_bin(o->psi, c, n, o->threshold));
  return 0;
}

// Offer accounts 1000, 1001, ... with scores i / n to a fresh reservoir and print what it
// kept. The notebook runs the same offers through the same algorithm in Python.
static int cmd_reservoir(const Options* o) {
  if (o->k < 1 || o->k > RESERVOIR_MAX) {
    fprintf(stderr, "error: --k must be between 1 and %d\n", RESERVOIR_MAX);
    return 3;
  }
  Reservoir* r = &g_monitor.tail;
  reservoir_init(r, o->k, o->seed);
  for (long long i = 0; i < o->n; i++)
    reservoir_offer(r, (uint64_t)(1000 + i), (double)i / (double)o->n);
  int outside = 0;  // slots k.. were zeroed by reservoir_init; any write there is a bug
  for (int s = o->k; s < RESERVOIR_MAX; s++) outside += r->ids[s] != 0 || r->scores[s] != 0.0;
  printf("@ seen=%lld\n", r->seen);
  printf("@ state=%" PRIu64 "\n", r->state);
  printf("@ outside_writes=%d\n", outside);
  long long kept = r->seen < r->k ? r->seen : r->k;
  for (long long j = 0; j < kept; j++)
    printf("sample %lld %" PRIu64 " %.17g\n", j, r->ids[j], r->scores[j]);
  return 0;
}

static int cmd_facts(void) {
  uint16_t probe = 1;
  unsigned char first;
  memcpy(&first, &probe, 1);
  printf("@ stream_buffer_bytes=%d\n", STREAM_BUFFER_BYTES);
  printf("@ record_bytes=%d\n", RECORD_BYTES);
  printf("@ sizeof_record=%zu\n", sizeof(Record));
  printf("@ max_bins=%d\n", MAX_BINS);
  printf("@ reservoir_max=%d\n", RESERVOIR_MAX);
  printf("@ checkpoint_every=%d\n", CHECKPOINT_EVERY);
  printf("@ psi_floor=%.17g\n", kPsiFloor);
  printf("@ psi_threshold=%.17g\n", kPsiThreshold);
  printf("@ little_endian=%d\n", first == 1);
  printf("@ monitor_struct_bytes=%zu\n", sizeof(Monitor));
  return 0;
}

// ---------------------------------------------------------------------------------------
// The self-test: your feedback loop. Every case below is hand-workable.
// ---------------------------------------------------------------------------------------

static Profile make_profile(int n_bins, const double* edges, int tail_bin) {
  Profile p;
  memset(&p, 0, sizeof p);
  p.n_bins = n_bins;
  for (int i = 0; i <= n_bins; i++) p.edges[i] = edges[i];
  for (int i = 0; i < n_bins; i++) p.counts[i] = 1;
  p.n = n_bins;
  p.tail_bin = tail_bin;
  return p;
}

static void check_bin_index(void) {
  const double open[5] = {-INFINITY, 0.25, 0.5, 0.75, INFINITY};
  Profile p = make_profile(4, open, 4);
  struct { double x; int want; const char* why; } cases[] = {
      {0.1, 0, "0.1 is inside [-inf, 0.25)"},
      {0.25, 1, "a score ON an interior edge belongs to the bin ABOVE it"},
      {0.24999999999999997, 0, "the double just below 0.25 is still bin 0"},
      {0.74, 2, "0.74 is inside [0.5, 0.75)"},
      {1.0, 3, "1.0 is inside the last bin"},
      {0.0, 0, "0.0 is inside [-inf, 0.25)"},
  };
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    int got = bin_index(&p, cases[i].x);
    require(got == cases[i].want, "bin_index(%.17g) on edges {-inf, 0.25, 0.5, 0.75, inf} "
            "gave %d, expected %d — %s", cases[i].x, got, cases[i].want, cases[i].why);
  }
  const double closed[3] = {0.1, 0.5, 0.9};
  Profile q = make_profile(2, closed, 2);
  require(bin_index(&q, 0.0) == 0, "bin_index(0.0) on edges {0.1, 0.5, 0.9} gave %d; a score "
          "below the first edge is clipped into bin 0", bin_index(&q, 0.0));
  require(bin_index(&q, 0.9) == 1, "bin_index(0.9) on edges {0.1, 0.5, 0.9} gave %d; the "
          "last edge itself is in the last bin, which is closed on the right",
          bin_index(&q, 0.9));
  require(bin_index(&q, 7.0) == 1, "bin_index(7.0) on edges {0.1, 0.5, 0.9} gave %d; a score "
          "above the last edge is clipped into the last bin", bin_index(&q, 7.0));
}

static void check_psi(void) {
  double c[MAX_BINS];
  const long long same[3] = {10, 20, 30};
  double psi = psi_from_counts(same, 60, same, 60, 3, kPsiFloor, c);
  require(psi == 0.0, "identical counts must give psi 0.0, you returned %.17g", psi);
  const long long small[2] = {10, 10}, big[2] = {20, 20};
  psi = psi_from_counts(small, 20, big, 40, 2, kPsiFloor, c);
  require(psi == 0.0, "the same shape at twice the size must give psi 0.0, you returned %.17g "
          "— divide each side by its OWN total, as doubles", psi);
  const long long e[2] = {50, 50}, a[2] = {100, 0};
  psi = psi_from_counts(e, 100, a, 100, 2, kPsiFloor, c);
  double want0 = (1.0 - 0.5) * log(1.0 / 0.5);
  double want1 = (1e-6 - 0.5) * log(1e-6 / 0.5);
  require(isfinite(psi), "a bin that emptied out gave psi %.17g; floor BOTH shares at "
          "1e-6 before the logarithm", psi);
  require(fabs(c[0] - want0) <= 1e-12 && fabs(c[1] - want1) <= 1e-12,
          "contributions for expected {50, 50} against actual {100, 0} were {%.17g, %.17g}, "
          "expected {%.17g, %.17g}", c[0], c[1], want0, want1);
  require(fabs(psi - (c[0] + c[1])) <= 1e-12, "psi %.17g is not the sum of its "
          "contributions %.17g + %.17g", psi, c[0], c[1]);
  const long long from_nothing[2] = {100, 0}, spread[2] = {50, 50};
  psi = psi_from_counts(from_nothing, 100, spread, 100, 2, kPsiFloor, c);
  require(isfinite(psi) && fabs(c[1] - want1) <= 1e-12, "expected {100, 0} against actual "
          "{50, 50} — a bin that APPEARED from nothing — gave psi %.17g and contribution "
          "%.17g; the floor goes on the expected side as well as the actual one", psi, c[1]);
  const long long none[2] = {0, 0};
  psi = psi_from_counts(e, 100, none, 0, 2, kPsiFloor, c);
  require(isnan(psi) && isnan(c[0]) && isnan(c[1]), "with n_actual = 0 there are no shares: "
          "return NAN and set every contribution to NAN; you returned %.17g", psi);
}

static void check_breach(void) {
  const double c[4] = {0.05, 0.12, 0.12, 0.01};
  int got = breach_bin(0.30, c, 4, 0.25);
  require(got == 1, "psi 0.30 over threshold 0.25 with contributions {0.05, 0.12, 0.12, 0.01} "
          "gave %d; bins 1 and 2 tie, and the LOWEST-numbered of a tie is named: 1", got);
  got = breach_bin(0.25, c, 4, 0.25);
  require(got == -1, "psi exactly ON the threshold is within it (module 1's rule); you "
          "returned %d instead of -1", got);
  got = breach_bin(NAN, c, 4, 0.25);
  require(got == -1, "a NaN psi is greater than nothing, so it is not a breach; you returned "
          "%d instead of -1", got);
  const double d[3] = {0.0, 0.0, 0.4};
  got = breach_bin(0.4, d, 3, 0.25);
  require(got == 2, "psi 0.4 with contributions {0, 0, 0.4} gave %d, expected 2", got);
}

static void check_reservoir(void) {
  Reservoir r;
  reservoir_init(&r, 3, 12345);
  for (int i = 0; i < 3; i++) reservoir_offer(&r, (uint64_t)(10 + i), 0.5 + i);
  require(r.seen == 3, "after three offers r->seen must be 3, it is %lld", r.seen);
  require(r.state == 12345, "the first k offers must NOT draw from the generator; the state "
          "moved from 12345 to %" PRIu64, r.state);
  require(r.ids[0] == 10 && r.ids[1] == 11 && r.ids[2] == 12 && r.scores[2] == 2.5,
          "the first k offers fill slots 0..k-1 in order");
  uint64_t copy = 12345;
  uint64_t j = splitmix64_next(&copy) % 4;
  reservoir_offer(&r, 99, 9.5);
  require(r.state == copy, "the fourth offer must draw exactly once from the generator");
  require(r.seen == 4, "after four offers r->seen must be 4, it is %lld", r.seen);
  if (j < 3)
    require(r.ids[j] == 99 && r.scores[j] == 9.5, "the draw gave j = %" PRIu64 " < k, so "
            "account 99 must now be in slot %" PRIu64 " (j = draw %% (t + 1) with t = 3)", j, j);
  else
    require(r.ids[0] == 10 && r.ids[1] == 11 && r.ids[2] == 12, "the draw gave j = 3 >= k, so "
            "account 99 must NOT replace anything");
  for (int i = 0; i < 200; i++) reservoir_offer(&r, (uint64_t)(2000 + i), 0.25);
  for (int s = 3; s < RESERVOIR_MAX; s++)
    require(r.ids[s] == 0 && r.scores[s] == 0.0, "after 204 offers to a reservoir of k = 3, "
            "slot %d holds account %" PRIu64 ". Nothing may be written outside slots 0..k-1: "
            "the replacement test is j < k, strictly", s, r.ids[s]);
}

static void write_record(FILE* f, uint64_t id, double score) {
  unsigned char b[RECORD_BYTES];
  memcpy(b, &id, 8);
  memcpy(b + 8, &score, 8);
  fwrite(b, 1, RECORD_BYTES, f);
}

static void check_monitor(void) {
  FILE* f = tmpfile();
  require(f != NULL, "could not create a temporary file for the self-test");
  const double scores[5] = {0.30, NAN, 1.0, -0.5, 0.9};
  for (int i = 0; i < 5; i++) write_record(f, (uint64_t)(500 + i), scores[i]);
  fwrite("xyz", 1, 3, f);
  rewind(f);
  const double edges[3] = {-INFINITY, 0.5, INFINITY};
  Profile p = make_profile(2, edges, 2);  // tail_bin == n_bins: no tail, no sampling
  Monitor m;
  monitor_init(&m, 4, 1);
  int rc = monitor_stream(f, &p, &m);
  fclose(f);
  require(rc == 0, "monitor_stream returned %d on a readable stream", rc);
  require(m.records_read == 5, "five complete records were written; records_read is %lld",
          m.records_read);
  require(m.rejected == 2, "NaN and -0.5 are not probabilities, so rejected must be 2; it is "
          "%lld. If it is 1, your validity test lets NaN through: every ordered comparison "
          "with NaN is false", m.rejected);
  require(m.valid == 3, "valid must be 3; it is %lld", m.valid);
  require(m.counts[0] == 1 && m.counts[1] == 2, "0.30 belongs in bin 0 and 1.0 and 0.9 in bin 1; "
          "counts were {%lld, %lld}", m.counts[0], m.counts[1]);
  require(m.truncated_bytes == 3, "three stray bytes followed the last record; truncated_bytes "
          "is %lld. fread with an element size of 16 swallows them — read bytes, not records",
          m.truncated_bytes);
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

static int cmd_selftest(void) {
  uint16_t probe = 1;
  unsigned char first;
  memcpy(&first, &probe, 1);
  if (first != 1 || sizeof(Record) != RECORD_BYTES) {
    printf("  FAIL  this machine is not little-endian, or a Record is not 16 bytes; the "
           "extract's format assumes both\n");
    return 1;
  }
  int todos = 0, fails = 0;
  struct {
    const char* name;
    void (*fn)(void);
  } checks[] = {
      {"exercise 2  bin_index", check_bin_index},
      {"exercise 3  psi_from_counts", check_psi},
      {"exercise 4  breach_bin", check_breach},
      {"exercise 5  reservoir_offer", check_reservoir},
      {"exercise 6  monitor_stream", check_monitor},
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
          "  selftest     run the built-in checks for the five C exercises\n"
          "  facts        print the contract: buffer size, record size, cadence, floor\n"
          "  bins         --profile P  v1 v2 ...   the bin of each value\n"
          "  psi          --expected c,c,.. --actual c,c,.. [--floor F]\n"
          "  breach       --psi X --contrib c,c,.. [--threshold T]\n"
          "  reservoir    --n N --k K --seed S\n"
          "  monitor      --profile P --k K --seed S [--no-psi|--no-tail|--counts-only]\n"
          "               reads the month from a pipe on stdin\n"
          "  baseline     --profile P --k K --seed S    the worker's own footprint\n"
          "  touch        --mib M    touch M MiB in a worker (the ru_maxrss unit probe)\n"
          "  hold         read the whole pipe into memory (what the gate exists to catch)\n");
  return 3;
}

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  Options o;
  memset(&o, 0, sizeof o);
  o.k = 1000;
  o.seed = 1;
  o.floor = kPsiFloor;
  o.threshold = kPsiThreshold;
  o.first_value = argc;
  const char* cmd = argv[1];
  for (int i = 2; i < argc; i++) {
    const char* key = argv[i];
    if (!strcmp(key, "--counts-only") || !strcmp(key, "--no-psi") ||
        !strcmp(key, "--no-tail")) {
      if (strcmp(key, "--no-tail") != 0) g_with_psi = 0;
      if (strcmp(key, "--no-psi") != 0) g_with_tail = 0;
      continue;
    }
    if (strncmp(key, "--", 2) != 0) {  // the first positional value: bins reads from here
      o.first_value = i;
      break;
    }
    if (i + 1 >= argc) {
      fprintf(stderr, "error: %s needs a value\n", key);
      return 3;
    }
    const char* v = argv[++i];
    if (!strcmp(key, "--profile")) o.profile = v;
    else if (!strcmp(key, "--k")) o.k = atoi(v);
    else if (!strcmp(key, "--seed")) o.seed = strtoull(v, NULL, 10);
    else if (!strcmp(key, "--n")) o.n = strtoll(v, NULL, 10);
    else if (!strcmp(key, "--mib")) o.mib = strtoll(v, NULL, 10);
    else if (!strcmp(key, "--floor")) o.floor = strtod(v, NULL);
    else if (!strcmp(key, "--threshold")) o.threshold = strtod(v, NULL);
    else if (!strcmp(key, "--psi")) o.psi = strtod(v, NULL);
    else if (!strcmp(key, "--expected")) o.expected = v;
    else if (!strcmp(key, "--actual")) o.actual = v;
    else if (!strcmp(key, "--contrib")) o.contrib = v;
    else {
      fprintf(stderr, "error: unknown option %s\n", key);
      return 3;
    }
  }
  if ((!strcmp(cmd, "monitor") || !strcmp(cmd, "baseline")) &&
      (o.k < 1 || o.k > RESERVOIR_MAX)) {
    fprintf(stderr, "error: --k must be between 1 and %d\n", RESERVOIR_MAX);
    return 3;
  }
  if ((!strcmp(cmd, "monitor") || !strcmp(cmd, "baseline") || !strcmp(cmd, "bins")) &&
      !o.profile) {
    fprintf(stderr, "error: %s needs --profile\n", cmd);
    return 3;
  }
  if (!strcmp(cmd, "selftest")) return cmd_selftest();
  if (!strcmp(cmd, "facts")) return cmd_facts();
  if (!strcmp(cmd, "bins")) return cmd_bins(argc, argv, &o);
  if (!strcmp(cmd, "psi")) return cmd_psi(&o);
  if (!strcmp(cmd, "breach")) return cmd_breach(&o);
  if (!strcmp(cmd, "reservoir")) return cmd_reservoir(&o);
  if (!strcmp(cmd, "monitor")) return cmd_monitor(&o);
  if (!strcmp(cmd, "baseline")) return cmd_baseline(&o);
  if (!strcmp(cmd, "touch")) return cmd_touch(&o);
  if (!strcmp(cmd, "hold")) return cmd_hold(&o);
  return usage();
}
