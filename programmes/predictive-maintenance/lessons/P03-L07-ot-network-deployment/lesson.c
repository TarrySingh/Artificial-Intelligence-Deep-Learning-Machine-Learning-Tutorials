// P03-L07 — Deployment on an OT network. THE GATEWAY.
//
// Four exercises live in this file: gw_alloc, frame_decode, q_health, and the three
// store-and-forward functions. Each one carries its requirements and a worked example in the
// comment block above it. The self-test is your feedback loop.
//
//     make                                                       build lesson.c -> lesson_bin
//     make test                                                  build, then run the self-test
//     make clean                                                 delete every build product
//
// RUN `make clean` AFTER MOVING OR COPYING YOUR CHECKOUT. make compares timestamps, and a
// binary that arrived with a copied directory is newer than the .c beside it, so make will
// not rebuild it and you will be testing yesterday's answers.
//
// Nothing in this file opens a socket. The "wire" is a recorded capture, synthesised at
// start-up from a fixed seed, replayed out of a static buffer that is NOT part of the
// gateway's memory budget — exactly as a stored trace off a plant historian would be.

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =======================================================================================
// The plant, and the network it has to live on. Every one of these is a DESIGN NUMBER: the
// notebook prints what each one costs you rather than asserting that it is enough.
// =======================================================================================
#define N_MACHINES      80
#define N_HOURS         168                       // one week of hourly polls
#define BURST_LEN       32                        // samples in one vibration burst
#define N_FRAMES        (N_MACHINES * N_HOURS)    // one frame per machine per hour
#define FRAME_BYTES     (3 + 2 * BURST_LEN + 2)   // addr, func, bytecount, payload, crc
#define CAPTURE_BYTES   ((size_t)N_FRAMES * FRAME_BYTES)

#define GW_ARENA_BYTES  40960   // the gateway's ENTIRE heap. There is no other memory.
#define RING_CAPACITY   3072    // records the store-and-forward buffer can hold

#define SAMPLE_SHIFT    8       // a raw count is engineering units x 2^8
#define Q_FRAC          16      // the gateway computes in Q16.16
#define WIRE_FRAC       8       // the register map carries Q8.8 in one 16-bit word
#define WIRE_SCALE      256     // = 1 << WIRE_FRAC

#define FUNC_READ_INPUT 0x04    // the function code every frame in this capture carries
#define CORRUPT_STRIDE  997     // line noise: frame i is corrupt when i % 997 == 13
#define CORRUPT_PHASE   13

static const uint64_t kSeed = 20260922ULL;
static const uint64_t kSeed2 = 20260922ULL ^ 0xA5A5A5A5A5A5A5A5ULL;
static const uint64_t kGamma = 0x9E3779B97F4A7C15ULL;

// Frame decode return codes. Negative because 0 is "this frame is good".
#define FD_OK            0
#define FD_SHORT        (-1)    // fewer bytes than a frame header plus a checksum
#define FD_FUNC         (-2)    // not the function code this map uses
#define FD_BYTECOUNT    (-3)    // the byte count does not match the frame length
#define FD_CRC          (-4)    // the checksum does not match the bytes
#define FD_CAPACITY     (-5)    // more registers than the caller's buffer holds

#define SF_REFUSED       0u     // sf_push returns this when the buffer is full

#define COMMONS_MAYBE_UNUSED __attribute__((unused))

typedef struct {
  uint8_t addr;
  uint8_t func;
  int nreg;
  uint16_t reg[BURST_LEN];
} Frame;

typedef struct {
  uint32_t seq;
  uint16_t hour;
  uint16_t wire;
  uint8_t machine;
  uint8_t flags;
  uint16_t pad;
} Record;

typedef struct {
  Record* slot;
  int cap;
  int head;
  int count;
  uint32_t next_seq;
  long refused;
  long peak;
} Ring;

// =======================================================================================
// Plumbing for the self-test. Read it once, then ignore it.
// =======================================================================================
//
// C has no exceptions, so an unfinished exercise reports itself with setjmp/longjmp: todo()
// records a message and jumps back to the check that called it. main() turns an unfinished
// exercise into exit code 2, which the notebook's Python wrapper translates into
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

// =======================================================================================
// The gateway's memory. All of it.
// =======================================================================================
//
// g_arena is the whole heap this device has. Nothing in the gateway may call malloc: a
// gateway that allocates while it is running is a gateway that runs out of memory at 3 a.m.
// on the one day the link was down long enough to matter. Allocation happens at start-up,
// gw_freeze() closes the door, and everything after that runs out of what was taken then.
static unsigned char g_arena[GW_ARENA_BYTES];
static size_t g_arena_used = 0;
static size_t g_arena_peak = 0;
static int g_frozen = 0;
static long g_alloc_calls = 0;
static long g_alloc_refused = 0;
static long g_alloc_after_freeze = 0;

// GIVEN. Closes the door: every later gw_alloc must refuse.
static void gw_freeze(void) { g_frozen = 1; }

// GIVEN. Puts the arena back to its start-up state. Only the self-test uses this; a real
// gateway reboots instead.
static void gw_reset(void) {
  g_arena_used = 0;
  g_frozen = 0;
  g_alloc_calls = 0;
  g_alloc_refused = 0;
  g_alloc_after_freeze = 0;
}

// This #define is a tripwire, not decoration. Every line between here and the matching
// #undef is code that runs on the gateway, and the gateway has no heap beyond g_arena. Write
// malloc() below this line and the LINKER will tell you so, by name.
#define malloc  gw_no_malloc_in_the_gateway
#define calloc  gw_no_calloc_in_the_gateway
#define realloc gw_no_realloc_in_the_gateway
#define free    gw_no_free_in_the_gateway

// =======================================================================================
// EXERCISE 1 — gw_alloc()
// =======================================================================================
//
// A bump allocator over a fixed arena. It is the only allocator the gateway has, it never
// frees, and once gw_freeze() has been called it hands out nothing at all.
//
// Requirements, all graded:
//   * return NULL for bytes == 0 — there is nothing to hand out;
//   * after gw_freeze(), return NULL, and count the attempt in BOTH g_alloc_refused and
//     g_alloc_after_freeze. An allocation attempted after start-up is a design defect and
//     the device has to be able to say so;
//   * align the cursor UP to 8 bytes before carving, so a Record or an int32_t lands on its
//     natural boundary;
//   * if the aligned cursor plus bytes would pass GW_ARENA_BYTES, return NULL, count it in
//     g_alloc_refused, and LEAVE THE CURSOR WHERE IT WAS. A refusal that consumed the
//     alignment padding anyway would make the remaining budget depend on the requests that
//     failed;
//   * on success, count the call in g_alloc_calls, advance g_arena_used, keep g_arena_peak
//     at the high-water mark, and return the address inside g_arena.
//
// Returns: a pointer into g_arena on success, NULL on refusal. Never a pointer from malloc.
//
// Worked example, on an empty arena: gw_alloc(1) returns &g_arena[0] and leaves
// g_arena_used == 1. The next gw_alloc(1) aligns 1 up to 8 and returns &g_arena[8], leaving
// g_arena_used == 9 — so the two pointers are 8 bytes apart, not 1.
static void* gw_alloc(size_t bytes) {
  // YOUR CODE HERE
  (void)bytes;
  todo("gw_alloc",
       "carve `bytes` out of g_arena at a cursor aligned up to 8, or return NULL and count "
       "the refusal; after gw_freeze() always return NULL");
  return NULL;
}

// =======================================================================================
// GIVEN — the checksum the protocol uses, and an integer square root.
// =======================================================================================
//
// CRC-16 with the reversed polynomial 0xA001 and an initial value of 0xFFFF: the checksum
// this capture's frames carry. You do not have to derive it; you have to use it.
static uint16_t crc16(const uint8_t* p, size_t n) {
  uint16_t crc = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    crc ^= (uint16_t)p[i];
    for (int b = 0; b < 8; b++)
      crc = (crc & 1u) ? (uint16_t)((crc >> 1) ^ 0xA001u) : (uint16_t)(crc >> 1);
  }
  return crc;
}

// Floor of the square root of a 64-bit integer, digit by digit, with no floating point
// anywhere. A gateway that has no FPU cannot call sqrt(), and one that has an FPU still
// should not: the answer would then depend on the chip. This truncates — isqrt64(8) is 2 —
// and that truncation is one of the two places the gateway's answer drifts below the
// notebook's, which is a thing you will measure rather than be told.
static COMMONS_MAYBE_UNUSED uint64_t isqrt64(uint64_t v) {
  uint64_t rem = 0, root = 0;
  for (int i = 0; i < 32; i++) {
    root <<= 1;
    rem = (rem << 2) | (v >> 62);
    v <<= 2;
    if (root < rem) {
      root++;
      rem -= root;
      root++;
    }
  }
  return root >> 1;
}

// =======================================================================================
// EXERCISE 2 — frame_decode()
// =======================================================================================
//
// One frame off the recorded capture. The layout is a Modbus RTU register-read response,
// taken from the Modbus Organization's own specifications rather than invented here
// (see claims.yaml):
//
//     byte 0        unit address        (1..N_MACHINES; the machine index plus one)
//     byte 1        function code       (must be FUNC_READ_INPUT, 0x04)
//     byte 2        byte count          (= 2 * number of registers)
//     bytes 3..     payload             (byte count bytes: BIG-ENDIAN 16-bit registers)
//     last 2 bytes  checksum            (crc16 over every byte before it, LITTLE-endian)
//
// `len` is the length of THIS frame, not of the capture.
//
// Requirements, in this order, all graded:
//   * len < 5 (three header bytes plus a two-byte checksum) -> return FD_SHORT;
//   * buf[1] != FUNC_READ_INPUT -> return FD_FUNC;
//   * buf[2] != len - 5 -> return FD_BYTECOUNT. The byte count is the only thing that ties
//     the header to the length, so an agreeing pair is what makes the rest safe to read;
//   * (len - 5) / 2 > BURST_LEN -> return FD_CAPACITY, before writing anything;
//   * the last two bytes, read LITTLE-endian, must equal crc16(buf, len - 2); if not,
//     return FD_CRC;
//   * only then fill *out: addr, func, nreg = (len - 5) / 2, and reg[i] assembled BIG-endian
//     as (buf[3 + 2*i] << 8) | buf[4 + 2*i]. Return FD_OK.
//
// Returns: FD_OK (0) on a good frame, one of the negative codes above otherwise. On any
// failure *out is left alone — a half-filled frame is worse than no frame, because the
// caller cannot tell which half.
//
// Worked example: the five bytes {0x01, 0x04, 0x00, 0x22, 0xC0} decode to addr 1, func 4,
// nreg 0 and FD_OK — an empty but valid response, whose checksum is crc16(buf, 3) = 0xC022
// stored low byte first. Change the 0x22 to 0x23 and the same call returns FD_CRC and writes
// nothing.
static int frame_decode(const uint8_t* buf, size_t len, Frame* out) {
  // YOUR CODE HERE
  (void)buf;
  (void)len;
  (void)out;
  todo("frame_decode",
       "check the length, then the function code, then the byte count, then the capacity, "
       "then the checksum, and only then fill *out with BIG-endian registers");
  return FD_SHORT;
}

// =======================================================================================
// EXERCISE 3 — q_health()
// =======================================================================================
//
// The inference path, in integers, on the gateway. It is the same arithmetic the notebook
// does in float64, and it will not give the same answer — measuring that difference is what
// this lesson is for.
//
// The fixed-point convention, fixed here because two implementations that disagree about it
// do not reproduce each other:
//
//     a raw count c        is the engineering value c * 2^-SAMPLE_SHIFT
//     x_q[i]               is that value in Q16.16, i.e. (int32_t)c << SAMPLE_SHIFT
//     baseline_q           is the commissioned baseline in Q16.16
//     the answer           is rms(x) / baseline, in Q16.16
//
// The recipe:
//     acc   = sum over i of (int64_t)x_q[i] * x_q[i]     // Q32.32, so acc/2^32 is x*x
//     mean  = acc / n                                    // integer divide: truncates
//     rms_q = isqrt64(mean)                              // Q16.16, because sqrt halves the
//                                                        // exponent of the scale factor
//     health_q = ((int64_t)rms_q << Q_FRAC) / baseline_q // Q16.16
//
// Requirements, all graded:
//   * return -1 for n <= 0 or baseline_q <= 0 — a health index needs a baseline, and 0 is a
//     legitimate answer so the failure value has to be outside the range;
//   * accumulate in int64_t. A burst of 32 samples at full scale sums to 32 * 16776960^2 =
//     about 9.0e15, four million times the largest value an int32_t holds, and the overflow
//     is silent;
//   * shift rms_q into the numerator in int64_t too: rms_q << 16 is about 1.1e12 for a
//     full-scale burst, so `(rms_q << 16) / baseline_q` written in int arithmetic is the
//     same bug one line further down;
//   * every intermediate truncates towards zero — do not round. The gateway's answer must be
//     reproducible on a device with no floating point at all, and "round to nearest" is a
//     choice the notebook and the gateway would then have to agree on twice;
//   * if the quotient will not fit in an int32_t, return INT32_MAX rather than letting it
//     wrap. A saturated reading is visibly wrong at a glance; a wrapped one is plausible,
//     wrong, and will be believed.
//
// Returns: the health index in Q16.16 as an int32_t (divide by 65536.0 to read it), or -1.
//
// Worked example: a burst of 32 samples all equal to 2048 counts, with baseline_q = 2048 <<
// SAMPLE_SHIFT. Every x_q is 2048 << 8 = 524288, each square is 274877906944, the sum over
// 32 is 8796093022208 and the mean is 274877906944 again. isqrt64 of that is exactly 524288,
// and (524288 << 16) / 524288 is 65536 — Q16.16 for 1.0. A machine sitting exactly on its
// commissioned baseline reads 1.0, which is the whole point of dividing by a baseline.
static int32_t q_health(const int32_t* x_q, int n, int32_t baseline_q) {
  // YOUR CODE HERE
  (void)x_q;
  (void)n;
  (void)baseline_q;
  todo("q_health",
       "sum the squares in int64_t, divide by n, take isqrt64 of that, then shift the result "
       "left by Q_FRAC in int64_t and divide by baseline_q");
  return -1;
}

// =======================================================================================
// EXERCISE 4 — sf_push(), sf_take(), sf_ack()
// =======================================================================================
//
// Store and forward. The link to the plant historian goes down; the machines do not. The
// gateway holds readings in a ring inside the arena until the link comes back, and the
// property it must have is EXACTLY ONCE: every reading that entered the buffer reaches the
// historian, and none reaches it twice.
//
// The two halves of that property are defended in two different places, and you need both:
//
//   NOTHING IS LOST      sf_push refuses when the buffer is full rather than overwriting the
//                        oldest unacknowledged record. A refusal is a number you can report;
//                        an overwrite is a reading that never existed.
//   NOTHING IS DUPLICATED every record carries a sequence number that only ever increases,
//                        and the historian drops anything it has already seen. Delivery is
//                        at-least-once — the link may take a batch and lose the
//                        acknowledgement on the way back, and the gateway will then offer
//                        the same records again — so the sequence number is what turns
//                        at-least-once into exactly-once.
//
// A REFUSED READING MUST NOT CONSUME A SEQUENCE NUMBER. That is what makes the delivered
// sequence contiguous, and a contiguous sequence is how the historian knows, without asking
// anybody, that nothing went missing in transit.
//
// --- sf_push(r, machine, hour, wire) -------------------------------------------------
// Requirements, all graded:
//   * if r->count == r->cap: increment r->refused, change NOTHING else — not the cursor, not
//     the count, not r->next_seq — and return SF_REFUSED (0);
//   * otherwise write the record at (r->head + r->count) % r->cap with seq = r->next_seq,
//     then advance r->next_seq, increment r->count, keep r->peak at the high-water mark of
//     r->count, and return the sequence number that was assigned;
//   * flags and pad are set to 0.
// Returns: the assigned sequence number, which is never 0, or SF_REFUSED (0) on refusal.
//
// --- sf_take(r, out, max) ------------------------------------------------------------
// Copy up to `max` of the held records into `out`, OLDEST FIRST, and return how many were
// copied. This is a peek, not a pop: it must not change the ring at all, so calling it twice
// without an acknowledgement in between returns the same records twice. That is the point —
// the link is allowed to lose an acknowledgement.
// Returns: the number of records copied, between 0 and min(max, r->count).
//
// --- sf_ack(r, seq) ------------------------------------------------------------------
// The historian has durably stored everything up to and including `seq`. Drop every held
// record whose own seq is <= that, oldest first, and return how many were dropped. An
// acknowledgement for a sequence number already dropped drops nothing and is not an error:
// acknowledgements arrive late, arrive twice, and arrive out of order.
// Returns: the number of records dropped, 0 or more.
//
// Worked example: on an empty ring of capacity 4, three pushes return 1, 2 and 3.
// sf_take(out, 2) returns 2 and leaves count at 3. sf_ack(2) drops records 1 and 2 and
// returns 2, leaving count at 1. sf_ack(2) again returns 0. A fourth push returns 4.
static uint32_t sf_push(Ring* r, uint8_t machine, uint16_t hour, uint16_t wire) {
  // YOUR CODE HERE
  (void)r;
  (void)machine;
  (void)hour;
  (void)wire;
  todo("sf_push",
       "refuse and count when the ring is full, changing nothing else; otherwise write at "
       "(head + count) % cap and return the sequence number you assigned");
  return SF_REFUSED;
}

static int sf_take(const Ring* r, Record* out, int max) {
  // YOUR CODE HERE
  (void)r;
  (void)out;
  (void)max;
  todo("sf_take",
       "copy up to max held records into out, oldest first, WITHOUT changing the ring");
  return 0;
}

static int sf_ack(Ring* r, uint32_t seq) {
  // YOUR CODE HERE
  (void)r;
  (void)seq;
  todo("sf_ack",
       "drop every held record whose own seq is <= the one acknowledged, oldest first, and "
       "return how many were dropped");
  return 0;
}

#undef malloc
#undef calloc
#undef realloc
#undef free

// =======================================================================================
// GIVEN — the capture, the gateway's start-up, and the link harness.
// =======================================================================================
//
// The capture is SYNTHETIC and is built here, at start-up, from a counter-based stream
// seeded with 20260922. The notebook builds the same bytes in numpy and checks that the two
// agree byte for byte, because a comparison between two implementations means nothing unless
// they were handed the same input.
//
// It is a static buffer, and it is deliberately NOT inside g_arena: it stands in for the
// wire, and the gateway never holds more than one frame of it at a time.
static uint8_t g_capture[CAPTURE_BYTES];
static int g_capture_built = 0;

static uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

static uint64_t draw_frame(long i) { return mix64(kSeed + (uint64_t)(i + 1) * kGamma); }
static uint64_t draw_sample(long j) { return mix64(kSeed2 + (uint64_t)(j + 1) * kGamma); }

// The commissioned baseline of machine m, in raw counts. Set once, at commissioning, and
// stored in the gateway's configuration — which is why both sides of this lesson hold the
// same number and the only differences left are arithmetic and the wire.
static int32_t baseline_counts(int m) { return 8192 + 64 * m; }

// Machine classes. 40 healthy, 24 sitting close to the alarm line, 16 degrading.
#define N_HEALTHY   40
#define N_MARGINAL  24

// The target health of machine m at hour h, in parts per million. Integer arithmetic only,
// so the notebook reproduces it exactly.
static int64_t target_health_ppm(int m, int h, uint64_t u) {
  if (m < N_HEALTHY) return 1000000 + (int64_t)(u % 120001ULL) - 60000;
  if (m < N_HEALTHY + N_MARGINAL) return 1473200 + (int64_t)(u % 80001ULL) - 40000;
  int d = m - (N_HEALTHY + N_MARGINAL);
  int onset = 24 + 6 * d;
  int64_t slope = 8000 + 400 * (int64_t)d;
  int64_t ramp = (h > onset) ? (int64_t)(h - onset) * slope : 0;
  return 1000000 + ramp + (int64_t)(u % 40001ULL) - 20000;
}

static void capture_build(void) {
  if (g_capture_built) return;
  for (long i = 0; i < N_FRAMES; i++) {
    int h = (int)(i / N_MACHINES);
    int m = (int)(i % N_MACHINES);
    int64_t ppm = target_health_ppm(m, h, draw_frame(i));
    int64_t level = (int64_t)baseline_counts(m) * ppm / 1000000;
    uint8_t* f = &g_capture[(size_t)i * FRAME_BYTES];
    f[0] = (uint8_t)(m + 1);
    f[1] = FUNC_READ_INPUT;
    f[2] = (uint8_t)(2 * BURST_LEN);
    for (int k = 0; k < BURST_LEN; k++) {
      int64_t jitter = (int64_t)(draw_sample(i * BURST_LEN + k) % 129ULL) - 64;
      int64_t c = level + jitter;
      if (c < 0) c = 0;
      if (c > 65535) c = 65535;
      f[3 + 2 * k] = (uint8_t)((uint64_t)c >> 8);
      f[4 + 2 * k] = (uint8_t)((uint64_t)c & 0xFFu);
    }
    uint16_t crc = crc16(f, FRAME_BYTES - 2);
    f[FRAME_BYTES - 2] = (uint8_t)(crc & 0xFFu);
    f[FRAME_BYTES - 1] = (uint8_t)(crc >> 8);
    // Line noise. The protocol carries a checksum because frames arrive damaged; a gateway
    // that trusts every frame it is handed publishes a health index built out of noise.
    if (i % CORRUPT_STRIDE == CORRUPT_PHASE) f[7] ^= 0x40u;
  }
  g_capture_built = 1;
}

// The gateway's start-up: take every byte it will ever need, then close the door.
typedef struct {
  int32_t* baseline_q;   // N_MACHINES entries
  int32_t* scratch;      // BURST_LEN entries, the one frame being worked on
  Record* batch;         // the outbound batch handed to the link
  Ring ring;
  int ok;
} Gateway;

#define BATCH_MAX 64

static void gateway_start(Gateway* g, int capacity) {
  memset(g, 0, sizeof *g);
  g->baseline_q = (int32_t*)gw_alloc(sizeof(int32_t) * N_MACHINES);
  g->scratch = (int32_t*)gw_alloc(sizeof(int32_t) * BURST_LEN);
  g->ring.slot = (Record*)gw_alloc(sizeof(Record) * (size_t)capacity);
  g->batch = (Record*)gw_alloc(sizeof(Record) * BATCH_MAX);
  g->ok = (g->baseline_q && g->scratch && g->ring.slot && g->batch) ? 1 : 0;
  if (g->ok) {
    for (int m = 0; m < N_MACHINES; m++)
      g->baseline_q[m] = baseline_counts(m) << SAMPLE_SHIFT;
    g->ring.cap = capacity;
    g->ring.head = 0;
    g->ring.count = 0;
    g->ring.next_seq = 1;
    g->ring.refused = 0;
    g->ring.peak = 0;
  }
  gw_freeze();   // start-up is over. Nothing else gets memory.
}

// The 16-bit register the health index goes back out on. Truncation, not rounding: the
// gateway has no rounding mode to agree on, and the notebook will measure what truncation
// costs rather than be told.
static uint16_t health_to_wire(int32_t health_q) {
  if (health_q < 0) return 0;
  int64_t w = (int64_t)health_q >> (Q_FRAC - WIRE_FRAC);
  if (w > 65535) return 65535;
  return (uint16_t)w;
}

typedef struct {
  uint32_t last_seq;
  long accepted;
  long duplicates;
  unsigned long long sum_seq;
} Historian;

// GIVEN. The far end of the link. It keeps one number — the highest sequence number it has
// durably stored — and that number is the entire defence against duplication.
static void historian_accept(Historian* hh, const Record* rec) {
  if (rec->seq <= hh->last_seq) {
    hh->duplicates++;
    return;
  }
  hh->last_seq = rec->seq;
  hh->accepted++;
  hh->sum_seq += rec->seq;
}

// =======================================================================================
// Commands
// =======================================================================================

typedef struct {
  long n;
  long dump;
  const char* scenario;
  int capacity;
} Options;

static int cmd_facts(const Options* o) {
  (void)o;
  printf("@ n_machines=%d\n", N_MACHINES);
  printf("@ n_hours=%d\n", N_HOURS);
  printf("@ burst_len=%d\n", BURST_LEN);
  printf("@ n_frames=%d\n", N_FRAMES);
  printf("@ frame_bytes=%d\n", FRAME_BYTES);
  printf("@ capture_bytes=%zu\n", CAPTURE_BYTES);
  printf("@ arena_bytes=%d\n", GW_ARENA_BYTES);
  printf("@ ring_capacity=%d\n", RING_CAPACITY);
  printf("@ record_bytes=%zu\n", sizeof(Record));
  printf("@ sample_shift=%d\n", SAMPLE_SHIFT);
  printf("@ q_frac=%d\n", Q_FRAC);
  printf("@ wire_frac=%d\n", WIRE_FRAC);
  printf("@ wire_scale=%d\n", WIRE_SCALE);
  printf("@ corrupt_stride=%d\n", CORRUPT_STRIDE);
  printf("@ corrupt_phase=%d\n", CORRUPT_PHASE);
  printf("@ batch_max=%d\n", BATCH_MAX);
  printf("@ sizeof_int=%zu\n", sizeof(int));
  printf("@ sizeof_long=%zu\n", sizeof(long));
  return 0;
}

static int cmd_capture(const Options* o) {
  capture_build();
  unsigned long long sum = 0, wsum = 0;
  for (size_t i = 0; i < CAPTURE_BYTES; i++) {
    sum += g_capture[i];
    wsum += (unsigned long long)(i + 1) * g_capture[i];
  }
  printf("@ capture_bytes=%zu\n", CAPTURE_BYTES);
  printf("@ capture_sum=%llu\n", sum);
  printf("@ capture_wsum=%llu\n", wsum);
  long dump = o->dump;
  if (dump > N_FRAMES) dump = N_FRAMES;
  for (long i = 0; i < dump; i++) {
    printf("raw %ld ", i);
    for (int b = 0; b < FRAME_BYTES; b++) printf("%02x", g_capture[(size_t)i * FRAME_BYTES + b]);
    printf("\n");
  }
  return 0;
}

// Walk the capture, decode every frame, and report what the gateway would publish.
static int cmd_health(const Options* o) {
  capture_build();
  gw_reset();
  Gateway g;
  gateway_start(&g, o->capacity > 0 ? o->capacity : RING_CAPACITY);
  require(g.ok, "the gateway could not take its memory out of a %d byte arena; gw_alloc "
                "returned NULL during start-up", GW_ARENA_BYTES);
  long good = 0, bad = 0;
  long limit = o->n > 0 && o->n < N_FRAMES ? o->n : N_FRAMES;
  Frame fr;
  for (long i = 0; i < limit; i++) {
    int rc = frame_decode(&g_capture[(size_t)i * FRAME_BYTES], FRAME_BYTES, &fr);
    if (rc != FD_OK) {
      bad++;
      printf("bad %ld %d %d\n", i, (int)(i / N_MACHINES), rc);
      continue;
    }
    int m = fr.addr - 1;
    for (int k = 0; k < fr.nreg; k++) g.scratch[k] = (int32_t)fr.reg[k] << SAMPLE_SHIFT;
    int32_t hq = q_health(g.scratch, fr.nreg, g.baseline_q[m]);
    printf("h %ld %d %d %d %u\n", i, m, (int)(i / N_MACHINES), (int)hq,
           (unsigned)health_to_wire(hq));
    good++;
  }
  printf("@ decoded=%ld\n", good);
  printf("@ rejected=%ld\n", bad);
  printf("@ arena_used=%zu\n", g_arena_used);
  printf("@ arena_peak=%zu\n", g_arena_peak);
  return 0;
}

static int link_up(const char* scenario, int hour) {
  if (!strcmp(scenario, "outage")) return !(hour >= 72 && hour < 108);
  if (!strcmp(scenario, "overflow")) return !(hour >= 40 && hour < 140);
  return 1;
}

static int ack_lost(const char* scenario, int hour) {
  if (!strcmp(scenario, "lostack")) return hour >= 40 && hour < 44;
  return 0;
}

static int cmd_forward(const Options* o) {
  capture_build();
  gw_reset();
  Gateway g;
  gateway_start(&g, o->capacity > 0 ? o->capacity : RING_CAPACITY);
  require(g.ok, "the gateway could not take its memory out of a %d byte arena", GW_ARENA_BYTES);
  Historian hist = {0, 0, 0, 0};
  long pushed = 0, corrupt = 0;
  Frame fr;
  const char* sc = o->scenario ? o->scenario : "clean";
  for (int h = 0; h < N_HOURS; h++) {
    for (int m = 0; m < N_MACHINES; m++) {
      long i = (long)h * N_MACHINES + m;
      int rc = frame_decode(&g_capture[(size_t)i * FRAME_BYTES], FRAME_BYTES, &fr);
      if (rc != FD_OK) {
        corrupt++;
        continue;
      }
      int mm = fr.addr - 1;
      for (int k = 0; k < fr.nreg; k++) g.scratch[k] = (int32_t)fr.reg[k] << SAMPLE_SHIFT;
      int32_t hq = q_health(g.scratch, fr.nreg, g.baseline_q[mm]);
      uint32_t seq = sf_push(&g.ring, (uint8_t)mm, (uint16_t)h, health_to_wire(hq));
      if (seq != SF_REFUSED) pushed++;
    }
    if (!link_up(sc, h)) continue;
    for (long round = 0; round < 100000; round++) {
      int n = sf_take(&g.ring, g.batch, BATCH_MAX);
      if (n <= 0) break;
      for (int k = 0; k < n; k++) historian_accept(&hist, &g.batch[k]);
      if (ack_lost(sc, h)) break;   // the batch landed; the acknowledgement did not
      sf_ack(&g.ring, g.batch[n - 1].seq);
    }
  }
  printf("@ scenario_clean=%d\n", !strcmp(sc, "clean"));
  printf("@ corrupt=%ld\n", corrupt);
  printf("@ pushed=%ld\n", pushed);
  printf("@ refused=%ld\n", g.ring.refused);
  printf("@ accepted=%ld\n", hist.accepted);
  printf("@ duplicates=%ld\n", hist.duplicates);
  printf("@ sum_seq=%llu\n", hist.sum_seq);
  printf("@ max_seq=%u\n", hist.last_seq);
  printf("@ held_at_end=%d\n", g.ring.count);
  printf("@ peak_backlog=%ld\n", g.ring.peak);
  printf("@ ring_capacity=%d\n", g.ring.cap);
  printf("@ arena_used=%zu\n", g_arena_used);
  printf("@ arena_peak=%zu\n", g_arena_peak);
  printf("@ alloc_calls=%ld\n", g_alloc_calls);
  printf("@ alloc_refused=%ld\n", g_alloc_refused);
  printf("@ alloc_after_freeze=%ld\n", g_alloc_after_freeze);
  return 0;
}

// A scripted probe of each exercise, printed as `@ key=value`. The self-test is the
// student's feedback loop; this is what the autograder reads, so the two sets of cases are
// deliberately different numbers.
static void probe(const char* name, void* p) {
  if (p == NULL) printf("@ %s=-1\n", name);
  else printf("@ %s=%ld\n", name, (long)((unsigned char*)p - g_arena));
}

static int cmd_alloc(const Options* o) {
  (void)o;
  gw_reset();
  probe("zero", gw_alloc(0));
  probe("first", gw_alloc(1));
  probe("second", gw_alloc(1));
  printf("@ used_after_two=%zu\n", g_arena_used);

  gw_reset();
  probe("whole", gw_alloc(GW_ARENA_BYTES));
  printf("@ used_after_whole=%zu\n", g_arena_used);
  probe("one_past", gw_alloc(1));
  printf("@ refused_after_one_past=%ld\n", g_alloc_refused);

  gw_reset();
  gw_alloc(GW_ARENA_BYTES - 8);
  probe("too_big_for_the_hole", gw_alloc(16));
  printf("@ used_after_refusal=%zu\n", g_arena_used);
  probe("fits_the_hole", gw_alloc(8));

  // The same refusal from an UNALIGNED cursor. An allocator that writes the aligned base
  // back before deciding has eaten the padding of a request it then refused, and the two
  // cases above cannot see it because the cursor was already on a boundary.
  gw_reset();
  gw_alloc(GW_ARENA_BYTES - 9);
  probe("odd_too_big", gw_alloc(16));
  printf("@ used_after_unaligned_refusal=%zu\n", g_arena_used);
  probe("odd_fits", gw_alloc(8));

  gw_reset();
  probe("before_freeze", gw_alloc(8));
  gw_freeze();
  probe("after_freeze", gw_alloc(8));
  printf("@ after_freeze_count=%ld\n", g_alloc_after_freeze);
  printf("@ refused_total=%ld\n", g_alloc_refused);
  printf("@ granted_total=%ld\n", g_alloc_calls);
  gw_reset();
  return 0;
}

static int cmd_decode(const Options* o) {
  (void)o;
  Frame fr;
  memset(&fr, 0, sizeof fr);
  uint8_t one[7] = {0x05, FUNC_READ_INPUT, 0x02, 0x12, 0x34, 0, 0};
  uint16_t c = crc16(one, 5);
  one[5] = (uint8_t)(c & 0xFFu);
  one[6] = (uint8_t)(c >> 8);
  printf("@ rc_ok=%d\n", frame_decode(one, 7, &fr));
  printf("@ ok_nreg=%d\n", fr.nreg);
  printf("@ ok_addr=%d\n", (int)fr.addr);
  printf("@ ok_reg0=%u\n", (unsigned)fr.reg[0]);

  uint8_t t[7];
  memcpy(t, one, 7);
  printf("@ rc_short=%d\n", frame_decode(t, 4, &fr));
  t[1] = 0x03;
  printf("@ rc_func=%d\n", frame_decode(t, 7, &fr));
  memcpy(t, one, 7);
  t[2] = 0x04;
  printf("@ rc_bytecount=%d\n", frame_decode(t, 7, &fr));

  // The other direction, and the one a bounds check alone does not catch: a byte count
  // SMALLER than the payload, with a checksum that is valid for the frame as it stands. A
  // decoder that only asks whether the byte count is too big accepts this and invents a
  // register the device never declared.
  {
    uint8_t small[7] = {0x05, FUNC_READ_INPUT, 0x00, 0x12, 0x34, 0, 0};
    uint16_t sc = crc16(small, 5);
    small[5] = (uint8_t)(sc & 0xFFu);
    small[6] = (uint8_t)(sc >> 8);
    fr.nreg = -11;
    printf("@ rc_bytecount_small=%d\n", frame_decode(small, 7, &fr));
    printf("@ bytecount_small_left_out_alone=%d\n", (fr.nreg == -11) ? 1 : 0);
  }

  memcpy(t, one, 7);
  t[3] ^= 0x40u;
  fr.reg[0] = 0xBEEF;
  fr.nreg = -7;
  printf("@ rc_crc=%d\n", frame_decode(t, 7, &fr));
  printf("@ crc_left_out_alone=%d\n", (fr.reg[0] == 0xBEEF && fr.nreg == -7) ? 1 : 0);

  // More registers than the caller's buffer holds, with a deliberately WRONG checksum: the
  // capacity check has to come first, because writing them is what would corrupt memory.
  {
    int nreg = BURST_LEN + 4;
    size_t len = (size_t)(3 + 2 * nreg + 2);
    uint8_t big[3 + 2 * (BURST_LEN + 4) + 2];
    memset(big, 0, sizeof big);
    big[0] = 0x01;
    big[1] = FUNC_READ_INPUT;
    big[2] = (uint8_t)(2 * nreg);
    printf("@ rc_capacity=%d\n", frame_decode(big, len, &fr));
  }

  capture_build();
  int rc = frame_decode(&g_capture[0], FRAME_BYTES, &fr);
  printf("@ rc_capture0=%d\n", rc);
  printf("@ capture0_reg0=%u\n", (unsigned)fr.reg[0]);
  printf("@ capture0_reg31=%u\n", (unsigned)fr.reg[BURST_LEN - 1]);
  printf("@ rc_capture_corrupt=%d\n",
         frame_decode(&g_capture[(size_t)CORRUPT_PHASE * FRAME_BYTES], FRAME_BYTES, &fr));
  return 0;
}

static int cmd_qprobe(const Options* o) {
  (void)o;
  int32_t x[BURST_LEN];
  int32_t base = 2048 << SAMPLE_SHIFT;
  for (int i = 0; i < BURST_LEN; i++) x[i] = 2048 << SAMPLE_SHIFT;
  printf("@ q_on_baseline=%d\n", q_health(x, BURST_LEN, base));
  printf("@ q_n_zero=%d\n", q_health(x, 0, base));
  printf("@ q_n_negative=%d\n", q_health(x, -3, base));
  printf("@ q_baseline_zero=%d\n", q_health(x, BURST_LEN, 0));
  printf("@ q_baseline_negative=%d\n", q_health(x, BURST_LEN, -base));
  for (int i = 0; i < BURST_LEN; i++) x[i] = 2049 << SAMPLE_SHIFT;
  printf("@ q_2049=%d\n", q_health(x, BURST_LEN, base));
  for (int i = 0; i < BURST_LEN; i++) x[i] = 65535 << SAMPLE_SHIFT;
  printf("@ q_full_scale=%d\n", q_health(x, BURST_LEN, 65535 << SAMPLE_SHIFT));
  printf("@ q_full_over_small=%d\n", q_health(x, BURST_LEN, 1 << SAMPLE_SHIFT));
  for (int i = 0; i < BURST_LEN; i++) x[i] = (int32_t)(i + 1) << SAMPLE_SHIFT;
  printf("@ q_ramp=%d\n", q_health(x, BURST_LEN, 1 << SAMPLE_SHIFT));
  printf("@ q_ramp_half=%d\n", q_health(x, BURST_LEN / 2, 1 << SAMPLE_SHIFT));
  // Two cases built so that rounding and truncating give DIFFERENT answers. Nothing in the
  // capture separates them: a mean that rounds up only changes the integer square root when
  // it carries the value across a perfect square, which on real bursts happens about once
  // in five million readings. The contract says truncate, so the contract is graded.
  {
    int32_t tiny[2] = {7, 0};    // squares sum to 49; 49/2 is 24 and rounds to 25
    printf("@ q_truncates_the_mean=%d\n", q_health(tiny, 2, 65536));
    int32_t three[4] = {3, 3, 3, 3};   // (3 << 16) / 7 is 28086.857...
    printf("@ q_truncates_the_divide=%d\n", q_health(three, 4, 7));
  }
  return 0;
}

static int cmd_sfprobe(const Options* o) {
  (void)o;
  Record slots[5];
  Record out[9];
  Ring r;
  memset(&r, 0, sizeof r);
  r.slot = slots;
  r.cap = 5;
  r.next_seq = 1;
  printf("@ push_a=%u\n", sf_push(&r, 0, 0, 900));
  printf("@ push_b=%u\n", sf_push(&r, 1, 0, 901));
  printf("@ push_c=%u\n", sf_push(&r, 2, 0, 902));
  int n = sf_take(&r, out, 2);
  printf("@ take2_n=%d\n", n);
  printf("@ take2_first=%u\n", out[0].seq);
  printf("@ take2_last=%u\n", out[n - 1].seq);
  printf("@ count_after_take=%d\n", r.count);
  n = sf_take(&r, out, 2);
  printf("@ retake_first=%u\n", out[0].seq);
  printf("@ ack2_dropped=%d\n", sf_ack(&r, 2));
  printf("@ count_after_ack=%d\n", r.count);
  printf("@ ack2_again=%d\n", sf_ack(&r, 2));
  printf("@ push_d=%u\n", sf_push(&r, 3, 1, 903));
  printf("@ push_e=%u\n", sf_push(&r, 4, 1, 904));
  printf("@ push_f=%u\n", sf_push(&r, 0, 1, 905));
  printf("@ push_g=%u\n", sf_push(&r, 1, 1, 906));
  printf("@ count_when_full=%d\n", r.count);
  printf("@ push_h_refused=%u\n", sf_push(&r, 2, 1, 907));
  printf("@ next_seq_after_refusal=%u\n", r.next_seq);
  printf("@ count_after_refusal=%d\n", r.count);
  printf("@ refused_total=%ld\n", r.refused);
  n = sf_take(&r, out, 9);
  printf("@ drain_n=%d\n", n);
  printf("@ drain_first=%u\n", out[0].seq);
  printf("@ drain_last=%u\n", out[n - 1].seq);
  printf("@ drain_machine_first=%u\n", (unsigned)out[0].machine);
  printf("@ drain_wire_first=%u\n", (unsigned)out[0].wire);
  printf("@ drain_hour_last=%u\n", (unsigned)out[n - 1].hour);
  printf("@ ack_low_dropped=%d\n", sf_ack(&r, 4));
  printf("@ count_after_partial_ack=%d\n", r.count);
  printf("@ ack_all_dropped=%d\n", sf_ack(&r, 99));
  printf("@ count_at_end=%d\n", r.count);
  printf("@ take_on_empty=%d\n", sf_take(&r, out, 9));
  printf("@ peak=%ld\n", r.peak);
  return 0;
}

static int cmd_arena(const Options* o) {
  gw_reset();
  Gateway g;
  gateway_start(&g, o->capacity > 0 ? o->capacity : RING_CAPACITY);
  printf("@ started=%d\n", g.ok);
  printf("@ arena_bytes=%d\n", GW_ARENA_BYTES);
  printf("@ arena_used=%zu\n", g_arena_used);
  printf("@ arena_peak=%zu\n", g_arena_peak);
  printf("@ arena_free=%zu\n", (size_t)GW_ARENA_BYTES - g_arena_used);
  printf("@ alloc_calls=%ld\n", g_alloc_calls);
  printf("@ alloc_refused=%ld\n", g_alloc_refused);
  // One allocation attempted after the door closed. A correct gateway never does this; the
  // counter exists so that a gateway which DOES can say so out loud.
  void* p = gw_alloc(8);
  printf("@ post_freeze_null=%d\n", p == NULL);
  printf("@ alloc_after_freeze=%ld\n", g_alloc_after_freeze);
  return 0;
}

// =======================================================================================
// The self-test: your feedback loop. Every case below is hand-workable.
// =======================================================================================

static void check_gw_alloc(void) {
  gw_reset();
  require(gw_alloc(0) == NULL, "gw_alloc(0) must return NULL — there is nothing to hand out");
  unsigned char* a = (unsigned char*)gw_alloc(1);
  require(a == g_arena, "the first allocation out of an empty arena must be &g_arena[0]");
  unsigned char* b = (unsigned char*)gw_alloc(1);
  require(b == g_arena + 8,
          "two 1-byte allocations must be 8 bytes apart, not %ld — align the cursor up to 8 "
          "before carving, or an int64_t lands across a boundary",
          (long)(b - a));

  gw_reset();
  void* big = gw_alloc(GW_ARENA_BYTES);
  require(big != NULL, "a request for the whole arena must succeed on an empty arena");
  require(g_arena_used == (size_t)GW_ARENA_BYTES, "the cursor must sit at the end, it is %zu",
          g_arena_used);
  require(gw_alloc(1) == NULL, "one byte past a full arena must return NULL");
  require(g_alloc_refused == 1, "a refusal must be counted in g_alloc_refused, it is %ld",
          g_alloc_refused);

  gw_reset();
  gw_alloc(GW_ARENA_BYTES - 8);
  require(gw_alloc(16) == NULL, "16 bytes into an 8-byte hole must be refused");
  require(g_arena_used == (size_t)GW_ARENA_BYTES - 8,
          "a REFUSED allocation must leave the cursor where it was; it moved to %zu from %d. "
          "The remaining budget cannot depend on the requests that failed.",
          g_arena_used, GW_ARENA_BYTES - 8);
  require(gw_alloc(8) != NULL,
          "after a refusal the 8 bytes that were left must still be available");

  gw_reset();
  gw_freeze();
  require(gw_alloc(8) == NULL, "after gw_freeze() every allocation must return NULL");
  require(g_alloc_after_freeze == 1,
          "an allocation attempted after start-up must be counted in g_alloc_after_freeze, "
          "it is %ld", g_alloc_after_freeze);
  require(g_alloc_refused == 1, "it must be counted in g_alloc_refused too, it is %ld",
          g_alloc_refused);
  gw_reset();
}

static void check_frame_decode(void) {
  Frame fr;
  memset(&fr, 0xAA, sizeof fr);
  uint8_t empty[5] = {0x01, FUNC_READ_INPUT, 0x00, 0x00, 0x00};
  uint16_t c = crc16(empty, 3);
  empty[3] = (uint8_t)(c & 0xFF);
  empty[4] = (uint8_t)(c >> 8);
  require(frame_decode(empty, 5, &fr) == FD_OK,
          "a five-byte frame with a zero byte count and a good checksum is valid; you "
          "returned %d", frame_decode(empty, 5, &fr));
  require(fr.nreg == 0 && fr.addr == 1, "nreg must be 0 and addr 1, you gave %d and %d",
          fr.nreg, (int)fr.addr);
  require(frame_decode(empty, 4, &fr) == FD_SHORT,
          "four bytes is shorter than a header plus a checksum; expected FD_SHORT (%d)",
          FD_SHORT);

  // One register, 0x1234, big-endian on the wire.
  uint8_t one[7] = {0x05, FUNC_READ_INPUT, 0x02, 0x12, 0x34, 0x00, 0x00};
  c = crc16(one, 5);
  one[5] = (uint8_t)(c & 0xFF);
  one[6] = (uint8_t)(c >> 8);
  require(frame_decode(one, 7, &fr) == FD_OK, "a one-register frame must decode");
  require(fr.nreg == 1, "nreg must be 1, you gave %d", fr.nreg);
  require(fr.reg[0] == 0x1234,
          "the payload bytes 0x12 0x34 are one BIG-endian register, 0x1234 = 4660. You "
          "returned %u (0x%04X) — if that is 13330 you assembled it little-endian.",
          (unsigned)fr.reg[0], (unsigned)fr.reg[0]);

  uint8_t bad_func[7];
  memcpy(bad_func, one, 7);
  bad_func[1] = 0x03;
  require(frame_decode(bad_func, 7, &fr) == FD_FUNC,
          "function code 0x03 is not this map's; expected FD_FUNC (%d)", FD_FUNC);

  uint8_t bad_count[7];
  memcpy(bad_count, one, 7);
  bad_count[2] = 0x04;
  require(frame_decode(bad_count, 7, &fr) == FD_BYTECOUNT,
          "a byte count of 4 in a 7-byte frame is a mismatch; expected FD_BYTECOUNT (%d)",
          FD_BYTECOUNT);

  uint8_t corrupt[7];
  memcpy(corrupt, one, 7);
  corrupt[3] ^= 0x40;
  uint16_t before = fr.reg[0];
  require(frame_decode(corrupt, 7, &fr) == FD_CRC,
          "one flipped payload bit must fail the checksum; expected FD_CRC (%d)", FD_CRC);
  require(fr.reg[0] == before,
          "a frame that failed its checksum must leave *out untouched; reg[0] changed from "
          "%u to %u. A half-filled frame is worse than no frame.", (unsigned)before,
          (unsigned)fr.reg[0]);

  // And the real capture: the frame the generator corrupts must be rejected, its neighbour
  // must not.
  capture_build();
  long bad_i = CORRUPT_PHASE;
  require(frame_decode(&g_capture[(size_t)bad_i * FRAME_BYTES], FRAME_BYTES, &fr) == FD_CRC,
          "frame %ld of the capture carries a flipped bit and must be rejected", bad_i);
  require(frame_decode(&g_capture[(size_t)(bad_i + 1) * FRAME_BYTES], FRAME_BYTES, &fr) == FD_OK,
          "frame %ld of the capture is intact and must decode", bad_i + 1);
}

static void check_q_health(void) {
  int32_t x[BURST_LEN];
  for (int i = 0; i < BURST_LEN; i++) x[i] = 2048 << SAMPLE_SHIFT;
  int32_t base = 2048 << SAMPLE_SHIFT;
  int32_t got = q_health(x, BURST_LEN, base);
  require(got == 65536,
          "a burst sitting exactly on its baseline must read 1.0, which is 65536 in Q16.16; "
          "you returned %d (%.6f)", got, got / 65536.0);
  require(q_health(x, 0, base) == -1, "n <= 0 must return -1");
  require(q_health(x, BURST_LEN, 0) == -1, "a baseline of 0 must return -1");

  for (int i = 0; i < BURST_LEN; i++) x[i] = 4096 << SAMPLE_SHIFT;
  got = q_health(x, BURST_LEN, base);
  require(got == 131072, "twice the baseline is 2.0 = 131072 in Q16.16; you returned %d", got);

  // Full scale. Every sample at 65535 counts: the squares sum to about 9.0e15, four million
  // times the largest value an int32_t holds. An accumulator of the wrong width does not
  // warn, it wraps.
  for (int i = 0; i < BURST_LEN; i++) x[i] = 65535 << SAMPLE_SHIFT;
  int32_t full = q_health(x, BURST_LEN, 65535 << SAMPLE_SHIFT);
  require(full == 65536,
          "a full-scale burst against a full-scale baseline is still 1.0 (65536); you "
          "returned %d. The sum of 32 squares of 16776960 is 8.99e15 — accumulate in "
          "int64_t, and shift rms into the numerator in int64_t as well.", full);

  // Truncation, stated out loud: 2049 counts against a 2048 baseline is 1.00048828125, and
  // Q16.16 cannot hold it exactly.
  for (int i = 0; i < BURST_LEN; i++) x[i] = 2049 << SAMPLE_SHIFT;
  int32_t t = q_health(x, BURST_LEN, base);
  require(t == 65568,
          "2049 counts against a 2048 baseline is 65568 in Q16.16 (1.00048828125 exactly); "
          "you returned %d. Every step truncates towards zero — no rounding anywhere.", t);
}

static void check_store_and_forward(void) {
  gw_reset();
  Record slots[4];
  Record out[8];
  Ring r;
  memset(&r, 0, sizeof r);
  r.slot = slots;
  r.cap = 4;
  r.next_seq = 1;

  require(sf_push(&r, 0, 0, 100) == 1, "the first push must be assigned sequence number 1");
  require(sf_push(&r, 1, 0, 101) == 2, "the second push must be 2");
  require(sf_push(&r, 2, 0, 102) == 3, "the third push must be 3");
  require(r.count == 3, "three pushes leave count at 3, it is %d", r.count);

  require(sf_take(&r, out, 2) == 2, "sf_take with max 2 over 3 held records returns 2");
  require(out[0].seq == 1 && out[1].seq == 2, "sf_take hands back the OLDEST records first");
  require(r.count == 3,
          "sf_take is a peek, not a pop: the ring still holds 3 records, it holds %d. The "
          "link is allowed to lose the acknowledgement, and then the gateway has to be able "
          "to offer the same records again.", r.count);
  require(sf_take(&r, out, 2) == 2 && out[0].seq == 1,
          "calling sf_take twice without an acknowledgement must return the same records");

  require(sf_ack(&r, 2) == 2, "acknowledging sequence 2 drops records 1 and 2");
  require(r.count == 1, "one record is left, you have %d", r.count);
  require(sf_ack(&r, 2) == 0,
          "a repeated acknowledgement drops nothing and is not an error — acknowledgements "
          "arrive late and arrive twice");
  require(sf_take(&r, out, 8) == 1 && out[0].seq == 3,
          "the record left behind is sequence 3, and it must not have been dropped by an "
          "acknowledgement for 2");

  // Wrap: the ring is 4 slots, and the indices must come back round.
  require(sf_push(&r, 3, 1, 103) == 4, "the next sequence number is 4");
  require(sf_push(&r, 4, 1, 104) == 5, "then 5");
  require(sf_push(&r, 5, 1, 105) == 6, "then 6");
  require(r.count == 4, "the ring is now full with 4 records, it holds %d", r.count);
  uint32_t refused = sf_push(&r, 6, 1, 106);
  require(refused == SF_REFUSED,
          "a push into a full ring must return SF_REFUSED (0) rather than overwriting the "
          "oldest unacknowledged record; you returned %u", (unsigned)refused);
  require(r.refused == 1, "the refusal must be counted, r.refused is %ld", r.refused);
  require(r.count == 4, "a refused push must not change the count, it is %d", r.count);
  require(r.next_seq == 7,
          "a refused push must NOT consume a sequence number — next_seq is %u and must be 7. "
          "A gap in the delivered sequence is how the historian detects a loss in transit, "
          "so a refusal at the door must not look like one.", (unsigned)r.next_seq);
  require(sf_take(&r, out, 8) == 4 && out[0].seq == 3 && out[3].seq == 6,
          "after the wrap the four held records are 3, 4, 5 and 6, oldest first");
  require(sf_ack(&r, 6) == 4 && r.count == 0, "acknowledging 6 empties the ring");
  require(sf_take(&r, out, 8) == 0, "an empty ring hands back nothing");
}

static void check_exactly_once_over_an_outage(void) {
  gw_reset();
  Record slots[16];
  Record out[8];
  Ring r;
  memset(&r, 0, sizeof r);
  r.slot = slots;
  r.cap = 16;
  r.next_seq = 1;
  Historian hist = {0, 0, 0, 0};

  // Ten readings arrive while the link is down, then it comes back and the first
  // acknowledgement is lost.
  for (int i = 0; i < 10; i++) require(sf_push(&r, (uint8_t)i, 0, (uint16_t)(200 + i)) != 0,
                                       "push %d should have been accepted", i);
  int n = sf_take(&r, out, 8);
  require(n == 8, "sf_take with max 8 over 10 held records returns 8, it returned %d", n);
  for (int i = 0; i < n; i++) historian_accept(&hist, &out[i]);
  // The acknowledgement never arrives. The gateway offers the same eight again.
  n = sf_take(&r, out, 8);
  for (int i = 0; i < n; i++) historian_accept(&hist, &out[i]);
  require(hist.duplicates == 8,
          "the historian should have rejected the eight records it had already stored; it "
          "rejected %ld. sf_take must hand back the same records when nothing was "
          "acknowledged.", hist.duplicates);
  sf_ack(&r, out[n - 1].seq);
  n = sf_take(&r, out, 8);
  for (int i = 0; i < n; i++) historian_accept(&hist, &out[i]);
  sf_ack(&r, out[n - 1].seq);
  require(hist.accepted == 10,
          "all ten readings must reach the historian exactly once; it stored %ld",
          hist.accepted);
  require(hist.sum_seq == 55,
          "the ten stored sequence numbers must be 1..10, which sum to 55; they sum to %llu",
          hist.sum_seq);
  require(r.count == 0, "the ring must be empty once everything is acknowledged, it holds %d",
          r.count);
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
  struct {
    const char* name;
    void (*fn)(void);
  } checks[] = {
      {"exercise 1  gw_alloc", check_gw_alloc},
      {"exercise 2  frame_decode", check_frame_decode},
      {"exercise 3  q_health", check_q_health},
      {"exercise 4  sf_push / sf_take / sf_ack", check_store_and_forward},
      {"exercise 4  exactly once over an outage", check_exactly_once_over_an_outage},
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

static int usage(void) {
  fprintf(stderr,
          "usage: lesson_bin <command> [options]\n"
          "  selftest    run the built-in checks for the four exercises\n"
          "  facts       print the design numbers this gateway was built to\n"
          "  capture     checksums of the recorded capture, and --dump N frames of hex\n"
          "  health      decode the capture and publish a health index per reading\n"
          "  forward     store and forward under --scenario clean|outage|lostack|overflow\n"
          "  arena       what start-up took out of the arena, and what is left\n"
          "  alloc       a scripted probe of gw_alloc\n"
          "  decode      a scripted probe of frame_decode\n"
          "  qprobe      a scripted probe of q_health\n"
          "  sfprobe     a scripted probe of sf_push / sf_take / sf_ack\n"
          "options: --n --dump --scenario --capacity\n");
  return 3;
}

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  Options o = {0, 0, "clean", 0};
  const char* cmd = argv[1];
  for (int i = 2; i < argc; i++) {
    const char* k = argv[i];
    if (i + 1 >= argc) {
      fprintf(stderr, "error: %s needs a value\n", k);
      return 3;
    }
    const char* v = argv[++i];
    if (!strcmp(k, "--n")) o.n = strtol(v, NULL, 10);
    else if (!strcmp(k, "--dump")) o.dump = strtol(v, NULL, 10);
    else if (!strcmp(k, "--scenario")) o.scenario = v;
    else if (!strcmp(k, "--capacity")) o.capacity = (int)strtol(v, NULL, 10);
    else {
      fprintf(stderr, "error: unknown option %s\n", k);
      return 3;
    }
  }
  if (!strcmp(cmd, "selftest")) return cmd_selftest(&o);
  if (!strcmp(cmd, "facts")) return cmd_facts(&o);
  if (!strcmp(cmd, "capture")) return cmd_capture(&o);
  if (!strcmp(cmd, "health")) return cmd_health(&o);
  if (!strcmp(cmd, "forward")) return cmd_forward(&o);
  if (!strcmp(cmd, "arena")) return cmd_arena(&o);
  if (!strcmp(cmd, "alloc")) return cmd_alloc(&o);
  if (!strcmp(cmd, "decode")) return cmd_decode(&o);
  if (!strcmp(cmd, "qprobe")) return cmd_qprobe(&o);
  if (!strcmp(cmd, "sfprobe")) return cmd_sfprobe(&o);
  return usage();
}
