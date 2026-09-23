/* P02-L10-streaming-scanner-in-c — the streaming field scanner, in C.
 *
 * Four functions are yours to write. Everything else — the synthetic export generator, the
 * chunk driver, the money parser, argument parsing, the self-test, the timing and the RSS
 * reporting — is already here and is not part of the exercise.
 *
 *     make test                      build this file and run its self-test
 *     make bench BYTES=33554432      scan a generated export and print throughput and peak RSS
 *     make paths                     show what the build discovered
 *     make clean                     delete every compiled artefact (read section 13 first)
 *
 * Only the C standard library and POSIX getrusage are used. No third-party header, no CSV
 * library: the whole point is that the parser is a state machine you can hold in your head
 * and that its memory is a constant you can name. How big the constant is, and how much
 * faster this is than the Python scanner, are not for this comment to say — sections 9 to 11
 * of the notebook measure both on your machine, and those measured numbers are the only ones
 * you should ever quote.
 */
/* clock_gettime and CLOCK_MONOTONIC are POSIX, not ISO C. Under -std=c11, glibc (Linux)
   hides them unless asked before the first header; macOS shows them regardless. */
#define _DEFAULT_SOURCE
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

/* ======================================================================================
 * The export's shape. Given.
 * ====================================================================================== */

/* Six columns, no header row: the schema travels in the manifest beside the export, not in
 * its first line. Column indices are named so the aggregation reads as policy, not magic. */
enum {
  kColumns = 6,
  kColDocId = 0,
  kColSupplier = 1,
  kColInvoice = 2,
  kColAmount = 3,
  kColCurrency = 4,
  kColNotes = 5
};

/* The field buffer. This is the number that makes the memory constant: no field, however
 * long the export makes it, is allowed to grow the scanner. A field longer than this is
 * truncated and counted, which is a DECLARED policy — the scanner reports how often it fired
 * so nobody downstream has to guess. */
enum { kFieldCap = 128 };

/* The four states of the parser. A streaming parser's state is the only thing that survives
 * a chunk boundary, so it has to be small enough to write down. */
enum {
  kFieldStart = 0, /* between fields; a '"' HERE opens a quoted field, anywhere else it is data */
  kUnquoted = 1,   /* reading an unquoted field */
  kQuoted = 2,     /* inside a quoted field; ',' and '\n' are ordinary data here */
  kQuoteSeen = 3   /* inside a quoted field, having just read one '"' */
};

typedef struct {
  int state;
  int col;            /* index of the field being built, 0..kColumns */
  size_t field_len;   /* bytes currently held in `field` */
  size_t seen_len;    /* bytes the current field HAS, including any truncated away */
  char field[kFieldCap];

  long long records;           /* complete records */
  long long fields;            /* completed fields */
  long long bytes;             /* bytes fed to scan_chunk */
  long long amount_cents;      /* running total of column kColAmount, in integer cents */
  long long eur_records;       /* fields in column kColCurrency equal to "EUR" */
  long long max_field_len;     /* longest field seen, truncated part included */
  long long field_overflow;    /* fields longer than kFieldCap */
  long long embedded_newlines; /* '\n' bytes inside a quoted field */
  long long bad_arity;         /* complete records whose field count was not kColumns */
  int truncated_tail;          /* 1 if the stream ended part-way through a record */
} Scanner;

static void scanner_init(Scanner* s) { memset(s, 0, sizeof *s); }

/* The TODO signal. Exit code 3 is what the notebook turns back into a Python
 * NotImplementedError, so the grader can tell "not written yet" from "written and wrong". */
static void not_implemented(const char* name) {
  fprintf(stderr, "NOT_IMPLEMENTED %s\n", name);
  exit(3);
}

/* GIVEN: money, in integer cents, or a refusal.
 *
 * "1234.56" -> 123456, "-40.00" -> -4000, "7" -> 700. Anything else — a thousands separator,
 * one decimal place, three decimal places, a stray letter — returns false and adds nothing.
 * Money never touches a double in this lesson: 0.10 has no exact binary representation, and
 * an export of a million invoices is long enough for that to show up in the total. */
static bool parse_cents(const char* s, size_t len, long long* out) {
  size_t i = 0;
  int neg = 0, digits = 0, fdigits = 0;
  long long whole = 0, frac = 0;
  if (len == 0) return false;
  if (s[0] == '-') {
    neg = 1;
    i = 1;
  }
  for (; i < len && s[i] != '.'; ++i) {
    if (s[i] < '0' || s[i] > '9') return false;
    whole = whole * 10 + (s[i] - '0');
    ++digits;
    if (whole > 900000000LL) return false; /* refuse to overflow rather than wrap silently */
  }
  if (digits == 0) return false;
  if (i < len) {
    for (++i; i < len; ++i) {
      if (s[i] < '0' || s[i] > '9') return false;
      frac = frac * 10 + (s[i] - '0');
      ++fdigits;
    }
    if (fdigits != 2) return false;
  }
  {
    long long cents = whole * 100 + frac;
    *out = neg ? -cents : cents;
  }
  return true;
}

/* GIVEN: a record has ended. Arity is checked here so exercise 2 never has to think about
 * records, only about fields. A record with the wrong number of fields still counts as a
 * record — it arrived, it was terminated, and somebody downstream has to be told about it. */
static void record_complete(Scanner* s) {
  if (s->col != kColumns) s->bad_arity++;
  s->records++;
  s->col = 0;
}

/* ======================================================================================
 * EXERCISE 1 — field_push: put one byte into a buffer that never grows.
 *
 * Append `c` to the current field, but ONLY while there is room. The whole memory argument
 * of this lesson rests on this function refusing to allocate.
 *
 * Three things happen here:
 *   * `seen_len` counts every byte the field has, whether or not it was kept. That is how a
 *     constant-memory scanner can still report the true length of a field it threw away.
 *   * `field` / `field_len` hold at most kFieldCap bytes. Never write past the end.
 *   * `field_overflow` counts FIELDS that did not fit, not BYTES that did not fit. A field
 *     of 5000 bytes is one overflow, not 4872 of them. Fire it exactly once, at the moment
 *     the field first exceeds the cap.
 *
 * Worked example, with kFieldCap == 128:
 *   pushing 130 bytes into a fresh field leaves field_len == 128, seen_len == 130,
 *   field_overflow == 1 — and the first 128 bytes, in order, in `field`.
 *   Pushing 128 bytes leaves field_len == 128, seen_len == 128, field_overflow == 0:
 *   a field that exactly fills the buffer has not overflowed.
 *
 * Cost: O(1), no allocation, no call to anything.
 * ====================================================================================== */
static void field_push(Scanner* s, char c) {
  /* YOUR CODE HERE */
  (void)s;
  (void)c;
  not_implemented("field_push");
}

/* ======================================================================================
 * EXERCISE 2 — field_complete: the field is finished; fold it into the running totals.
 *
 * This is where a streaming scanner earns its name. The field is about to be forgotten, so
 * every question anybody will ever ask about it has to be answered NOW, into a fixed set of
 * counters. Do, in this order or any order that gives the same answers:
 *
 *   * count the field: `fields`;
 *   * raise `max_field_len` to `seen_len` if this field is the longest so far — seen_len,
 *     not field_len, or a truncated field reports the cap instead of its real size;
 *   * if this is column kColAmount, call parse_cents(s->field, s->field_len, &cents) and add
 *     `cents` to `amount_cents` ONLY when it returns true. An unparseable amount adds
 *     nothing; it does not add zero-as-a-guess and it does not abort the scan;
 *   * if this is column kColCurrency and the field is exactly the three bytes "EUR",
 *     increment `eur_records`;
 *   * advance `col` by one, and reset `field_len` and `seen_len` to zero so the next field
 *     starts clean.
 *
 * Worked example: with col == kColAmount and the field holding "1234.56",
 *   fields goes up by one, amount_cents goes up by 123456, col becomes kColCurrency,
 *   and field_len and seen_len are both back to 0.
 *
 * Do NOT touch `records` here. A field ending and a record ending are different events, and
 * record_complete() above already owns the second one.
 *
 * Cost: O(length of this field), which is bounded by kFieldCap. That bound is the lesson.
 * ====================================================================================== */
static void field_complete(Scanner* s) {
  /* YOUR CODE HERE */
  (void)s;
  not_implemented("field_complete");
}

/* ======================================================================================
 * EXERCISE 3 — scan_chunk: the resumable state machine.
 *
 * Consume buf[0..n) one byte at a time, updating `s`. The caller hands you the export in
 * chunks of whatever size it likes, and it may split the stream ANYWHERE — between the two
 * quotes of an escaped "", between the CR and the LF, in the middle of a supplier name. The
 * only thing that crosses a chunk boundary is the Scanner struct, so everything you need to
 * remember has to live in it. Nothing outside it may be assumed.
 *
 * Count every byte you consume in `s->bytes`.
 *
 * The grammar, which is RFC 4180 with two leniencies named below:
 *
 *   kFieldStart  '"'   -> kQuoted (this quote opens a quoted field; it is not data)
 *                '\r'  -> dropped, stay in kFieldStart
 *                ','   -> field_complete(); stay in kFieldStart   (an empty field)
 *                '\n'  -> field_complete(); record_complete(); stay in kFieldStart
 *                else  -> field_push(); go to kUnquoted
 *
 *   kUnquoted    ','   -> field_complete(); go to kFieldStart
 *                '\n'  -> field_complete(); record_complete(); go to kFieldStart
 *                '\r'  -> dropped                                  (leniency 1: see below)
 *                else  -> field_push()
 *
 *   kQuoted      '"'   -> go to kQuoteSeen (do not push it: it may be a closing quote)
 *                '\n'  -> s->embedded_newlines++; field_push()     (a newline INSIDE a field)
 *                else  -> field_push()
 *
 *   kQuoteSeen   '"'   -> field_push('"'); go back to kQuoted      (an escaped "" )
 *                ','   -> field_complete(); go to kFieldStart
 *                '\n'  -> field_complete(); record_complete(); go to kFieldStart
 *                '\r'  -> dropped
 *                else  -> field_push(); go to kUnquoted            (leniency 2: see below)
 *
 * Leniency 1: a bare CR outside a quoted field is dropped rather than treated as data, so
 * an export written with CRLF line endings scans identically to one written with LF. Inside
 * a quoted field a CR is data, because inside quotes everything is data.
 * Leniency 2: bytes after a closing quote ( "abc"def ) are a malformed field. A strict
 * parser would reject the file. This one keeps the bytes and carries on, because an export
 * scanner that dies on record 4,000,001 has told you nothing about the first four million.
 *
 * Worked example, and the reason the chunk size must not matter:
 *   scanning  a,"b""c",d  in one chunk, and scanning it in twenty chunks of one byte each,
 *   must leave the scanner in exactly the same state, with the middle field holding b"c.
 *
 * Cost: O(n), one pass, no allocation, no lookahead, no push-back.
 * ====================================================================================== */
static void scan_chunk(Scanner* s, const char* buf, size_t n) {
  /* YOUR CODE HERE */
  (void)s;
  (void)buf;
  (void)n;
  not_implemented("scan_chunk");
}

/* ======================================================================================
 * EXERCISE 4 — scan_finish: the stream stopped. Decide what the tail was.
 *
 * Called once, after the last chunk. An export that was cut off — a copy that ran out of
 * disk, a pipe that closed, a reader that stopped at a byte limit — ends part-way through a
 * record, and the half-record it leaves behind is not a record. Two things follow: it is not
 * counted in `records`, and its PENDING field — the one the cut landed in — is not counted
 * either, because you cannot know whether you have all of it. Fields of that record which
 * were already terminated by their own comma keep their contribution; the scanner raises
 * `truncated_tail` so the operator knows the last record is incomplete. That is a convention,
 * not a law of nature, and section 12 of the notebook measures what it costs.
 *
 * But a file that simply has no final newline is NOT truncated. RFC 4180 says in as many
 * words that the last record may or may not have an ending line break, so a final record
 * with all kColumns fields present is a record, newline or no newline. Telling those two
 * apart is the whole exercise.
 *
 * The rule, in order:
 *   1. state is kQuoted — the stream stopped inside an unterminated quoted field. Set
 *      truncated_tail = 1, discard the partial field, and count nothing.
 *   2. nothing is pending — state is kFieldStart, col == 0 and field_len == 0. The stream
 *      ended cleanly on a record boundary. Do nothing at all.
 *   3. completing the pending field would finish a whole record — that is, col + 1 ==
 *      kColumns. Call field_complete() then record_complete(). This is the missing-final-
 *      newline case, and it is legal.
 *   4. anything else is a short record: set truncated_tail = 1 and count nothing.
 *
 * In every case, leave the scanner clean afterwards: state kFieldStart, col 0, field_len 0,
 * seen_len 0. Note that kQuoteSeen is NOT truncation — a field that ended with its closing
 * quote is finished, and rule 3 or 4 decides the record.
 *
 * Worked example:
 *   ...,EUR,ok            (no newline, six fields) -> one more record, truncated_tail 0
 *   ...,EUR               (no newline, five fields) -> no record at all, truncated_tail 1
 *   ...,"unterminated     (stopped inside quotes)   -> no record at all, truncated_tail 1
 *
 * Cost: O(1) plus one field.
 * ====================================================================================== */
static void scan_finish(Scanner* s) {
  /* YOUR CODE HERE */
  (void)s;
  not_implemented("scan_finish");
}

/* ======================================================================================
 * Below this line is plumbing: the synthetic export, the chunk driver, the sub-commands the
 * notebook calls, and the self-test that `make test` runs. Nothing here is an exercise.
 * ====================================================================================== */

/* ---- The synthetic export ------------------------------------------------------------
 * Generated deterministically from a fixed seed, so every student scans the same bytes and
 * therefore reads the same totals. It is a block of kBlockRecords records, built once, and
 * then repeated for as long as the caller asks. A real export is not periodic; the scanner
 * cannot tell, and neither can its memory profile, which is the only property under test.
 * The block deliberately contains the three things a naive split-on-comma gets wrong:
 * quoted fields holding commas, fields holding an embedded newline and an escaped "", and
 * notes longer than kFieldCap. */
enum { kBlockRecords = 997, kBlockSlot = 400 };
static char g_block[kBlockRecords * kBlockSlot];
static size_t g_block_len = 0;

static uint64_t xorshift(uint64_t* s) {
  uint64_t x = *s;
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  return *s = x;
}

static const char kLongNote[] =
    "goods inspected on arrival at bay 14 and cross-checked against the packing list, "
    "three cartons re-weighed, discrepancy report filed with the carrier the same afternoon";

static void build_block(void) {
  static const char* kSuppliers[8] = {"Nordvik Papir AS",
                                      "\"Vantor Logistics, S.A.\"",
                                      "Kessler & Roth GmbH",
                                      "\"Delta Marine Supply, Ltd\"",
                                      "Aquila Chemical NV",
                                      "\"Bergen Freight, Oy\"",
                                      "Tamm Industrial OU",
                                      "\"Larkfield Print, PLC\""};
  static const char* kCurrency[4] = {"EUR", "EUR", "USD", "GBP"};
  uint64_t rng = 0x5eed4a7451ULL;
  char* p = g_block;
  long i;
  if (g_block_len) return;
  for (i = 0; i < kBlockRecords; ++i) {
    uint64_t r = xorshift(&rng);
    long units = (long)(r % 90000) + 100;
    long cents = (long)((r >> 17) % 100);
    int note_kind = (int)((r >> 29) % 8);
    const char* supplier = kSuppliers[(r >> 33) % 8];
    const char* ccy = kCurrency[(r >> 41) % 4];
    size_t room = (size_t)(g_block + sizeof g_block - p);
    p += snprintf(p, room, "DOC-%07ld,%s,INV-2026-%05ld,%ld.%02ld,%s,", i, supplier,
                  (i * 7) % 100000, units, cents, ccy);
    room = (size_t)(g_block + sizeof g_block - p);
    if (note_kind == 0)
      p += snprintf(p, room, "\"received %ld units\nchecked by ops, marked \"\"urgent\"\"\"\n",
                    units % 40 + 1);
    else if (note_kind == 1)
      p += snprintf(p, room, "\"%s\"\n", kLongNote);
    else
      p += snprintf(p, room, "ok\n");
  }
  g_block_len = (size_t)(p - g_block);
}

/* Fill `want` bytes of the export starting at byte `offset`. Pure function of the offset:
 * the same byte is produced whatever chunking the caller chose. */
static void gen_fill(long long offset, char* out, size_t want) {
  size_t done = 0;
  build_block();
  while (done < want) {
    size_t pos = (size_t)((offset + (long long)done) % (long long)g_block_len);
    size_t take = g_block_len - pos;
    if (take > want - done) take = want - done;
    memcpy(out + done, g_block + pos, take);
    done += take;
  }
}

/* ---- The chunk driver ----------------------------------------------------------------
 * One buffer, one size, for any export. It is `static` rather than malloc'd so that there is
 * nothing to leak and nothing to grow: the scan's entire working set is this buffer plus one
 * Scanner. That is the claim section 11 measures. */
enum { kMaxChunk = 1 << 16 };
static char g_chunk[kMaxChunk];

static size_t clamp_chunk(long requested) {
  if (requested <= 0 || requested > kMaxChunk) return kMaxChunk;
  return (size_t)requested;
}

static long long scan_generated(Scanner* s, long long total, size_t chunk) {
  long long done = 0;
  while (done < total) {
    size_t want = (size_t)(total - done);
    if (want > chunk) want = chunk;
    gen_fill(done, g_chunk, want);
    scan_chunk(s, g_chunk, want);
    done += (long long)want;
  }
  scan_finish(s);
  return done;
}

static long long scan_file(Scanner* s, const char* path, size_t chunk) {
  FILE* f = fopen(path, "rb");
  long long done = 0;
  size_t n;
  if (!f) {
    fprintf(stderr, "error: cannot open export: %s\n", path);
    exit(1);
  }
  while ((n = fread(g_chunk, 1, chunk, f)) > 0) {
    scan_chunk(s, g_chunk, n);
    done += (long long)n;
  }
  fclose(f);
  scan_finish(s);
  return done;
}

/* ---- Reporting ----------------------------------------------------------------------- */

static long long peak_rss_raw(void) {
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  /* The UNIT of ru_maxrss differs between platforms, so this program does not name it. The
   * notebook measures the unit with the `rssunit` sub-command below and converts. */
  return (long long)ru.ru_maxrss;
}

static void print_stats(const Scanner* s) {
  printf("records %lld\n", s->records);
  printf("fields %lld\n", s->fields);
  printf("bytes %lld\n", s->bytes);
  printf("amount_cents %lld\n", s->amount_cents);
  printf("eur_records %lld\n", s->eur_records);
  printf("max_field_len %lld\n", s->max_field_len);
  printf("field_overflow %lld\n", s->field_overflow);
  printf("embedded_newlines %lld\n", s->embedded_newlines);
  printf("bad_arity %lld\n", s->bad_arity);
  printf("truncated_tail %d\n", s->truncated_tail);
}

static const char* option(int argc, char** argv, const char* name, const char* fallback) {
  int i;
  for (i = 1; i + 1 < argc; ++i)
    if (strcmp(name, argv[i]) == 0) return argv[i + 1];
  return fallback;
}

static long long option_ll(int argc, char** argv, const char* name, long long fallback) {
  const char* raw = option(argc, argv, name, NULL);
  return raw ? strtoll(raw, NULL, 10) : fallback;
}

/* ---- Sub-commands -------------------------------------------------------------------- */

static int cmd_push(int argc, char** argv) {
  const char* text = option(argc, argv, "--text", "");
  Scanner s;
  size_t i;
  scanner_init(&s);
  for (i = 0; text[i]; ++i) field_push(&s, text[i]);
  printf("field_len %zu\n", s.field_len);
  printf("seen_len %zu\n", s.seen_len);
  printf("field_overflow %lld\n", s.field_overflow);
  printf("field %.*s\n", (int)s.field_len, s.field);
  return 0;
}

/* "--record a|b|c|d|e|f" pushes each '|'-separated value and completes it as a field. The
 * separator is a pipe so that a value may contain a comma. With --close 1 the record is
 * closed too, which is how the arity counter is exercised without the state machine. */
static int cmd_field(int argc, char** argv) {
  const char* record = option(argc, argv, "--record", "");
  const long close = (long)option_ll(argc, argv, "--close", 0);
  Scanner s;
  size_t i = 0;
  scanner_init(&s);
  for (;;) {
    if (record[i] == '|' || record[i] == '\0') {
      field_complete(&s);
      if (record[i] == '\0') break;
    } else {
      field_push(&s, record[i]);
    }
    ++i;
  }
  if (close) record_complete(&s);
  print_stats(&s);
  printf("col %d\n", s.col);
  return 0;
}

static int cmd_scan(int argc, char** argv) {
  const char* text = option(argc, argv, "--text", "");
  const size_t chunk = clamp_chunk((long)option_ll(argc, argv, "--chunk", 0));
  Scanner s;
  size_t len = strlen(text), done = 0;
  scanner_init(&s);
  while (done < len) {
    size_t want = len - done;
    if (want > chunk) want = chunk;
    scan_chunk(&s, text + done, want);
    done += want;
  }
  scan_finish(&s);
  print_stats(&s);
  return 0;
}

static int cmd_scanfile(int argc, char** argv) {
  const char* path = option(argc, argv, "--file", "");
  const size_t chunk = clamp_chunk((long)option_ll(argc, argv, "--chunk", 0));
  Scanner s;
  struct timespec t0, t1;
  double seconds;
  scanner_init(&s);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  scan_file(&s, path, chunk);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  seconds = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
  print_stats(&s);
  printf("scan_seconds %.6f\n", seconds);
  printf("chunk %zu\n", chunk);
  printf("peak_rss_raw %lld\n", peak_rss_raw());
  return 0;
}

static int cmd_bench(int argc, char** argv) {
  const long long total = option_ll(argc, argv, "--bytes", 1 << 20);
  const size_t chunk = clamp_chunk((long)option_ll(argc, argv, "--chunk", 0));
  Scanner s;
  struct timespec t0, t1;
  double seconds;
  scanner_init(&s);
  build_block(); /* outside the timed region: the export is the input, not the work */
  clock_gettime(CLOCK_MONOTONIC, &t0);
  scan_generated(&s, total, chunk);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  seconds = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
  print_stats(&s);
  printf("scan_seconds %.6f\n", seconds);
  printf("chunk %zu\n", chunk);
  printf("block_bytes %zu\n", g_block_len);
  printf("peak_rss_raw %lld\n", peak_rss_raw());
  return 0;
}

static int cmd_gen(int argc, char** argv) {
  const long long total = option_ll(argc, argv, "--bytes", 1 << 20);
  const char* out = option(argc, argv, "--out", "");
  FILE* f = fopen(out, "wb");
  long long done = 0;
  if (!f) {
    fprintf(stderr, "error: cannot write export: %s\n", out);
    return 1;
  }
  build_block();
  while (done < total) {
    size_t want = (size_t)(total - done);
    if (want > kMaxChunk) want = kMaxChunk;
    gen_fill(done, g_chunk, want);
    fwrite(g_chunk, 1, want, f);
    done += (long long)want;
  }
  fclose(f);
  printf("bytes %lld\n", done);
  printf("block_bytes %zu\n", g_block_len);
  return 0;
}

/* The unit of ru_maxrss is not the same on every platform, and this lesson will not assert
 * one. Touch a known number of bytes in a SEPARATE process and report what the counter did;
 * the notebook divides. Keeping this out of the scanning processes is the point — a 64 MiB
 * calibration inside a bench run would be the peak it was trying to measure. */
static int cmd_rssunit(void) {
  const long long probe = 64LL << 20;
  long long before = peak_rss_raw();
  volatile char* block = (volatile char*)malloc((size_t)probe);
  long long after, i, checksum = 0;
  if (!block) {
    fprintf(stderr, "error: calibration allocation failed\n");
    return 1;
  }
  /* Touch every page, and READ it back into a value that gets printed. Both halves matter:
   * an untouched page is never charged to this process, and a buffer nothing ever reads is a
   * buffer -O2 is entitled to delete along with the malloc that produced it. */
  for (i = 0; i < probe; i += 4096) block[i] = (char)((i / 4096) & 0x7f);
  for (i = 0; i < probe; i += 4096) checksum += block[i];
  after = peak_rss_raw();
  printf("probe_bytes %lld\n", probe);
  printf("rss_before_raw %lld\n", before);
  printf("rss_after_raw %lld\n", after);
  printf("probe_checksum %lld\n", checksum);
  free((void*)block);
  return 0;
}

/* ---- Self-test ------------------------------------------------------------------------
 * Small cases a human can check on paper. The notebook's checks are far stricter; these
 * exist so `make test` alone tells a student whether the four pieces hold together. */
static int g_failures = 0;

static void require(int ok, const char* what) {
  if (!ok) {
    fprintf(stderr, "selftest failed: %s\n", what);
    ++g_failures;
  }
}

static void scan_all(Scanner* s, const char* text) {
  scanner_init(s);
  scan_chunk(s, text, strlen(text));
  scan_finish(s);
}

static int selftest(void) {
  Scanner s;
  long i;

  scanner_init(&s);
  for (i = 0; i < 130; ++i) field_push(&s, 'x');
  require(s.field_len == kFieldCap, "field_push must stop writing at kFieldCap");
  require(s.seen_len == 130, "field_push must count every byte in seen_len, kept or not");
  require(s.field_overflow == 1, "one over-long field is ONE overflow, not one per extra byte");
  printf("  field_push ok\n");

  scanner_init(&s);
  s.col = kColAmount;
  field_push(&s, '1');
  field_push(&s, '2');
  field_push(&s, '.');
  field_push(&s, '5');
  field_push(&s, '0');
  field_complete(&s);
  require(s.amount_cents == 1250, "an amount field must be folded into amount_cents");
  require(s.col == kColCurrency, "field_complete must advance col by one");
  require(s.field_len == 0 && s.seen_len == 0, "field_complete must reset the field");
  require(s.fields == 1, "field_complete must count the field");
  printf("  field_complete ok\n");

  scan_all(&s, "a,b,c,d,EUR,f\n");
  require(s.records == 1 && s.fields == 6, "one six-field record is one record and six fields");
  require(s.eur_records == 1, "column 4 equal to EUR must be counted");
  require(s.bad_arity == 0, "a six-field record has the right arity");
  scan_all(&s, "a,\"b\"\"c\",d,1.00,EUR,\"two\nlines\"\n");
  require(s.records == 1 && s.fields == 6, "a quoted field holding a comma or a newline is one field");
  require(s.embedded_newlines == 1, "a newline inside quotes is data, and is counted");
  require(s.amount_cents == 100, "1.00 is 100 cents");
  printf("  scan_chunk ok\n");

  scan_all(&s, "a,b,c,1.00,EUR,f");
  require(s.records == 1 && s.truncated_tail == 0,
          "a final record with all six fields is a record even without a newline");
  scan_all(&s, "a,b,c,1.00,EUR");
  require(s.records == 0 && s.truncated_tail == 1 && s.fields == 4,
          "a short final record is truncation: no record, and the PENDING field is dropped");
  require(s.amount_cents == 100,
          "the amount was terminated by its comma long before the cut, so it still counts - "
          "section 12 of the notebook measures what that convention costs");
  scan_all(&s, "a,b,c,1.00,EUR,\"unterminated");
  require(s.records == 0 && s.truncated_tail == 1,
          "a stream that stops inside quotes is truncated");
  scan_all(&s, "a,b,c,1.00,EUR,f\n");
  require(s.truncated_tail == 0 && s.records == 1,
          "a trailing newline ends the stream cleanly; do not invent a seventh field");
  printf("  scan_finish ok\n");

  if (g_failures) return 1;
  printf("selftest ok\n");
  return 0;
}

int main(int argc, char** argv) {
  const char* cmd = argc > 1 ? argv[1] : "selftest";
  if (strcmp(cmd, "selftest") == 0) return selftest();
  if (strcmp(cmd, "push") == 0) return cmd_push(argc, argv);
  if (strcmp(cmd, "field") == 0) return cmd_field(argc, argv);
  if (strcmp(cmd, "scan") == 0) return cmd_scan(argc, argv);
  if (strcmp(cmd, "scanfile") == 0) return cmd_scanfile(argc, argv);
  if (strcmp(cmd, "bench") == 0) return cmd_bench(argc, argv);
  if (strcmp(cmd, "gen") == 0) return cmd_gen(argc, argv);
  if (strcmp(cmd, "rssunit") == 0) return cmd_rssunit();
  fprintf(stderr, "unknown command %s (selftest|push|field|scan|scanfile|bench|gen|rssunit)\n",
          cmd);
  return 2;
}
