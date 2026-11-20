// SPDX-License-Identifier: CC-BY-NC-SA-4.0
// Synapsa Commons - Copyright 2026 RealAI - free to learn from and share, not to sell; see NOTICE.
// T05-L04 - 4-bit number formats in C++: INT4, MXFP4 and NVFP4.
//
// Fourteen functions are yours to write, in seven exercises. Fill them in, then run:
//
//     make test
//
// The self-test reports each exercise on its own line, so you can finish them one at a time and
// watch the TODOs turn into PASSes. Everything below the exercises (the commands the notebook
// drives, the file reader, the byte counting) is the harness and is given.
//
// Moved or copied this checkout? Run `make clean` first. See the common-mistakes section of the
// notebook for why.
//
// Rules of this file: C++17 and the standard library only; no threads, no inline assembly, no
// compiler intrinsics for float conversion. Every float is a 32-bit `float`, and every operation
// below is a single-precision operation in the order the comments give: the notebook compares
// your output with a Python reference BIT FOR BIT, so a different order of operations is a
// different answer. The values your functions are given are finite, with magnitudes between 1e-20
// and 1e20 (or exactly zero), unless a comment says otherwise.

#include <algorithm>
#include <cmath>
#include <csetjmp>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------------------
// The contract. Every constant here is mirrored in the notebook, which checks that the two
// agree before it trusts a single number from this binary.
// ---------------------------------------------------------------------------------------
#define INT4_BLOCK 32  // values that share one float scale in the INT4 format
#define MX_BLOCK 32    // values that share one E8M0 scale in MXFP4
#define NV_BLOCK 16    // values that share one E4M3 scale in NVFP4
#define E2M1_MAX 6.0f  // the largest magnitude an E2M1 element can hold
#define E4M3_MAX 448.0f

#define LESSON_MAYBE_UNUSED __attribute__((unused))

// =======================================================================================
// Plumbing for the self-test. Read it once, then ignore it.
// =======================================================================================
//
// C++ code without exceptions here: an unfinished exercise reports itself with sigsetjmp/siglongjmp.
// todo() records a message and jumps back to the check that called it. Outside the self-test it
// prints the message and exits 2, which the notebook's Python wrapper turns into NotImplementedError,
// so the grader prints TODO rather than a stack trace. A real failure jumps with a different code
// and becomes exit 1.
static sigjmp_buf g_jmp;
static int g_jmp_active = 0;
static char g_msg[1024];

static LESSON_MAYBE_UNUSED void todo(const char* what, const char* hint) {
  snprintf(g_msg, sizeof g_msg, "%s() is still a stub - %s", what, hint);
  if (g_jmp_active) siglongjmp(g_jmp, 1);
  fprintf(stderr, "NOT IMPLEMENTED: %s\n", g_msg);
  exit(2);
}

static void require(int ok, const char* fmt, ...) {
  if (ok) return;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_msg, sizeof g_msg, fmt, ap);
  va_end(ap);
  if (g_jmp_active) siglongjmp(g_jmp, 2);
  fprintf(stderr, "error: %s\n", g_msg);
  exit(1);
}

// The bits of a float, and a float from bits. memcpy, never a pointer cast.
static LESSON_MAYBE_UNUSED uint32_t bits_of(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}
static LESSON_MAYBE_UNUSED float float_of(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// =======================================================================================
// EXERCISE 1 - nibbles: pack_nibbles, unpack_nibbles, int4_to_nibble, nibble_to_int4
// =======================================================================================
//
// A 4-bit code is a "nibble". Memory is addressed in bytes, so two nibbles share a byte.
//
//   pack_nibbles(nib, n, bytes): nib holds n entries, each a nibble in the LOW 4 BITS of an
//     uint8_t; the other four bits of an entry are junk and must be ignored (mask them off).
//     Write (n + 1) / 2 bytes: entry 2i goes in the LOW nibble of byte i and entry 2i + 1 in
//     the HIGH nibble. When n is odd the last byte's high nibble is zero. Write nothing past
//     byte (n + 1) / 2 - 1.
//   unpack_nibbles(bytes, n, nib): the inverse. Write exactly n entries, each in 0..15, entry
//     2i from the low nibble of byte i and entry 2i + 1 from its high nibble. Write nothing past
//     entry n - 1 (for odd n the last byte's high nibble is not an entry).
//   int4_to_nibble(q): q is a signed integer in [-8, 7]; return its 4-bit two's-complement
//     pattern as 0..15 (-1 is 15, -8 is 8, 7 is 7).
//   nibble_to_int4(v): the inverse, for a value whose low 4 bits are the pattern (ignore any
//     other bit of v): return the signed integer in [-8, 7] (15 is -1, 8 is -8).
//
// Worked example: nib = {1, 2, 3, 4, 5}, n = 5 packs to bytes {0x21, 0x43, 0x05}.
static void pack_nibbles(const uint8_t* nib, int n, uint8_t* bytes) {
  // YOUR CODE HERE
  todo("pack_nibbles", "the comment above it has the layout and a worked example");
}

static void unpack_nibbles(const uint8_t* bytes, int n, uint8_t* nib) {
  // YOUR CODE HERE
  todo("unpack_nibbles", "the comment above pack_nibbles has the layout");
}

static int int4_to_nibble(int q) {
  // YOUR CODE HERE
  todo("int4_to_nibble", "the comment above pack_nibbles has the rule");
  return 0;  // unreachable; todo() never returns
}

static int nibble_to_int4(int v) {
  // YOUR CODE HERE
  todo("nibble_to_int4", "the comment above pack_nibbles has the rule");
  return 0;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 2 - INT4 with one float scale per block: int4_quantise_block, int4_dequantise_block
// =======================================================================================
//
// n values (n <= INT4_BLOCK) share one float scale. In single precision, in this order:
//
//     amax  = the largest |x[i]|
//     scale = amax / 7.0f                      (a float division)
//     q[i]  = nearbyintf(x[i] / scale)         (a float division; nearbyintf rounds halves to EVEN
//                                               in the default rounding mode, so 2.5 -> 2 and 3.5 -> 4)
//
// Because |x[i]| <= amax, q[i] lies in [-7, 7] without any clamping, and -8, which two's complement could
// hold, is never used: the grid is symmetric.
//
// An all-zero block (amax == 0) has scale 1.0f and every q[i] = 0. int4_quantise_block returns the
// scale and writes n values to q (each an int8_t in [-7, 7]); it writes nothing past q[n - 1].
// int4_dequantise_block writes out[i] = (float)q[i] * scale for i < n and nothing else.
//
// Worked example: x = {7, -7, 2.5, 0.5}: amax 7, scale 1, q = {7, -7, 2, 0}.
static float int4_quantise_block(const float* x, int n, int8_t* q) {
  // YOUR CODE HERE
  todo("int4_quantise_block", "the comment above it has the rule and a worked example");
  return 0.0f;  // unreachable; todo() never returns
}

static void int4_dequantise_block(float scale, const int8_t* q, int n, float* out) {
  // YOUR CODE HERE
  todo("int4_dequantise_block", "the comment above int4_quantise_block has the rule");
}

// =======================================================================================
// EXERCISE 3 - the 4-bit float E2M1: e2m1_decode, e2m1_encode
// =======================================================================================
//
// E2M1 has 1 sign bit, 2 exponent bits and 1 mantissa bit: the code is s e e m (bit 3 is the sign).
// Its exponent bias is 1. Reading the fields s, E (0..3) and M (0..1):
//     E == 0 (subnormal):  value = (-1)^s * 0.5 * M          -> 0 and 0.5
//     E  > 0 (normal):     value = (-1)^s * 2^(E-1) * (1 + M/2)
// so the eight magnitudes, in code order 0..7, are 0, 0.5, 1, 1.5, 2, 3, 4, 6. There is no infinity
// and no NaN. Code 8 is NEGATIVE zero: e2m1_decode(8) must return -0.0f (its sign bit set).
//
// e2m1_decode(code): use the low 4 bits of `code` and return the value.
//
// e2m1_encode(x): return the code (0..15) of the representable value nearest to x:
//   * the sign bit is the sign of x (signbit), even when the magnitude rounds to zero: -0.1f
//     encodes as 8, not 0, and -0.0f encodes as 8;
//   * halfway between two representable magnitudes goes to the one whose CODE is even (round half
//     to even: 0.25 -> code 0, 0.75 -> code 2, 1.25 -> code 2, 1.75 -> code 4, 2.5 -> code 4,
//     3.5 -> code 6, 5.0 -> code 6);
//   * anything at or above 6 saturates to the largest magnitude, code 7 (or 15): infinity too.
//     Never wrap round to a small code.
// x is not NaN.
static float e2m1_decode(int code) {
  // YOUR CODE HERE
  todo("e2m1_decode", "the comment above it has the fields and the eight values");
  return 0.0f;  // unreachable; todo() never returns
}

static int e2m1_encode(float x) {
  // YOUR CODE HERE
  todo("e2m1_encode", "the comment above e2m1_decode has the rounding rule and the examples");
  return 0;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 4 - the E8M0 scale and the shared exponent: e8m0_decode, mx_shared_exponent
// =======================================================================================
//
// E8M0 is eight exponent bits and nothing else: a byte b means the power of two 2^(b - 127),
// for b = 0..254. The byte 255 is the single NaN. (b = 0 is 2^-127, a float that is subnormal
// but exactly representable; ldexpf(1.0f, b - 127) makes it.)
//
// e8m0_decode(byte): use the low 8 bits of `byte`; return 2^(byte - 127) as a float, or NaN for 255.
//
// mx_shared_exponent(x, n, emax_elem): the exponent X of the shared scale 2^X of a block of n values (emax_elem is
// 0 or more: 2 for E2M1),
// by the OCP rule: "the largest power of two less than or equal to max |x|, divided by the largest
// power of two representable in the element type". For E2M1 that is 2^2 = 4, so emax_elem is 2:
//
//     X = floor(log2(amax)) - emax_elem          where amax = the largest |x[i]|
//
// clamped to [-127, 127] (the range E8M0 can hold); an all-zero block (amax == 0) returns -127.
// The floor of the logarithm must be EXACT. A float32 log2f is rounded, so floorf(log2f(a)) is wrong
// for a value just below a power of two: for 7.9999995f, log2f returns exactly 3 and the floor is 3,
// but the true floor is 2. Read the exponent field instead (ilogbf or frexpf), which is exact.
//
// This function must accept ANY finite float (denormals and the largest floats included), not only the
// 1e-20 to 1e20 range of the rest of this file.
//
// Worked example: x = {6, -3}, emax_elem 2: amax 6 = 1.5 * 2^2, floor(log2) = 2, X = 0, scale 1.
static float e8m0_decode(int byte) {
  // YOUR CODE HERE
  todo("e8m0_decode", "the comment above it has the rule");
  return 0.0f;  // unreachable; todo() never returns
}

static int mx_shared_exponent(const float* x, int n, int emax_elem) {
  // YOUR CODE HERE
  todo("mx_shared_exponent", "the comment above e8m0_decode has the rule, the clamp and the trap");
  return 0;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 5 - an MXFP4 block: mxfp4_quantise_block, mxfp4_dequantise_block
// =======================================================================================
//
// An MXFP4 block is up to MX_BLOCK = 32 values stored as ONE scale byte (E8M0) and one E2M1 code per
// value (the caller packs the codes two to a byte). To encode n values:
//     X = mx_shared_exponent(x, n, 2)
//     *scale_byte = X + 127                       (the E8M0 byte of 2^X)
//     scale = e8m0_decode(*scale_byte)
//     codes[i] = e2m1_encode(x[i] / scale)        (a float division; e2m1_encode saturates, so a value
//                                                  that lands above 6 is clamped to 6)
// Each codes[i] is 0..15. Write codes[0 .. n - 1] and nothing else.
// To decode: out[i] = e2m1_decode(codes[i]) * e8m0_decode(scale_byte)  (a float multiplication).
// If scale_byte is 255 (NaN) every out[i] is NaN, whatever the codes.
//
// Like mx_shared_exponent, both functions must accept ANY finite float, not only 1e-20 to 1e20.
//
// Worked example: x = {6, -3, 0.25}: X = 0, scale byte 127, codes {7, 13, 0}; decoded {6, -3, 0}.
static void mxfp4_quantise_block(const float* x, int n, uint8_t* scale_byte, uint8_t* codes) {
  // YOUR CODE HERE
  todo("mxfp4_quantise_block", "the comment above it has the steps and a worked example");
}

static void mxfp4_dequantise_block(uint8_t scale_byte, const uint8_t* codes, int n, float* out) {
  // YOUR CODE HERE
  todo("mxfp4_dequantise_block", "the comment above mxfp4_quantise_block has the rule");
}

// =======================================================================================
// EXERCISE 6 - the 8-bit float E4M3: e4m3_decode, e4m3_encode
// =======================================================================================
//
// E4M3 has 1 sign bit, 4 exponent bits (bias 7) and 3 mantissa bits: the byte is s eeee mmm. Fields S, E, M:
//     E == 0 (subnormal): value = (-1)^S * 2^-6 * (M / 8)
//     E  > 0 (normal):    value = (-1)^S * 2^(E-7) * (1 + M / 8)
// There is NO infinity. The one NaN pattern (in either sign) is E = 15 and M = 7 (bytes 0x7F and 0xFF),
// so the largest magnitude is 448 = 2^8 * 1.75 (0x7E). Byte 0x80 is negative zero.
//
// e4m3_decode(byte): use the low 8 bits of `byte`; return the value (NaN for 0x7F and 0xFF).
//
// e4m3_encode(x): return the byte of the representable value nearest to x:
//   * NaN encodes as 0x7F; infinity, and any magnitude at or above 448, SATURATES to 448 (0x7E or 0xFE);
//   * the sign bit is the sign of x even when the magnitude rounds to zero (-0.0f encodes as 0x80);
//   * halfway between two representable magnitudes goes to the one whose BYTE is even (so 1.0625, halfway
//     between 1.0 and 1.125, encodes as 0x38 and 1.1875 as 0x3A; 2^-10, halfway between 0 and the smallest
//     subnormal 2^-9, encodes as 0x00).
//
// Worked example: 1.0 is 0x38 (E = 7, M = 0); 0.0625 is 0x18; 448 is 0x7E; -1000 saturates to 0xFE.
static float e4m3_decode(int byte) {
  // YOUR CODE HERE
  todo("e4m3_decode", "the comment above it has the fields and the special bytes");
  return 0.0f;  // unreachable; todo() never returns
}

static int e4m3_encode(float x) {
  // YOUR CODE HERE
  todo("e4m3_encode", "the comment above e4m3_decode has the rounding rule and the examples");
  return 0;  // unreachable; todo() never returns
}

// =======================================================================================
// EXERCISE 7 - an NVFP4 tensor: nvfp4_quantise, nvfp4_dequantise
// =======================================================================================
//
// NVFP4 has TWO levels of scale: an E4M3 byte per block of NV_BLOCK = 16 values, and one float for the
// whole tensor. The elements are E2M1 codes. In single precision, in this order (n values, any n >= 1;
// block b holds x[16b .. min(16b + 16, n) - 1], so the last block may be shorter):
//
//   1. amax_x = the largest |x[i]| over the whole tensor.
//      s_enc  = 2688.0f / amax_x                 (2688 = 6 * 448)       s_dec = 1.0f / s_enc
//      If amax_x == 0, take s_enc = 1.0f and s_dec = 1.0f.
//   2. For each block b with amax_b = its largest |x|:
//        s_dec_b       = amax_b / 6.0f
//        scale_bytes[b] = e4m3_encode(s_dec_b * s_enc)
//        sb            = e4m3_decode(scale_bytes[b])
//        if sb == 0.0f: every code of the block is 0 (and go to the next block);
//        otherwise  s_enc_b = 1.0f / (sb * s_dec)       and   codes[i] = e2m1_encode(x[i] * s_enc_b).
//   3. Return s_dec (the per-tensor decode scale, stored as a float).
//
// nvfp4_quantise writes scale_bytes[0 .. blocks - 1] (blocks = (n + 15) / 16) and codes[0 .. n - 1], and
// nothing else. nvfp4_dequantise(tensor_scale, scale_bytes, codes, n, out) writes, for every i,
//     out[i] = (e2m1_decode(codes[i]) * e4m3_decode(scale_bytes[i / 16])) * tensor_scale
// (two float multiplications, in that order). A scale byte that decodes to NaN gives NaN outputs.
// This is NVIDIA's published recipe; the notebook states the equations.
//
// Worked example: x = {6, -3, 1.5} followed by thirteen zeros: amax_x = 6, s_enc = 448; block 0 has
// amax_b = 6, s_dec_b = 1, scale byte e4m3_encode(448) = 0x7E, and the codes are {7, 13, 3, 0, ...}.
static float nvfp4_quantise(const float* x, int n, uint8_t* scale_bytes, uint8_t* codes) {
  // YOUR CODE HERE
  todo("nvfp4_quantise", "the comment above it has the three steps and a worked example");
  return 0.0f;  // unreachable; todo() never returns
}

static void nvfp4_dequantise(float tensor_scale, const uint8_t* scale_bytes, const uint8_t* codes, int n,
                             float* out) {
  // YOUR CODE HERE
  todo("nvfp4_dequantise", "the comment above nvfp4_quantise has the rule");
}

// =======================================================================================
// The harness below is given. The notebook drives it through the commands listed in usage().
// Numbers cross the pipe as the 8 hex digits of a float's bits, so nothing is rounded in text.
// =======================================================================================

static const int GUARD = 0xA5;

static int check_nibbles_ok(const uint8_t* got, const uint8_t* want, int n) {
  for (int i = 0; i < n; i++)
    if (got[i] != want[i]) return 0;
  return 1;
}

static void check_nibbles(void) {
  const uint8_t nib[5] = {1, 2, 3, 4, 5};
  uint8_t bytes[4];
  memset(bytes, GUARD, sizeof bytes);
  pack_nibbles(nib, 5, bytes);
  const uint8_t want[4] = {0x21, 0x43, 0x05, (uint8_t)GUARD};
  require(check_nibbles_ok(bytes, want, 4),
          "packing {1,2,3,4,5} gave bytes %02x %02x %02x and a guard byte of %02x, expected 21 43 05 and a guard of a5: "
          "entry 2i is the LOW nibble, an odd count leaves the high nibble zero, and nothing is written past the last byte",
          bytes[0], bytes[1], bytes[2], bytes[3]);
  const uint8_t junk[2] = {0xF1, 0x32};
  uint8_t one[2] = {0, (uint8_t)GUARD};
  pack_nibbles(junk, 2, one);
  require(one[0] == 0x21 && one[1] == GUARD,
          "packing {0xF1, 0x32} gave %02x, expected 21: only the low 4 bits of each entry count, so mask before you shift",
          one[0]);
  uint8_t back[6];
  memset(back, 0x55, sizeof back);
  const uint8_t packed[3] = {0x21, 0x43, 0x05};
  unpack_nibbles(packed, 5, back);
  const uint8_t want_back[6] = {1, 2, 3, 4, 5, 0x55};
  require(check_nibbles_ok(back, want_back, 6),
          "unpacking 21 43 05 with n = 5 gave %d %d %d %d %d (guard %02x), expected 1 2 3 4 5 and an untouched guard: "
          "write exactly n entries, each 0..15",
          back[0], back[1], back[2], back[3], back[4], back[5]);
  require(int4_to_nibble(-1) == 15 && int4_to_nibble(-8) == 8 && int4_to_nibble(7) == 7 && int4_to_nibble(0) == 0,
          "int4_to_nibble(-1, -8, 7, 0) gave %d %d %d %d, expected 15 8 7 0: the two's-complement pattern in 0..15",
          int4_to_nibble(-1), int4_to_nibble(-8), int4_to_nibble(7), int4_to_nibble(0));
  require(nibble_to_int4(15) == -1 && nibble_to_int4(8) == -8 && nibble_to_int4(7) == 7 && nibble_to_int4(0xF8) == -8,
          "nibble_to_int4(15, 8, 7, 0xF8) gave %d %d %d %d, expected -1 -8 7 -8: sign-extend bit 3, and ignore the other bits",
          nibble_to_int4(15), nibble_to_int4(8), nibble_to_int4(7), nibble_to_int4(0xF8));
}

static void check_int4(void) {
  const float x[8] = {7.0f, -7.0f, 2.5f, -2.5f, 0.5f, -0.5f, 1.5f, -1.5f};
  int8_t q[9];
  memset(q, 99, sizeof q);
  float scale = int4_quantise_block(x, 8, q);
  const int8_t want[9] = {7, -7, 2, -2, 0, 0, 2, -2, 99};
  require(scale == 1.0f && memcmp(q, want, 9) == 0,
          "the block {7,-7,2.5,-2.5,0.5,-0.5,1.5,-1.5} gave scale %g and codes %d %d %d %d %d %d %d %d (guard %d), expected "
          "scale 1 and 7 -7 2 -2 0 0 2 -2 (guard 99): nearbyintf rounds halves to even, and nothing is written past n",
          scale, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8]);
  const float zeros[3] = {0.0f, -0.0f, 0.0f};
  int8_t z[4] = {9, 9, 9, 99};
  scale = int4_quantise_block(zeros, 3, z);
  require(scale == 1.0f && z[0] == 0 && z[1] == 0 && z[2] == 0 && z[3] == 99,
          "an all-zero block gave scale %g (codes %d %d %d), expected scale 1 and codes 0", scale, z[0], z[1], z[2]);
  const int8_t codes[3] = {7, -7, 1};
  float out[4] = {-1.0f, -1.0f, -1.0f, 12345.0f};
  int4_dequantise_block(0.5f, codes, 3, out);
  require(out[0] == 3.5f && out[1] == -3.5f && out[2] == 0.5f && out[3] == 12345.0f,
          "dequantising {7,-7,1} with scale 0.5 gave %g %g %g (guard %g), expected 3.5 -3.5 0.5 and an untouched guard",
          out[0], out[1], out[2], out[3]);
}

static void check_e2m1(void) {
  const float val[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int c = 0; c < 8; c++) {
    require(e2m1_decode(c) == val[c] && e2m1_decode(c + 8) == -val[c],
            "e2m1_decode(%d) gave %g and e2m1_decode(%d) gave %g, expected %g and %g", c, e2m1_decode(c), c + 8,
            e2m1_decode(c + 8), val[c], -val[c]);
  }
  require(std::signbit(e2m1_decode(8)) && e2m1_decode(8) == 0.0f,
          "e2m1_decode(8) must be negative zero (sign bit set): the sign bit is bit 3, even when the magnitude is zero");
  const float in[12] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 7.0f, -0.1f, -6.5f, 0.2f, 100.0f};
  const int want[12] = {0, 2, 2, 4, 4, 6, 6, 7, 8, 15, 0, 7};
  for (int i = 0; i < 12; i++)
    require(e2m1_encode(in[i]) == want[i], "e2m1_encode(%g) gave %d, expected %d (round half to even code; saturate at 6; keep the sign)",
            in[i], e2m1_encode(in[i]), want[i]);
  require(e2m1_encode(-0.0f) == 8 && e2m1_encode(INFINITY) == 7 && e2m1_encode(-INFINITY) == 15,
          "e2m1_encode(-0.0f, +inf, -inf) gave %d %d %d, expected 8 7 15", e2m1_encode(-0.0f), e2m1_encode(INFINITY),
          e2m1_encode(-INFINITY));
}

static void check_shared_exponent(void) {
  require(e8m0_decode(127) == 1.0f && e8m0_decode(130) == 8.0f && e8m0_decode(120) == 0.0078125f,
          "e8m0_decode(127, 130, 120) gave %g %g %g, expected 1 8 0.0078125 (2^(byte - 127))", e8m0_decode(127),
          e8m0_decode(130), e8m0_decode(120));
  float tiny = e8m0_decode(0);
  require(tiny > 0.0f && tiny < 1.17549435e-38f && tiny * 8388608.0f == 5.877471754111438e-39f * 8388608.0f && std::isnan(e8m0_decode(255)),
          "e8m0_decode(0) must be 2^-127 (a subnormal float) and e8m0_decode(255) NaN; got %g and %g", tiny, e8m0_decode(255));
  const float a[2] = {6.0f, -3.0f};
  require(mx_shared_exponent(a, 2, 2) == 0, "mx_shared_exponent({6,-3}, 2, 2) gave %d, expected 0", mx_shared_exponent(a, 2, 2));
  const float just_below[1] = {7.9999995f};
  require(mx_shared_exponent(just_below, 1, 2) == 0,
          "mx_shared_exponent({7.9999995f}) gave %d, expected 0: floorf(log2f(7.9999995f)) is 3, which is wrong, because log2f rounds "
          "to exactly 3; read the exponent field with ilogbf instead",
          mx_shared_exponent(just_below, 1, 2));
  const float eight[1] = {8.0f};
  require(mx_shared_exponent(eight, 1, 2) == 1, "mx_shared_exponent({8}) gave %d, expected 1", mx_shared_exponent(eight, 1, 2));
  const float zeros[3] = {0.0f, -0.0f, 0.0f};
  require(mx_shared_exponent(zeros, 3, 2) == -127, "an all-zero block must give -127, got %d", mx_shared_exponent(zeros, 3, 2));
  const float denorm[1] = {1e-45f};
  const float huge[1] = {3e38f};
  require(mx_shared_exponent(denorm, 1, 2) == -127 && mx_shared_exponent(huge, 1, 2) == 125,
          "mx_shared_exponent of 1e-45 and 3e38 gave %d and %d, expected -127 (clamped) and 125", mx_shared_exponent(denorm, 1, 2),
          mx_shared_exponent(huge, 1, 2));
}

static void check_mxfp4(void) {
  const float x[3] = {6.0f, -3.0f, 0.25f};
  uint8_t sb = 0, codes[4] = {0, 0, 0, (uint8_t)GUARD};
  mxfp4_quantise_block(x, 3, &sb, codes);
  require(sb == 127 && codes[0] == 7 && codes[1] == 13 && codes[2] == 0 && codes[3] == GUARD,
          "the block {6,-3,0.25} gave scale byte %d and codes %d %d %d (guard %02x), expected 127 and 7 13 0 (guard a5)", sb,
          codes[0], codes[1], codes[2], codes[3]);
  const float clip[2] = {7.5f, 0.5f};
  uint8_t c2[2];
  mxfp4_quantise_block(clip, 2, &sb, c2);
  require(sb == 127 && c2[0] == 7 && c2[1] == 1,
          "the block {7.5, 0.5} gave scale byte %d and codes %d %d, expected 127 and 7 1: 7.5 is above 6, so it saturates to code 7",
          sb, c2[0], c2[1]);
  float out[4] = {0, 0, 0, 12345.0f};
  const uint8_t cc[3] = {7, 13, 0};
  mxfp4_dequantise_block(127, cc, 3, out);
  require(out[0] == 6.0f && out[1] == -3.0f && out[2] == 0.0f && out[3] == 12345.0f,
          "decoding codes {7,13,0} with scale byte 127 gave %g %g %g (guard %g), expected 6 -3 0 (guard 12345)", out[0], out[1],
          out[2], out[3]);
  mxfp4_dequantise_block(129, cc, 3, out);
  require(out[0] == 24.0f && out[1] == -12.0f, "scale byte 129 is 2^2: decoded %g %g, expected 24 -12", out[0], out[1]);
  mxfp4_dequantise_block(255, cc, 3, out);
  require(std::isnan(out[0]) && std::isnan(out[1]) && std::isnan(out[2]), "scale byte 255 is NaN: every output must be NaN");
}

static void check_e4m3(void) {
  require(e4m3_decode(0x38) == 1.0f && e4m3_decode(0x7E) == 448.0f && e4m3_decode(0xFE) == -448.0f && e4m3_decode(0x08) == 0.015625f &&
              e4m3_decode(0x01) == 0.001953125f && e4m3_decode(0x18) == 0.0625f,
          "e4m3_decode(0x38, 0x7E, 0xFE, 0x08, 0x01, 0x18) gave %g %g %g %g %g %g, expected 1 448 -448 0.015625 0.001953125 0.0625",
          e4m3_decode(0x38), e4m3_decode(0x7E), e4m3_decode(0xFE), e4m3_decode(0x08), e4m3_decode(0x01), e4m3_decode(0x18));
  require(std::isnan(e4m3_decode(0x7F)) && std::isnan(e4m3_decode(0xFF)) && std::signbit(e4m3_decode(0x80)) && e4m3_decode(0x80) == 0.0f,
          "0x7F and 0xFF must decode to NaN, and 0x80 to negative zero");
  const float in[14] = {1.0f, 448.0f, 1000.0f, -1000.0f, 0.0f, -0.0f, 0.0009765625f, 0.0004f, 1.0625f, 1.1875f, 0.0625f, 1e-30f,
                        INFINITY, -INFINITY};
  const int want[14] = {0x38, 0x7E, 0x7E, 0xFE, 0x00, 0x80, 0x00, 0x00, 0x38, 0x3A, 0x18, 0x00, 0x7E, 0xFE};
  for (int i = 0; i < 14; i++)
    require(e4m3_encode(in[i]) == want[i], "e4m3_encode(%g) gave 0x%02X, expected 0x%02X (round half to an even byte; saturate at 448; keep the sign)",
            in[i], e4m3_encode(in[i]), want[i]);
  require(e4m3_encode(NAN) == 0x7F, "e4m3_encode(NaN) gave 0x%02X, expected 0x7F", e4m3_encode(NAN));
}

static void check_nvfp4(void) {
  float x[16] = {6.0f, -3.0f, 1.5f};
  uint8_t sbytes[3] = {0xEE, 0xEE, 0xEE}, codes[17];
  memset(codes, 0xEE, sizeof codes);
  float ts = nvfp4_quantise(x, 16, sbytes, codes);
  require(sbytes[0] == 0x7E && sbytes[1] == 0xEE && codes[0] == 7 && codes[1] == 13 && codes[2] == 3 && codes[3] == 0 && codes[15] == 0 &&
              codes[16] == 0xEE,
          "the worked example {6,-3,1.5,0,...} gave scale byte 0x%02X (a guard 0x%02X after it) and codes %d %d %d %d ... (guard 0x%02X), "
          "expected 0x7E and 7 13 3 0 ... with nothing written past block 0 and code 15",
          sbytes[0], sbytes[1], codes[0], codes[1], codes[2], codes[3], codes[16]);
  require(bits_of(ts) == 0x3B124925u, "the returned tensor scale has bits %08x, expected 3b124925 (1.0f / (2688.0f / 6.0f)): s_dec = 1 / s_enc", bits_of(ts));
  float out[17];
  for (int i = 0; i < 17; i++) out[i] = 12345.0f;
  nvfp4_dequantise(ts, sbytes, codes, 16, out);
  require(bits_of(out[0]) == 0x40C00001u && bits_of(out[1]) == 0xC0400001u && bits_of(out[2]) == 0x3FC00001u && out[3] == 0.0f &&
              out[16] == 12345.0f,
          "decoding the worked example gave %08x %08x %08x (guard %g), expected 40c00001 c0400001 3fc00001 (6, -3, 1.5 to the last bit) and an untouched guard: "
          "(code * block scale) * tensor scale, in that order",
          bits_of(out[0]), bits_of(out[1]), bits_of(out[2]), out[16]);
  float z[20];
  memset(z, 0, sizeof z);
  uint8_t zb[3] = {9, 9, 9}, zc[21];
  memset(zc, 7, sizeof zc);
  ts = nvfp4_quantise(z, 20, zb, zc);
  require(ts == 1.0f && zb[0] == 0 && zb[1] == 0 && zc[0] == 0 && zc[19] == 0 && zc[20] == 7,
          "an all-zero tensor gave tensor scale %g and scale bytes %d %d, expected 1 and 0 0 with every code 0", ts, zb[0], zb[1]);
}

// Each check runs in a CHILD process (fork), under an alarm. A function that never returns is stopped by the alarm
// (CHECK_SECONDS), and a function that reads or writes outside its arrays and crashes takes down only its own check:
// the self-test reports it as a failure and goes on to the next exercise, so one bad function cannot hide the
// feedback on the others. The alarm jumps back with code 3 and the crash is code 4; both are failures.
#define CHECK_SECONDS 1

static void on_alarm(int sig) {
  (void)sig;
  if (g_jmp_active) siglongjmp(g_jmp, 3);
}

static int run_check(const char* name, void (*fn)(void)) {
  int fd[2];
  if (pipe(fd) != 0) {
    printf("  FAIL  %s: the self-test could not create a pipe\n", name);
    return 1;
  }
  fflush(stdout);
  pid_t pid = fork();
  if (pid == 0) {
    close(fd[0]);
    g_jmp_active = 1;
    int rc = sigsetjmp(g_jmp, 1);
    if (rc == 0) {
      signal(SIGALRM, on_alarm);
      alarm(CHECK_SECONDS);
      fn();
    }
    alarm(0);
    g_jmp_active = 0;
    char buf[1200];
    int len = snprintf(buf, sizeof buf, "%d\n%s", rc, rc ? g_msg : "");
    ssize_t ignored = write(fd[1], buf, (size_t)len);
    (void)ignored;
    _exit(0);
  }
  close(fd[1]);
  char buf[1200];
  size_t have = 0;
  ssize_t got;
  while (have < sizeof buf - 1 && (got = read(fd[0], buf + have, sizeof buf - 1 - have)) > 0) have += (size_t)got;
  buf[have] = '\0';
  close(fd[0]);
  int status = 0;
  waitpid(pid, &status, 0);
  int rc = have > 0 ? atoi(buf) : 4;
  const char* msg = strchr(buf, '\n');
  msg = msg ? msg + 1 : "";
  if (rc == 0) {
    printf("  PASS  %s\n", name);
    return 0;
  }
  if (rc == 1) {
    printf("  TODO  %s: %s\n", name, msg);
    return 2;
  }
  if (rc == 3)
    printf("  FAIL  %s: was still running after %d s: a loop whose stopping condition is never met runs for ever; give every loop a cap\n",
           name, CHECK_SECONDS);
  else if (rc == 4)
    printf("  FAIL  %s: the check crashed (signal %d): a function read or wrote outside its arrays, or divided by zero; "
           "mind the length n, and write nothing past the last entry\n",
           name, WIFSIGNALED(status) ? WTERMSIG(status) : 0);
  else
    printf("  FAIL  %s: %s\n", name, msg);
  return 1;
}

static int cmd_selftest(void) {
  // No file of this program may pass 64 MiB, and a program that prints for ever is stopped by the notebook.
  signal(SIGXFSZ, SIG_IGN);
  struct rlimit limit = {(rlim_t)64 << 20, (rlim_t)64 << 20};
  setrlimit(RLIMIT_FSIZE, &limit);
  int todos = 0, fails = 0;
  struct {
    const char* name;
    void (*fn)(void);
  } checks[] = {
      {"exercise 1  nibbles", check_nibbles},
      {"exercise 2  int4_block", check_int4},
      {"exercise 3  e2m1", check_e2m1},
      {"exercise 4  shared_exponent", check_shared_exponent},
      {"exercise 5  mxfp4_block", check_mxfp4},
      {"exercise 6  e4m3", check_e4m3},
      {"exercise 7  nvfp4_tensor", check_nvfp4},
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

// ---- reading cases from stdin: one case per line, tokens separated by blanks ---------------------------

static char g_line[1 << 20];
static char* g_cursor;

static char* next_tok(void) {
  while (*g_cursor == ' ' || *g_cursor == '\t' || *g_cursor == '\r') g_cursor++;
  if (*g_cursor == '\n' || *g_cursor == '\0') return NULL;
  char* start = g_cursor;
  while (*g_cursor && *g_cursor != ' ' && *g_cursor != '\t' && *g_cursor != '\r' && *g_cursor != '\n') g_cursor++;
  if (*g_cursor) *g_cursor++ = '\0';
  return start;
}
static long next_int(void) {
  char* t = next_tok();
  return t ? strtol(t, NULL, 10) : 0;
}
static float next_float(void) {
  char* t = next_tok();
  return t ? float_of((uint32_t)strtoul(t, NULL, 16)) : 0.0f;
}
static int next_case(void) {
  while (fgets(g_line, sizeof g_line, stdin)) {
    g_cursor = g_line;
    while (*g_cursor == ' ' || *g_cursor == '\t') g_cursor++;
    if (*g_cursor && *g_cursor != '\n' && *g_cursor != '\r') return 1;
  }
  return 0;
}
static void print_hex(float f) { printf(" %08x", bits_of(f)); }

#define MAXV 4096

static int cmd_pack(void) {
  static uint8_t nib[MAXV], bytes[MAXV / 2 + 2];
  while (next_case()) {
    int n = (int)next_int();
    for (int i = 0; i < n; i++) nib[i] = (uint8_t)next_int();
    int nb = (n + 1) / 2;
    memset(bytes, GUARD, nb + 1);
    pack_nibbles(nib, n, bytes);
    for (int i = 0; i <= nb; i++) printf(" %d", bytes[i]);
    printf("\n");
  }
  return 0;
}

static int cmd_unpack(void) {
  static uint8_t bytes[MAXV / 2 + 2], nib[MAXV + 2];
  while (next_case()) {
    int n = (int)next_int();
    for (int i = 0; i < (n + 1) / 2; i++) bytes[i] = (uint8_t)next_int();
    memset(nib, 0x55, n + 1);
    unpack_nibbles(bytes, n, nib);
    for (int i = 0; i <= n; i++) printf(" %d", nib[i]);
    printf("\n");
  }
  return 0;
}

static int cmd_nibble_conversion(int to_nibble) {
  while (next_case()) {
    int v = (int)next_int();
    printf(" %d\n", to_nibble ? int4_to_nibble(v) : nibble_to_int4(v));
  }
  return 0;
}

static int cmd_int4q(void) {
  static float x[MAXV];
  static int8_t q[MAXV + 1];
  while (next_case()) {
    int n = (int)next_int();
    for (int i = 0; i < n; i++) x[i] = next_float();
    memset(q, 99, n + 1);
    float scale = int4_quantise_block(x, n, q);
    print_hex(scale);
    for (int i = 0; i <= n; i++) printf(" %d", q[i]);
    printf("\n");
  }
  return 0;
}

static int cmd_int4dq(void) {
  static int8_t q[MAXV];
  static float out[MAXV + 1];
  while (next_case()) {
    float scale = next_float();
    int n = (int)next_int();
    for (int i = 0; i < n; i++) q[i] = (int8_t)next_int();
    for (int i = 0; i <= n; i++) out[i] = 12345.0f;
    int4_dequantise_block(scale, q, n, out);
    for (int i = 0; i <= n; i++) print_hex(out[i]);
    printf("\n");
  }
  return 0;
}

static int cmd_scalar(const char* which) {
  while (next_case()) {
    if (!strcmp(which, "e2d")) print_hex(e2m1_decode((int)next_int()));
    else if (!strcmp(which, "e2e")) printf(" %d", e2m1_encode(next_float()));
    else if (!strcmp(which, "e8d")) print_hex(e8m0_decode((int)next_int()));
    else if (!strcmp(which, "e4d")) print_hex(e4m3_decode((int)next_int()));
    else if (!strcmp(which, "e4e")) printf(" %d", e4m3_encode(next_float()));
    printf("\n");
  }
  return 0;
}

static int cmd_shx(void) {
  static float x[MAXV];
  while (next_case()) {
    int n = (int)next_int(), emax = (int)next_int();
    for (int i = 0; i < n; i++) x[i] = next_float();
    printf(" %d\n", mx_shared_exponent(x, n, emax));
  }
  return 0;
}

static int cmd_mxq(void) {
  static float x[MAXV];
  static uint8_t codes[MAXV + 1];
  while (next_case()) {
    int n = (int)next_int();
    for (int i = 0; i < n; i++) x[i] = next_float();
    memset(codes, 0xEE, n + 1);
    uint8_t sb = 0xEE;
    mxfp4_quantise_block(x, n, &sb, codes);
    printf(" %d", sb);
    for (int i = 0; i <= n; i++) printf(" %d", codes[i]);
    printf("\n");
  }
  return 0;
}

static int cmd_mxdq(void) {
  static uint8_t codes[MAXV];
  static float out[MAXV + 1];
  while (next_case()) {
    int sb = (int)next_int(), n = (int)next_int();
    for (int i = 0; i < n; i++) codes[i] = (uint8_t)next_int();
    for (int i = 0; i <= n; i++) out[i] = 12345.0f;
    mxfp4_dequantise_block((uint8_t)sb, codes, n, out);
    for (int i = 0; i <= n; i++) print_hex(out[i]);
    printf("\n");
  }
  return 0;
}

static int cmd_nvq(void) {
  static float x[MAXV];
  static uint8_t sbytes[MAXV / NV_BLOCK + 2], codes[MAXV + 1];
  while (next_case()) {
    int n = (int)next_int();
    for (int i = 0; i < n; i++) x[i] = next_float();
    int nb = (n + NV_BLOCK - 1) / NV_BLOCK;
    memset(sbytes, 0xEE, nb + 1);
    memset(codes, 0xEE, n + 1);
    float ts = nvfp4_quantise(x, n, sbytes, codes);
    print_hex(ts);
    for (int b = 0; b <= nb; b++) printf(" %d", sbytes[b]);
    for (int i = 0; i <= n; i++) printf(" %d", codes[i]);
    printf("\n");
  }
  return 0;
}

static int cmd_nvdq(void) {
  static uint8_t sbytes[MAXV / NV_BLOCK + 2], codes[MAXV];
  static float out[MAXV + 1];
  while (next_case()) {
    float ts = next_float();
    int n = (int)next_int();
    int nb = (n + NV_BLOCK - 1) / NV_BLOCK;
    for (int b = 0; b < nb; b++) sbytes[b] = (uint8_t)next_int();
    for (int i = 0; i < n; i++) codes[i] = (uint8_t)next_int();
    for (int i = 0; i <= n; i++) out[i] = 12345.0f;
    nvfp4_dequantise(ts, sbytes, codes, n, out);
    for (int i = 0; i <= n; i++) print_hex(out[i]);
    printf("\n");
  }
  return 0;
}

// ---- the whole-tensor round trip: encode, PACK, count the bytes, UNPACK, decode ------------------------
//
// Reads n raw little-endian float32 values, stores them in the named format using your functions and
// the packing of exercise 1, counts the bytes of each stream, decodes from the PACKED bytes alone, and
// writes the n decoded floats. The counts are what the encoder really wrote, not a formula.
static int cmd_roundtrip(const char* fmt, const char* in_path, const char* out_path) {
  FILE* f = fopen(in_path, "rb");
  if (!f) {
    fprintf(stderr, "error: cannot open %s\n", in_path);
    return 3;
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  int n = (int)(size / 4);
  std::vector<float> x(n > 0 ? n : 1), y(n > 0 ? n : 1);
  if (n > 0 && fread(x.data(), 4, n, f) != (size_t)n) {
    fprintf(stderr, "error: short read\n");
    return 3;
  }
  fclose(f);
  std::vector<uint8_t> nib(n + 1), nib2(n + 1), packed((n + 1) / 2 + 1);
  long code_bytes = (n + 1) / 2, scale_bytes = 0, tensor_scale_bytes = 0;
  const bool is_int4 = !strcmp(fmt, "int4"), is_mx = !strcmp(fmt, "mxfp4"), is_nv = !strcmp(fmt, "nvfp4");
  if (!is_int4 && !is_mx && !is_nv) {
    fprintf(stderr, "error: --fmt must be int4, mxfp4 or nvfp4\n");
    return 3;
  }
  if (is_int4) {
    int nb = (n + INT4_BLOCK - 1) / INT4_BLOCK;
    std::vector<float> scales(nb + 1);
    std::vector<int8_t> q(n + 1), q2(n + 1);
    for (int b = 0; b < nb; b++) {
      int lo = b * INT4_BLOCK, len = std::min(INT4_BLOCK, n - lo);
      scales[b] = int4_quantise_block(&x[lo], len, &q[lo]);
    }
    for (int i = 0; i < n; i++) nib[i] = (uint8_t)int4_to_nibble(q[i]);
    pack_nibbles(nib.data(), n, packed.data());
    scale_bytes = 4L * nb;
    unpack_nibbles(packed.data(), n, nib2.data());
    for (int i = 0; i < n; i++) q2[i] = (int8_t)nibble_to_int4(nib2[i]);
    for (int b = 0; b < nb; b++) {
      int lo = b * INT4_BLOCK, len = std::min(INT4_BLOCK, n - lo);
      int4_dequantise_block(scales[b], &q2[lo], len, &y[lo]);
    }
  } else if (is_mx) {
    int nb = (n + MX_BLOCK - 1) / MX_BLOCK;
    std::vector<uint8_t> sb(nb + 1);
    for (int b = 0; b < nb; b++) {
      int lo = b * MX_BLOCK, len = std::min(MX_BLOCK, n - lo);
      mxfp4_quantise_block(&x[lo], len, &sb[b], &nib[lo]);
    }
    pack_nibbles(nib.data(), n, packed.data());
    scale_bytes = nb;
    unpack_nibbles(packed.data(), n, nib2.data());
    for (int b = 0; b < nb; b++) {
      int lo = b * MX_BLOCK, len = std::min(MX_BLOCK, n - lo);
      mxfp4_dequantise_block(sb[b], &nib2[lo], len, &y[lo]);
    }
  } else {
    int nb = (n + NV_BLOCK - 1) / NV_BLOCK;
    std::vector<uint8_t> sb(nb + 1);
    float ts = nvfp4_quantise(x.data(), n, sb.data(), nib.data());
    pack_nibbles(nib.data(), n, packed.data());
    scale_bytes = nb;
    tensor_scale_bytes = 4;
    unpack_nibbles(packed.data(), n, nib2.data());
    nvfp4_dequantise(ts, sb.data(), nib2.data(), n, y.data());
  }
  // Elements stored at the top code (+-7 for INT4, magnitude code 7 for E2M1), and those of them that came back
  // smaller in magnitude than they went in: the format's clipping.
  long top = 0, clipped = 0, nan_out = 0;
  for (int i = 0; i < n; i++) {
    if (std::isnan(y[i])) nan_out++;
    int is_top = is_int4 ? (nibble_to_int4(nib2[i]) == 7 || nibble_to_int4(nib2[i]) == -7) : ((nib2[i] & 7) == 7);
    if (is_top) {
      top++;
      if (fabsf(y[i]) < fabsf(x[i])) clipped++;
    }
  }
  FILE* g = fopen(out_path, "wb");
  if (!g) {
    fprintf(stderr, "error: cannot write %s\n", out_path);
    return 3;
  }
  if (n > 0) fwrite(y.data(), 4, n, g);
  fclose(g);
  printf("@ n=%d\n@ code_bytes=%ld\n@ scale_bytes=%ld\n@ tensor_scale_bytes=%ld\n@ top_codes=%ld\n@ clipped=%ld\n@ nan_out=%ld\n", n,
         code_bytes, scale_bytes, tensor_scale_bytes, top, clipped, nan_out);
  return 0;
}

static int cmd_facts(void) {
  uint16_t probe = 1;
  unsigned char first;
  memcpy(&first, &probe, 1);
  printf("@ int4_block=%d\n@ mx_block=%d\n@ nv_block=%d\n@ e2m1_max=%d\n@ e4m3_max=%d\n@ sizeof_float=%d\n@ little_endian=%d\n", INT4_BLOCK,
         MX_BLOCK, NV_BLOCK, (int)E2M1_MAX, (int)E4M3_MAX, (int)sizeof(float), first == 1 ? 1 : 0);
  return 0;
}

static int usage(void) {
  fprintf(stderr,
          "usage: lesson_bin <command> [options]\n"
          "  selftest                              run the built-in checks for the seven exercises\n"
          "  facts                                 print the contract: block sizes, maxima, byte order\n"
          "  roundtrip --fmt F --in P --out Q      float32 file -> store as int4|mxfp4|nvfp4 -> decode -> float32 file\n"
          "  pack unpack i4n n4i int4q int4dq e2d e2e e8d shx mxq mxdq e4d e4e nvq nvdq\n"
          "                                        one case per stdin line (used by the notebook's checks)\n");
  return 3;
}

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const char* cmd = argv[1];
  const char *fmt = NULL, *in_path = NULL, *out_path = NULL;
  for (int i = 2; i < argc; i++) {
    if (i + 1 >= argc) {
      fprintf(stderr, "error: %s needs a value\n", argv[i]);
      return 3;
    }
    const char *key = argv[i], *v = argv[++i];
    if (!strcmp(key, "--fmt")) fmt = v;
    else if (!strcmp(key, "--in")) in_path = v;
    else if (!strcmp(key, "--out")) out_path = v;
    else {
      fprintf(stderr, "error: unknown option %s\n", key);
      return 3;
    }
  }
  if (!strcmp(cmd, "selftest")) return cmd_selftest();
  if (!strcmp(cmd, "facts")) return cmd_facts();
  if (!strcmp(cmd, "roundtrip")) {
    if (!fmt || !in_path || !out_path) {
      fprintf(stderr, "error: roundtrip needs --fmt, --in and --out\n");
      return 3;
    }
    return cmd_roundtrip(fmt, in_path, out_path);
  }
  if (!strcmp(cmd, "pack")) return cmd_pack();
  if (!strcmp(cmd, "unpack")) return cmd_unpack();
  if (!strcmp(cmd, "i4n")) return cmd_nibble_conversion(1);
  if (!strcmp(cmd, "n4i")) return cmd_nibble_conversion(0);
  if (!strcmp(cmd, "int4q")) return cmd_int4q();
  if (!strcmp(cmd, "int4dq")) return cmd_int4dq();
  if (!strcmp(cmd, "e2d") || !strcmp(cmd, "e2e") || !strcmp(cmd, "e8d") || !strcmp(cmd, "e4d") || !strcmp(cmd, "e4e")) return cmd_scalar(cmd);
  if (!strcmp(cmd, "shx")) return cmd_shx();
  if (!strcmp(cmd, "mxq")) return cmd_mxq();
  if (!strcmp(cmd, "mxdq")) return cmd_mxdq();
  if (!strcmp(cmd, "nvq")) return cmd_nvq();
  if (!strcmp(cmd, "nvdq")) return cmd_nvdq();
  return usage();
}
