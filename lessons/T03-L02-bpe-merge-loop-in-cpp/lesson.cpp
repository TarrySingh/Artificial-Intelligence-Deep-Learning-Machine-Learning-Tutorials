// T03-L02 — the BPE merge loop, in C++.
//
// Four functions are yours to write. Everything else — pre-tokenisation, argument parsing,
// the self-test, the timing harness — is already here and is not part of the exercise.
//
//     make test                         build this file and run its self-test
//     make bench MERGES=300             train on the shipped corpus and print timings
//
// Only the C++ standard library is allowed. No third-party header, no hand-written assembly,
// no threads: the point of this lesson is that plain, ordinary C++ over a flat array is
// already worth a large constant factor against the same algorithm in Python. How large is
// not for this comment to say — section 8 of the notebook measures it on your machine, and
// that measured number is the only one you should ever quote.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace bpe {

// A symbol is an int: 0..255 are raw bytes, 256 is the word boundary, 257 and up are the
// merged symbols this trainer invents. int, not char: a merged symbol has no byte to live in.
constexpr int kBoundary = 256;
constexpr int kFirstMergeId = 257;
constexpr long kMinCount = 2;

// Pair counts live in a flat hash map keyed by the two symbol ids packed into one 64-bit
// integer. A std::pair<int,int> key would need a custom hash and would chase two words of
// memory per probe; this key is one register.
using PairCounts = std::unordered_map<uint64_t, long>;

inline uint64_t pack(int a, int b) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(b));
}
inline int pair_left(uint64_t key) { return static_cast<int>(key >> 32); }
inline int pair_right(uint64_t key) { return static_cast<int>(key & 0xffffffffULL); }

struct Merge {
  int a;
  int b;
  int new_id;
  long count;
};

inline bool is_ascii_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Given: pre-tokenisation. Bytes of one word, then kBoundary, then the next word. Whitespace
// itself is dropped. The Python trainer in lesson.py does exactly this, byte for byte, which
// is what makes the two merge lists comparable.
std::vector<int> pretokenise(const std::string& blob) {
  std::vector<int> seq;
  seq.reserve(blob.size());
  bool in_word = false;
  for (unsigned char c : blob) {
    if (is_ascii_space(c)) {
      if (in_word) {
        seq.push_back(kBoundary);
        in_word = false;
      }
    } else {
      seq.push_back(static_cast<int>(c));
      in_word = true;
    }
  }
  if (in_word) seq.push_back(kBoundary);
  return seq;
}

// ---------------------------------------------------------------------------------------
// EXERCISE 1 — count every adjacent pair in one pass.
//
// Fill `counts` so that counts[pack(x, y)] is the number of positions i where seq[i] == x and
// seq[i+1] == y. Two rules:
//   * a pair is never counted if either side is kBoundary — merging across a word boundary
//     would invent tokens that span two words, which no tokenizer wants;
//   * overlapping occurrences are all counted here. In [7,7,7] the pair (7,7) counts twice,
//     even though applying that merge will only replace one of them. Counting and merging
//     disagree on purpose; exercise 3 is where the disagreement is resolved.
//
// Worked example:
//   pretokenise("ab ab") == [97, 98, 256, 97, 98, 256]
//   count_pairs of that leaves exactly one entry: counts[pack(97, 98)] == 2
//   (97,256) and (256,97) are not entries at all — not zero entries, absent entries.
//
// Cost: one linear scan, O(n). Read each symbol once, keep the previous one in a local.
// ---------------------------------------------------------------------------------------
void count_pairs(const std::vector<int>& seq, PairCounts& counts) {
  counts.clear();
  // YOUR CODE HERE
  throw std::logic_error("NOT_IMPLEMENTED count_pairs");
}

// ---------------------------------------------------------------------------------------
// EXERCISE 2 — pick the winner, deterministically.
//
// Return the pair with the highest count through *out_a, *out_b and *out_count, and return
// true. For an empty map, touch nothing and return false.
//
// The tie-break is the whole exercise. std::unordered_map iterates in an unspecified order,
// so "the first pair I met with the maximum count" is a different answer on a different day,
// a different libc++ or a different corpus size — and a tokenizer whose vocabulary depends on
// hash order is not reproducible. Break ties towards the SMALLER pair: compare a first, then
// b, which is exactly what comparing the packed uint64_t keys does for you.
//
// Worked example:
//   counts = { pack(9,9): 4, pack(2,7): 4 }  ->  returns true, a=2, b=7, count=4
//   counts = { }                             ->  returns false
//
// Cost: one pass over the map, O(number of distinct pairs).
// ---------------------------------------------------------------------------------------
bool best_pair(const PairCounts& counts, int* out_a, int* out_b, long* out_count) {
  // YOUR CODE HERE
  throw std::logic_error("NOT_IMPLEMENTED best_pair");
}

// ---------------------------------------------------------------------------------------
// EXERCISE 3 — rewrite the sequence in place.
//
// Replace every non-overlapping left-to-right occurrence of (a, b) with new_id, shrink the
// vector to the new length with seq.resize(), and return that length.
//
// Do it with two indices into the SAME vector — a read cursor and a write cursor — not by
// building a second vector and swapping. The write cursor never overtakes the read cursor, so
// this is safe, and it is the difference between touching the data once and allocating a
// fresh array on every one of hundreds of merges.
//
// Worked example (the overlap trap):
//   seq = [7, 7, 7], merge (7,7) -> 300
//   correct:   [300, 7]      length 2 — the match consumes both symbols, so the scan resumes
//                                       after them, and the third 7 has nothing to pair with
//   wrong:     [300, 300, 7] length 3 — you advanced the read cursor by one after the match,
//                                       so the middle 7 was consumed twice and the sequence
//                                       barely shrank at all
//   A pair that never occurs must leave the sequence and its length untouched.
//
// Cost: one linear pass, O(n), zero allocations.
// ---------------------------------------------------------------------------------------
size_t apply_merge(std::vector<int>& seq, int a, int b, int new_id) {
  // YOUR CODE HERE
  throw std::logic_error("NOT_IMPLEMENTED apply_merge");
}

// ---------------------------------------------------------------------------------------
// EXERCISE 4 — the loop itself.
//
// Repeat at most n_merges times: count the pairs, take the best one, record it, apply it.
// Return the merges in the order they were made. Stop early — before recording anything —
// when there is no pair at all, or when the best pair occurs fewer than kMinCount times: a
// merge that fires once buys no compression and just burns a vocabulary slot.
//
// The k-th merge (counting from zero) is given the id kFirstMergeId + k. Ids are dense and
// sequential, so the merge list alone is enough to rebuild the vocabulary later.
//
// Worked example:
//   pretokenise("aaab aaab aaab") trained for 4 merges begins with
//   Merge{a='a', b='a', new_id=257, count=6} — (a,a) occurs twice per word, three words
//   pretokenise("abcdef") trained for 5 merges returns an EMPTY list: every pair occurs once.
//
// Cost: n_merges full rescans of a sequence that only shrinks — O(n_merges * n). That is the
// cost model this lesson is measuring, and it is identical in both languages.
// ---------------------------------------------------------------------------------------
std::vector<Merge> train(std::vector<int>& seq, int n_merges) {
  std::vector<Merge> merges;
  PairCounts counts;
  // YOUR CODE HERE
  throw std::logic_error("NOT_IMPLEMENTED train");
}

}  // namespace bpe

// =========================================================================================
// Below this line is plumbing: argument parsing, the sub-commands the notebook calls, and
// the self-test that `make test` runs. Nothing here is an exercise.
// =========================================================================================
namespace {

using bpe::PairCounts;

std::vector<int> parse_ints(const std::string& csv) {
  std::vector<int> out;
  std::string field;
  std::istringstream stream(csv);
  while (std::getline(stream, field, ',')) {
    if (!field.empty()) out.push_back(std::atoi(field.c_str()));
  }
  return out;
}

// "97:98:5,98:99:2" -> {(97,98): 5, (98,99): 2}
PairCounts parse_counts(const std::string& csv) {
  PairCounts counts;
  std::string field;
  std::istringstream stream(csv);
  while (std::getline(stream, field, ',')) {
    if (field.empty()) continue;
    int a = 0, b = 0;
    long n = 0;
    if (std::sscanf(field.c_str(), "%d:%d:%ld", &a, &b, &n) != 3)
      throw std::runtime_error("bad --counts field: " + field);
    counts[bpe::pack(a, b)] = n;
  }
  return counts;
}

std::string option(int argc, char** argv, const std::string& name, const std::string& fallback) {
  for (int i = 0; i + 1 < argc; ++i)
    if (name == argv[i]) return std::string(argv[i + 1]);
  return fallback;
}

std::string join(const std::vector<int>& seq) {
  std::string out;
  char buf[16];
  for (size_t i = 0; i < seq.size(); ++i) {
    std::snprintf(buf, sizeof buf, "%d", seq[i]);
    if (i) out += ',';
    out += buf;
  }
  return out;
}

std::string read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open corpus: " + path);
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("selftest failed: " + what);
}

// Small cases a human can check on paper. The notebook's checks are far stricter; these exist
// so `make test` alone tells a student whether the four pieces hold together.
int selftest() {
  // "ab ab" -> a b BOUNDARY a b BOUNDARY
  const std::vector<int> seq = bpe::pretokenise("ab ab");
  require(seq.size() == 6 && seq[2] == bpe::kBoundary && seq[5] == bpe::kBoundary,
          "pretokenise should end every word with the boundary symbol");

  PairCounts counts;
  bpe::count_pairs(seq, counts);
  require(counts.size() == 1, "only (a,b) is a countable pair; pairs touching the boundary are not");
  require(counts[bpe::pack('a', 'b')] == 2, "(a,b) occurs twice in \"ab ab\"");
  std::printf("  count_pairs ok\n");

  int a = 0, b = 0;
  long n = 0;
  PairCounts tie;
  tie[bpe::pack(9, 9)] = 4;
  tie[bpe::pack(2, 7)] = 4;
  require(bpe::best_pair(tie, &a, &b, &n), "best_pair must report a winner for a non-empty map");
  require(a == 2 && b == 7 && n == 4, "ties break towards the smaller (a,b), not towards insertion order");
  PairCounts empty;
  require(!bpe::best_pair(empty, &a, &b, &n), "best_pair must return false for an empty map");
  std::printf("  best_pair ok\n");

  std::vector<int> triple{7, 7, 7};
  const size_t left = bpe::apply_merge(triple, 7, 7, 300);
  require(left == 2 && triple.size() == 2 && triple[0] == 300 && triple[1] == 7,
          "merging (7,7) in [7,7,7] yields [300,7]: matches do not overlap");
  std::vector<int> untouched{1, 2, 3};
  bpe::apply_merge(untouched, 8, 9, 300);
  require(untouched.size() == 3, "a pair that never occurs must leave the sequence alone");
  std::printf("  apply_merge ok\n");

  std::vector<int> corpus = bpe::pretokenise("aaab aaab aaab");
  const std::vector<bpe::Merge> merges = bpe::train(corpus, 4);
  require(!merges.empty(), "train produced no merges on a corpus that clearly repeats");
  require(merges[0].a == 'a' && merges[0].b == 'a' && merges[0].new_id == bpe::kFirstMergeId,
          "the first merge should be the commonest pair (a,a), numbered 257");
  require(merges.size() <= 4, "train must stop at n_merges");
  std::vector<int> quiet = bpe::pretokenise("abcdef");
  require(bpe::train(quiet, 5).empty(), "no pair occurs twice here, so train must stop immediately");
  std::printf("  train ok\n");

  std::printf("selftest ok\n");
  return 0;
}

int cmd_count(int argc, char** argv) {
  const std::vector<int> seq = parse_ints(option(argc, argv, "--seq", ""));
  PairCounts counts;
  bpe::count_pairs(seq, counts);
  std::vector<uint64_t> keys;
  keys.reserve(counts.size());
  for (const auto& entry : counts) keys.push_back(entry.first);
  std::sort(keys.begin(), keys.end());
  for (uint64_t key : keys)
    std::printf("pair %d %d %ld\n", bpe::pair_left(key), bpe::pair_right(key), counts[key]);
  std::printf("pairs %zu\n", keys.size());
  return 0;
}

int cmd_best(int argc, char** argv) {
  const PairCounts counts = parse_counts(option(argc, argv, "--counts", ""));
  int a = 0, b = 0;
  long n = 0;
  if (bpe::best_pair(counts, &a, &b, &n))
    std::printf("best %d %d %ld\n", a, b, n);
  else
    std::printf("best none\n");
  return 0;
}

int cmd_merge(int argc, char** argv) {
  std::vector<int> seq = parse_ints(option(argc, argv, "--seq", ""));
  const int a = std::atoi(option(argc, argv, "--a", "0").c_str());
  const int b = std::atoi(option(argc, argv, "--b", "0").c_str());
  const int new_id = std::atoi(option(argc, argv, "--new-id", "257").c_str());
  const size_t length = bpe::apply_merge(seq, a, b, new_id);
  std::printf("seq %s\n", join(seq).c_str());
  std::printf("length %zu\n", length);
  return 0;
}

int cmd_train(int argc, char** argv) {
  const std::string path = option(argc, argv, "--corpus", "assets/corpus.txt");
  const int n_merges = std::atoi(option(argc, argv, "--merges", "200").c_str());
  const std::string blob = read_file(path);
  std::vector<int> seq = bpe::pretokenise(blob);
  std::printf("bytes %zu\n", blob.size());
  std::printf("symbols %zu\n", seq.size());

  // Time the training loop only: file reading and pre-tokenisation are the same work in
  // either language and would flatter whichever side did them faster.
  const auto start = std::chrono::steady_clock::now();
  const std::vector<bpe::Merge> merges = bpe::train(seq, n_merges);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

  for (size_t k = 0; k < merges.size(); ++k)
    std::printf("merge %zu %d %d %d %ld\n", k, merges[k].a, merges[k].b, merges[k].new_id,
                merges[k].count);
  std::printf("merges %zu\n", merges.size());
  std::printf("final_symbols %zu\n", seq.size());
  std::printf("train_seconds %.6f\n", seconds);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "selftest";
  try {
    if (cmd == "selftest") return selftest();
    if (cmd == "count") return cmd_count(argc, argv);
    if (cmd == "best") return cmd_best(argc, argv);
    if (cmd == "merge") return cmd_merge(argc, argv);
    if (cmd == "train") return cmd_train(argc, argv);
    std::fprintf(stderr, "unknown command %s (selftest|count|best|merge|train)\n", cmd.c_str());
    return 2;
  } catch (const std::logic_error& e) {
    // An exercise that still has its TODO in place. Exit code 3 is the signal the notebook
    // turns back into a Python NotImplementedError.
    std::fprintf(stderr, "%s\n", e.what());
    return 3;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
