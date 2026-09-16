# %% [markdown]
# # T03-L02 · The BPE merge loop, in C++
#
# **You will build:** a byte-pair-encoding trainer in C++ whose merge list is *identical* to
# the Python trainer's from T03-L01, on the same corpus — and a measurement of how much time
# the language change bought on your machine.
#
# **Time:** ~60 minutes · **Runs on:** a laptop CPU, no GPU, no download
# · **Prerequisites:** `T03-L01-bpe-from-scratch`
#
# By the end you will be able to:
# 1. Implement the four-function merge loop in C++ — count, argmax, merge, train — with nothing
#    but `std::` containers.
# 2. Measure the naive rescan's cost in both languages on one identical corpus, and report the
#    speedup your own machine produces.
# 3. Show that your C++ merge list matches Python's exactly, and name the one rule that makes
#    that reproducibility possible.
# 4. Explain why the speedup is a constant factor and not an asymptotic one.
# 5. Say which language each production tokenizer is really written in, and what that does and
#    does not change about the argument.
#
# **You edit `lesson.cpp`, not this file.** This notebook builds your C++, feeds it cases, and
# checks its answers against a Python trainer that is given to you, already working.

# %%
# Setup: one cell, everything the lesson needs, with versions printed.
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Work that costs real seconds lives inside `if __name__ == "__main__":` blocks, so the
# autograder can import this file without re-running the whole lesson. In a notebook
# __name__ IS "__main__", so every one of those cells runs normally when you run it.
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
IS_REFERENCE = HERE.name == "solutions"
LESSON_ROOT = HERE.parent if IS_REFERENCE else HERE
CPP_SRC = HERE / "lesson.cpp"                      # the file you edit
CPP_BIN = LESSON_ROOT / "build" / ("lesson_ref" if IS_REFERENCE else "lesson")
CORPUS = LESSON_ROOT / "assets" / "corpus.txt"     # ships with the lesson; nothing is downloaded

HEADLINE_MERGES = 300      # the size of the race in section 8
ORACLE_BYTES = 40_000      # a slice of the corpus, for the checks that compare merge lists
ORACLE_MERGES = 30

print("python", sys.version.split()[0], "·", platform.machine(), platform.system())
print("corpus", CORPUS.name, CORPUS.stat().st_size, "bytes")
print("source", CPP_SRC.relative_to(LESSON_ROOT))
_paths = subprocess.run(["make", "-C", str(LESSON_ROOT), f"PYTHON={sys.executable}", "paths"],
                        capture_output=True, text=True)
print(_paths.stdout.strip() or _paths.stderr.strip())

# %% [markdown]
# ## 1. The algorithm, in one paragraph and one cell
#
# BPE starts from bytes and invents a new symbol for the commonest adjacent pair, over and
# over. Sennrich, Haddow and Birch brought it to neural translation by borrowing a compression
# algorithm (sourced in `claims.yaml`); every modern tokenizer still runs this loop.
#
# The Python below is the T03-L01 trainer, reproduced here so this lesson stands alone and so
# you have something exact to test against. **Read it — it is the specification your C++ has to
# match.** Four steps: pre-tokenise, count, pick, merge.

# %%
BOUNDARY = 256          # the symbol between two words. Never merged across.
FIRST_MERGE_ID = 257    # the k-th merge is given id 257 + k
MIN_COUNT = 2           # a pair that occurs once buys no compression
_WHITESPACE = frozenset(b" \t\n\r\f\v")


def pretokenise(blob: bytes) -> list:
    """Bytes of a word, then BOUNDARY, then the next word. Whitespace itself is dropped."""
    seq, in_word = [], False
    for byte in blob:
        if byte in _WHITESPACE:
            if in_word:
                seq.append(BOUNDARY)
                in_word = False
        else:
            seq.append(byte)
            in_word = True
    if in_word:
        seq.append(BOUNDARY)
    return seq


def count_pairs_py(seq: list) -> dict:
    """Count adjacent pairs, skipping any pair that touches BOUNDARY. One O(n) scan."""
    counts = {}
    prev = seq[0] if seq else BOUNDARY
    for i in range(1, len(seq)):
        cur = seq[i]
        if prev != BOUNDARY and cur != BOUNDARY:
            key = (prev, cur)
            counts[key] = counts.get(key, 0) + 1
        prev = cur
    return counts


def best_pair_py(counts: dict):
    """Highest count wins; ties break towards the smaller (a, b). None for an empty dict."""
    best, best_n = None, 0
    for pair, n in counts.items():
        if best is None or n > best_n or (n == best_n and pair < best):
            best, best_n = pair, n
    return (best[0], best[1], best_n) if best is not None else None


def apply_merge_py(seq: list, a: int, b: int, new_id: int) -> list:
    """Replace every non-overlapping left-to-right occurrence of (a, b) with new_id."""
    out, i, n = [], 0, len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_py(seq: list, n_merges: int) -> dict:
    """Run the naive loop. Returns the merges, the final length and the measured seconds."""
    merges = []
    start = time.perf_counter()
    for k in range(n_merges):
        counts = count_pairs_py(seq)
        winner = best_pair_py(counts)
        if winner is None or winner[2] < MIN_COUNT:
            break
        a, b, count = winner
        new_id = FIRST_MERGE_ID + k
        merges.append((a, b, new_id, count))
        seq = apply_merge_py(seq, a, b, new_id)
    return {"merges": merges, "final_symbols": len(seq), "seconds": time.perf_counter() - start}


def corpus_symbols(path=None, limit: int = 0) -> list:
    """Pre-tokenised corpus, optionally only the first `limit` bytes (cut at whitespace)."""
    blob = (path or CORPUS).read_bytes()
    if limit and limit < len(blob):
        cut = limit
        while cut > 0 and not blob[cut - 1:cut].isspace():
            cut -= 1
        blob = blob[:cut]
    return pretokenise(blob)


# %%
if __name__ == "__main__":
    _demo = train_py(pretokenise(b"the cat sat on the mat the cat sat"), 3)
    print("first merges on a nine-word corpus:")
    for _a, _b, _new, _n in _demo["merges"]:
        _left = chr(_a) if _a < 256 else f"<{_a}>"
        _right = chr(_b) if _b < 256 else f"<{_b}>"
        print(f"  ({_left!r}, {_right!r}) -> {_new}   seen {_n} times")

# %% [markdown]
# ## 2. The cost model: every merge rescans everything
#
# Look again at `train_py`. Each iteration counts pairs over the *whole* sequence, then
# rebuilds the whole sequence. Nothing is remembered between merges. The work is
# `n_merges × n_symbols`, and since one merge changes only a handful of positions, almost all
# of that scanning re-derives counts that did not change.
#
# Do not take that on faith. Time the Python trainer on three slices of the corpus with the
# merge count held fixed, and watch the seconds track the symbol count.

# %%
def python_scaling(fractions=(0.25, 0.5, 1.0), merges: int = 30) -> list:
    """Time train_py on increasing slices. Rows of measured numbers, nothing typed by hand."""
    total = CORPUS.stat().st_size
    rows = []
    for fraction in fractions:
        seq = corpus_symbols(limit=int(total * fraction))
        result = train_py(list(seq), merges)
        rows.append({
            "bytes": int(total * fraction),
            "symbols": len(seq),
            "seconds": result["seconds"],
            "ns_per_symbol_per_merge": 1e9 * result["seconds"] / (len(seq) * merges),
        })
    return rows


if __name__ == "__main__":
    print(f"{'bytes':>8} {'symbols':>9} {'seconds':>8} {'ns/symbol/merge':>16}")
    for _row in python_scaling():
        print(f"{_row['bytes']:8d} {_row['symbols']:9d} {_row['seconds']:8.3f}"
              f" {_row['ns_per_symbol_per_merge']:16.1f}")
    print("\nSeconds roughly double when the corpus doubles. The last column is flat, and it")
    print("is the constant you are about to attack: the cost of one symbol, one time, here.")

# %% [markdown]
# ## 3. Your C++ binary, and how this notebook talks to it
#
# `lesson.cpp` compiles to a small command-line tool. The notebook builds it with `make` and
# calls one sub-command per exercise, so each function is graded on its own: an `apply_merge`
# that works still scores even while `train` is a TODO.
#
# Run this cell before writing any C++. It should fail — loudly, and by name.

# %%
_BUILD_LOG = {}


def build_cpp(force: bool = False) -> Path:
    """Compile CPP_SRC to CPP_BIN via the lesson Makefile. Cached: it builds once per session."""
    if _BUILD_LOG.get("built") and not force:
        return CPP_BIN
    if shutil.which("make") is None:
        raise RuntimeError("no `make` on PATH — this lesson needs make and a C++17 compiler")
    proc = subprocess.run(
        ["make", "-C", str(LESSON_ROOT),
         f"SRC={CPP_SRC.relative_to(LESSON_ROOT)}",
         f"BIN={CPP_BIN.relative_to(LESSON_ROOT)}", "all"],
        capture_output=True, text=True,
    )
    _BUILD_LOG["stdout"], _BUILD_LOG["stderr"] = proc.stdout, proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            "lesson.cpp did not compile. The compiler's FIRST complaint is usually the real "
            f"one, and it names a line:\n{proc.stdout}\n{proc.stderr}"
        )
    _BUILD_LOG["built"] = True
    return CPP_BIN


def run_cpp(args: list) -> dict:
    """Run the binary and parse its `tag value...` output into a dict.

    Exit code 3 means an exercise still has its TODO, and becomes a Python
    NotImplementedError, so the grader can tell "not written yet" from "written and wrong".
    """
    binary = build_cpp()
    proc = subprocess.run([str(binary), *args], capture_output=True, text=True, cwd=LESSON_ROOT)
    if proc.returncode == 3:
        raise NotImplementedError(proc.stderr.strip() or "an exercise still has its TODO")
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} {' '.join(args[:2])} exited {proc.returncode}:\n"
                           f"{proc.stdout}\n{proc.stderr}")
    out = {"pair_counts": {}, "merges": [], "raw": proc.stdout}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        tag, rest = parts[0], parts[1:]
        if tag == "pair":
            out["pair_counts"][(int(rest[0]), int(rest[1]))] = int(rest[2])
        elif tag == "merge":
            out["merges"].append(tuple(int(x) for x in rest[1:5]))
        elif tag == "best":
            out["best"] = None if rest[0] == "none" else tuple(int(x) for x in rest[:3])
        elif tag == "seq":
            out["seq"] = [int(x) for x in rest[0].split(",")] if rest else []
        elif tag == "train_seconds":
            out["train_seconds"] = float(rest[0])
        elif tag in ("pairs", "length", "bytes", "symbols", "final_symbols"):
            out[tag] = int(rest[0])
    return out


def cpp_selftest() -> str:
    """`make test` — your binary checking its own four functions. Returns its output."""
    build_cpp()
    proc = subprocess.run(
        ["make", "-C", str(LESSON_ROOT), f"SRC={CPP_SRC.relative_to(LESSON_ROOT)}",
         f"BIN={CPP_BIN.relative_to(LESSON_ROOT)}", "test"],
        capture_output=True, text=True,
    )
    if "NOT_IMPLEMENTED" in proc.stderr:
        raise NotImplementedError(proc.stderr.strip().splitlines()[-1])
    if proc.returncode != 0:
        raise AssertionError(f"make test failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def cpp_count_pairs(seq: list) -> dict:
    """Your count_pairs, as a Python dict."""
    return run_cpp(["count", "--seq", ",".join(str(s) for s in seq)])["pair_counts"]


def cpp_best_pair(counts: dict):
    """Your best_pair, handed counts directly — it does not depend on exercise 1."""
    packed = ",".join(f"{a}:{b}:{n}" for (a, b), n in counts.items())
    return run_cpp(["best", "--counts", packed])["best"]


def cpp_apply_merge(seq: list, a: int, b: int, new_id: int) -> list:
    """Your apply_merge, as a Python list."""
    return run_cpp(["merge", "--seq", ",".join(str(s) for s in seq),
                    "--a", str(a), "--b", str(b), "--new-id", str(new_id)])["seq"]


def cpp_train(corpus_path=None, merges: int = HEADLINE_MERGES) -> dict:
    """Your train, on a corpus file, including the binary's own timing of the loop."""
    return run_cpp(["train", "--corpus", str(corpus_path or CORPUS), "--merges", str(merges)])


def oracle_corpus(nbytes: int = ORACLE_BYTES) -> Path:
    """Write a fixed slice of the corpus to build/, so both trainers read the same file."""
    blob = CORPUS.read_bytes()[:nbytes]
    while blob and not blob[-1:].isspace():
        blob = blob[:-1]
    path = LESSON_ROOT / "build" / f"oracle_{CPP_BIN.name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


if __name__ == "__main__":
    try:
        print(cpp_selftest())
    except NotImplementedError as _exc:
        print("as expected, nothing is implemented yet:", _exc)

# %% [markdown]
# ## 4. Exercise 1 — `count_pairs`
#
# Open `lesson.cpp`, find EXERCISE 1, fill it in. One scan, one local holding the previous
# symbol, and no pair counted when either side is `kBoundary`. Then run this check.

# %%
def _check_count_pairs() -> None:
    build_cpp(force=True)
    seq = pretokenise(b"ab ab")
    got, want = cpp_count_pairs(seq), count_pairs_py(seq)
    assert got == want, (
        f"count_pairs on 'ab ab' gave {got}, expected {want} — entries mentioning {BOUNDARY} "
        "mean you are counting across the word boundary; every count equal to 1 means you are "
        "assigning where you meant to increment."
    )
    overlap = cpp_count_pairs([7, 7, 7, BOUNDARY])
    assert overlap == {(7, 7): 2}, (
        f"[7,7,7] should report (7,7) twice, got {overlap} — counting is allowed to overlap; "
        "only merging is not."
    )
    assert cpp_count_pairs([]) == {} and cpp_count_pairs([5]) == {}, (
        "a sequence shorter than two symbols has no pairs — guard the loop rather than "
        "indexing seq[i+1] past the end."
    )
    big = corpus_symbols(limit=20_000)
    mine = cpp_count_pairs(big)
    assert mine == count_pairs_py(big), (
        "your counts match on toy input but not on real text — the usual causes are holding "
        "symbols in a char (they exceed 255 after the first merge) and dropping the last pair."
    )
    print(f"exercise 1 looks right: {len(mine)} distinct pairs across {len(big)} real symbols")


if __name__ == "__main__":
    _check_count_pairs()

# %% [markdown]
# ## 5. Exercise 2 — `best_pair`, and why ties matter
#
# Two pairs tie at the top. Whichever you return becomes token 257, which changes every merge
# after it. So the tie-break is not a detail: it is the difference between a vocabulary you can
# ship and one that differs between runs. `std::unordered_map` promises you no order at all.

# %%
def _check_best_pair() -> None:
    build_cpp()
    tie = cpp_best_pair({(9, 9): 4, (2, 7): 4})
    assert tie == (2, 7, 4), (
        f"a tie between (9,9) and (2,7) must go to (2,7), got {tie} — comparing the packed "
        "keys does this for you; iteration order does not."
    )
    assert cpp_best_pair({(1, 2): 3, (4, 5): 9, (6, 7): 5}) == (4, 5, 9), (
        "the highest count must win outright when nothing ties — check the comparison is > "
        "and not >=, which lets a later equal-count pair displace the leader."
    )
    assert cpp_best_pair({}) is None, (
        "an empty count map must return false, not a pair of zeros — train relies on that "
        "signal to stop."
    )
    # The check the toy cases above cannot make. One tied set, three insertion orders: a real
    # tie-break answers the same every time, while "the first maximum I met" follows whatever
    # order the hash table happened to produce.
    _tied = [(3, 4), (17, 18), (41, 42), (88, 89), (150, 151), (260, 261), (301, 302)]
    answers = {cpp_best_pair({pair: 9 for pair in order})
               for order in (_tied, list(reversed(_tied)), _tied[3:] + _tied[:3])}
    assert answers == {(3, 4, 9)}, (
        f"the same seven tied pairs produced {sorted(answers)} — the only thing that changed "
        "was the order they were inserted in, so you are returning the first maximum the "
        "container handed you rather than the smallest (a,b). This is the check the small "
        "cases above can pass without a tie-break at all; compare the packed keys."
    )
    counts = count_pairs_py(corpus_symbols(limit=20_000))
    mine, reference = cpp_best_pair(counts), best_pair_py(counts)
    assert mine == reference, (
        f"on real text your winner is {mine} and Python's is {reference} — same tie-break, or "
        "the two merge lists diverge at the first tie."
    )
    print(f"exercise 2 looks right: winner on a real slice is {mine}")


if __name__ == "__main__":
    _check_best_pair()

# %% [markdown]
# ## 6. Exercise 3 — `apply_merge`, in place
#
# Two cursors over one vector: read and write. The write cursor can never overtake the read
# cursor, so there is no need for a second array — and no allocation inside the hot loop.

# %%
def _check_apply_merge() -> None:
    build_cpp()
    triple = cpp_apply_merge([7, 7, 7], 7, 7, 300)
    assert triple == [300, 7], (
        f"[7,7,7] merged on (7,7) must be [300,7], got {triple} — after a match, advance the "
        "read cursor by two; advancing by one consumes the same symbol twice."
    )
    assert cpp_apply_merge([1, 2, 3], 8, 9, 300) == [1, 2, 3], (
        "a pair that does not occur must leave the sequence untouched — your resize() is "
        "probably using the read cursor instead of the write cursor."
    )
    assert cpp_apply_merge([1, 2, 1, 2], 1, 2, 300) == [300, 300], (
        "two separate occurrences must both merge — do not stop at the first match."
    )
    assert cpp_apply_merge([5, 1, 2], 1, 2, 300) == [5, 300], (
        "symbols before a match must be copied through, in order."
    )
    seq = corpus_symbols(limit=20_000)
    a, b, _ = best_pair_py(count_pairs_py(seq))
    mine = cpp_apply_merge(seq, a, b, 999)
    assert mine == apply_merge_py(seq, a, b, 999), (
        "toy cases pass but real text does not — the classic cause is a match at the very "
        "last position, where seq[i+1] is off the end."
    )
    print(f"exercise 3 looks right: merging ({a},{b}) takes {len(seq)} symbols to {len(mine)}")


if __name__ == "__main__":
    _check_apply_merge()

# %% [markdown]
# ## 7. Exercise 4 — `train`, and the proof that it matches Python
#
# Now the loop: count, pick, record, apply, repeat. Stop when the best pair occurs fewer than
# `kMinCount` times. The test that matters is not "does this look plausible" — it is whether
# your merge list is **identical** to the Python trainer's on the same bytes.

# %%
def _check_train() -> None:
    build_cpp()
    path = oracle_corpus()
    cpp = cpp_train(path, ORACLE_MERGES)
    ref = train_py(pretokenise(path.read_bytes()), ORACLE_MERGES)
    assert len(cpp["merges"]) == len(ref["merges"]), (
        f"C++ made {len(cpp['merges'])} merges, Python {len(ref['merges'])} — too few means "
        "your stop rule fires early (it is count < kMinCount, tested before recording); too "
        "many means you are not stopping at n_merges."
    )
    for k, (got, want) in enumerate(zip(cpp["merges"], ref["merges"])):
        assert got == want, (
            f"merge {k} differs: C++ {got}, Python {want}. Everything before it agrees, so the "
            f"divergence starts here. A wrong new_id means you are not numbering from "
            f"{FIRST_MERGE_ID}+k; a different pair at equal counts is the tie-break."
        )
    assert cpp["final_symbols"] == ref["final_symbols"], (
        f"identical merges but different final lengths ({cpp['final_symbols']} against "
        f"{ref['final_symbols']}) — train is recording merges it never applies."
    )
    print(f"exercise 4 looks right: {len(cpp['merges'])} merges identical to Python, "
          f"{cpp['symbols']} symbols down to {cpp['final_symbols']}")


if __name__ == "__main__":
    _check_train()

# %% [markdown]
# ## 8. The measurement
#
# Same corpus, same algorithm, same merge list, two languages. The only honest comparison
# times the training loop alone — not file reading, not pre-tokenisation, and certainly not
# compilation. Every number printed below comes from the run you just did.

# %%
def speed_report(merges: int = HEADLINE_MERGES) -> dict:
    """Train both implementations on the full corpus and return the measured comparison."""
    cpp = cpp_train(CORPUS, merges)
    ref = train_py(corpus_symbols(), merges)
    return {
        "bytes": CORPUS.stat().st_size,
        "symbols": cpp["symbols"],
        "merges": len(cpp["merges"]),
        "python_seconds": ref["seconds"],
        "cpp_seconds": cpp["train_seconds"],
        "speedup": ref["seconds"] / cpp["train_seconds"],
        "identical_merges": list(cpp["merges"]) == list(ref["merges"]),
        "final_symbols": cpp["final_symbols"],
    }


def _check_speedup() -> None:
    report = speed_report()
    assert report["identical_merges"], (
        "the trainers disagree on the full corpus even though the slice matched — a speed "
        "comparison between two different answers is worthless; fix exercise 4 first."
    )
    assert report["speedup"] > 1.5, (
        f"C++ came out only {report['speedup']:.2f}x faster — check that you built with -O2 "
        "(the Makefile does), that apply_merge does not allocate a fresh vector per merge, "
        "and that nothing prints inside the loop."
    )
    print(f"corpus        {report['bytes']} bytes -> {report['symbols']} symbols")
    print(f"merges        {report['merges']} (identical lists: {report['identical_merges']})")
    print(f"compression   {report['symbols']} -> {report['final_symbols']} symbols")
    print(f"python        {report['python_seconds']:.3f} s")
    print(f"c++           {report['cpp_seconds']:.3f} s")
    print(f"speedup       {report['speedup']:.1f}x on this machine")


if __name__ == "__main__":
    _check_speedup()

# %% [markdown]
# ## 9. What the speedup is, and what it is not
#
# Your C++ is faster because every symbol is an `int` in a flat array, every pair is one
# 64-bit key, and the loop compiles to instructions instead of bytecode dispatch. That is a
# **constant factor**. Both programs still do `n_merges × n_symbols` work, so doubling the
# corpus still doubles both. Run this and watch the C++ curve keep Python's shape.

# %%
def cpp_scaling(fractions=(0.25, 0.5, 1.0), merges: int = 30) -> list:
    """Time the C++ trainer on the same slices section 2 used, for the same merge count."""
    total = CORPUS.stat().st_size
    rows = []
    for fraction in fractions:
        result = cpp_train(oracle_corpus(int(total * fraction)), merges)
        rows.append({"symbols": result["symbols"], "seconds": result["train_seconds"],
                     "ns_per_symbol_per_merge": 1e9 * result["train_seconds"]
                     / (result["symbols"] * merges)})
    return rows


if __name__ == "__main__":
    _py_rows, _cpp_rows = python_scaling(), cpp_scaling()
    print("per symbol, per merge — the only column that can tell a constant from a curve")
    print(f"{'symbols':>9} {'python ns':>11} {'c++ ns':>9} {'ratio':>9}")
    for _p, _c in zip(_py_rows, _cpp_rows):
        print(f"{_c['symbols']:9d} {_p['ns_per_symbol_per_merge']:11.1f}"
              f" {_c['ns_per_symbol_per_merge']:9.1f}"
              f" {_p['ns_per_symbol_per_merge'] / _c['ns_per_symbol_per_merge']:8.1f}x")
    print("\nRead the two middle columns DOWN, not across. Each is roughly flat as the corpus")
    print("grows, and that flatness is the straight line — the same line in both languages.")
    print("The ratio column is the whole of what the language bought: a constant. To bend the")
    print("line you would have to stop rescanning — keep the counts between merges and update")
    print("only the positions the merge touched. That is an algorithmic change, and no amount")
    print("of C++ substitutes for it.")

# %% [markdown]
# ## 10. The honest footnote
#
# The argument here is *systems language*, not *C++ specifically* — and the evidence cuts both
# ways, so here it is straight (every source in `claims.yaml`):
#
# - **HuggingFace `tokenizers`, the trainer behind most published BPE vocabularies, is Rust**,
#   with bindings for Python, Node and Ruby. **OpenAI's `tiktoken` is Rust too**, behind PyO3.
# - **Google's SentencePiece is C++**, and much of the research world trains with it.
#
# Rust wins those choices on memory safety and tooling, not on the loop you just wrote: a hash
# lookup and a two-cursor scan compile to the same shape in either. What does not change is the
# thing you measured — the interpreter was the cost, not the syntax.
#
# `tokenizers` is installed here, so run the real thing on your corpus. It is deliberately
# *not* a like-for-like race: it counts over a word-frequency table instead of rescanning a
# flat array, which is exactly the algorithmic fix section 9 described. You are seeing both
# wins at once, which is why the gap is bigger than yours.

# %%
def production_reference(vocab_size: int = 256 + HEADLINE_MERGES + 1) -> dict:
    """Train HuggingFace's Rust BPE on the same corpus. A different algorithm, on purpose.

    The vocabulary budget is derived rather than typed: one slot per byte value, one per merge
    this lesson asks for, plus the unknown token, so both trainers are at least asked for the
    same size. It is still not a like-for-like race, for the reason in the text above.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False)
    words = CORPUS.read_text(encoding="utf-8", errors="replace").split()
    start = time.perf_counter()
    tokenizer.train_from_iterator(words, trainer)
    return {"seconds": time.perf_counter() - start, "vocab": tokenizer.get_vocab_size()}


if __name__ == "__main__":
    try:
        import tokenizers
        _hf = production_reference()
        print(f"huggingface tokenizers {tokenizers.__version__} (Rust): {_hf['seconds']:.3f} s "
              f"for a vocabulary of {_hf['vocab']}")
        print("different algorithm, different tie-breaks, different vocabulary — read that")
        print("number as scale, not as a scoreboard.")
    except Exception as _exc:                      # noqa: BLE001 - an optional demonstration
        print("skipping the production reference:", _exc)

# %% [markdown]
# ## 11. Common mistakes
#
# - **Counting pairs across the boundary.** You invent tokens spanning two words and diverge
#   from Python at the first merge. Skip the pair; do not record it as zero.
# - **Advancing by one after a match in `apply_merge`.** `[7,7,7]` becomes `[300,300,7]`: the
#   middle 7 is consumed twice, the sequence barely shrinks, and the final length disagrees
#   with Python's. Do not take that on trust either — the cell below runs the bug.
# - **Letting hash order pick the winner.** It passes today and fails on another machine, or
#   after the corpus grows by one word. The tie-break *is* the reproducibility guarantee.
# - **`char` or `uint8_t` for symbols.** The 257th symbol has nowhere to live. Use `int`.
# - **Building a fresh `std::vector` per merge.** Correct, but it allocates and copies hundreds
#   of times, and hands back much of the constant factor you came for.
# - **`std::map` instead of `std::unordered_map`.** A tree costs a `log n` factor of cache
#   misses per probe to buy an ordering you need exactly once — at the tie-break, which the
#   packed key already gives you for free.
# - **Timing the wrong thing.** Including `make`, file reading or pre-tokenisation measures the
#   harness. The binary times the loop and nothing else; match that in anything you report.
# - **Benchmarking a `-O0` build.** Unoptimised C++ can lose to Python outright. The Makefile
#   uses `-O2`; if you compile by hand, match it.

# %%
def apply_merge_advancing_by_one(seq: list, a: int, b: int, new_id: int) -> list:
    """`apply_merge_py` with one character changed: `i += 2` became `i += 1`.

    This is the commonest wrong answer to exercise 3. It is here so you can watch what it
    does rather than guess — and so the numbers in the bullet above are printed, not typed.

    Example:
        >>> apply_merge_advancing_by_one([7, 7, 7], 7, 7, 300)
        [300, 300, 7]
    """
    out, i, n = [], 0, len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 1                      # the bug: the second symbol is read a second time
        else:
            out.append(seq[i])
            i += 1
    return out


if __name__ == "__main__":
    print(f"{'input':>18}  {'correct':<22} advance-by-one")
    for _case in ([7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7, 7]):
        print(f"{str(_case):>18}  {str(apply_merge_py(_case, 7, 7, 300)):<22}"
              f" {apply_merge_advancing_by_one(_case, 7, 7, 300)}")
    print("\nThe surplus symbols on the right are the ones consumed twice. A run of n sevens")
    print("should collapse to about n/2 symbols; the bug leaves it barely shorter than n, so")
    print("the sequence stops shrinking and your final length parts company with Python's.")

# %% [markdown]
# ## 12. Self-check
#
# 1. Your `best_pair` returns the first maximum it meets while iterating the
#    `std::unordered_map`, and every test passes on your machine. What is actually wrong?
#    - (a) nothing — the standard fixes iteration order
#    - (b) it is slower than comparing packed keys
#    - (c) two runs, two standard-library versions or two corpus sizes can produce different
#          vocabularies from identical input, and the vocabulary is the artefact you ship
#    - (d) it only breaks when two pairs both have count zero
#
# 2. `apply_merge` on `[7, 7, 7, 7]`, merging `(7,7)` into `300`, must give:
#    - (a) `[300, 300]`  - (b) `[300, 7, 7]`  - (c) `[300, 300, 300, 7]`  - (d) `[300, 7]`
#
# 3. Your C++ beat Python by the factor you measured. Where did that come from?
#    - (a) a better algorithm — C++ avoids the rescan
#    - (b) the same `n_merges × n_symbols` work at a much lower cost per symbol: no boxed
#          integers, no tuple keys, no interpreter dispatch
#    - (c) multiple cores
#    - (d) the compiler recognises BPE and replaces it
#
# 4. Which is true of the library most published BPE vocabularies are trained with?
#    - (a) it is C++, which is why this lesson is in C++
#    - (b) it is pure Python, and slow
#    - (c) it is Rust, and this lesson's argument is about systems languages rather than C++
#    - (d) it is C++, but only in the training path
#
# 5. You double the corpus and halve the merge count. Roughly what happens to both runtimes?
#    - (a) both unchanged — the product is unchanged  - (b) both double  - (c) both halve
#    - (d) Python doubles, C++ is unchanged
#
# Answers, with reasoning, are published in the course solution bundle.

# %% [markdown]
# ## What you built, and where it goes next
#
# A tokenizer trainer in C++ whose merge list is provably identical to the Python one, a
# measurement of what the language was worth on your own machine, and the sharper lesson
# underneath it: the language bought a constant, and the algorithm is still the naive rescan.
# F02 picks this up where the vocabulary becomes an encoder and the rescan becomes incremental.

# %%
if __name__ == "__main__":
    # A final sweep of the correctness checks. The measurement in section 8 is deliberately
    # not repeated here: it is the slowest cell in the lesson and nothing below it changed.
    _check_count_pairs()
    _check_best_pair()
    _check_apply_merge()
    _check_train()
    print("\nall correctness checks green")
