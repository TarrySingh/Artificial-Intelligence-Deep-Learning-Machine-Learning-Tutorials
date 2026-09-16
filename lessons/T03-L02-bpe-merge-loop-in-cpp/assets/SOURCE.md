# Where `corpus.txt` comes from

`corpus.txt` is a concatenation of twelve modules of the CPython 3.12 standard library, in the
fixed order listed in `make_corpus.py`, truncated at a whitespace byte so the cut never lands
inside a word.

| | |
|---|---|
| Source | CPython standard library, `Lib/` (the copy installed on the machine that built it) |
| Upstream URL | https://github.com/python/cpython/tree/3.12/Lib |
| A file, directly | https://raw.githubusercontent.com/python/cpython/3.12/Lib/argparse.py |
| Licence | Python Software Foundation License Version 2 — https://docs.python.org/3/license.html |
| Gated | No. No registration, no data-use agreement, no non-commercial clause. |
| Retrieved | 2026-09-16, from the local install — nothing was downloaded |
| Built from | CPython 3.12.13, `sysconfig.get_paths()["stdlib"]` |
| Size | 239,997 bytes |
| sha256 | `2560537a8c746bfb81bbcaa7f65e1d029a93fcc619c654370d150111982b98aa` |
| Regenerate | `python assets/make_corpus.py` (the defaults are the shipped ones) |

Check the shipped bytes against that digest with `shasum -a 256 assets/corpus.txt`. The
generator prints the same digest, so a regenerated corpus that disagrees means the local
interpreter's `Lib/` differs from 3.12.13 and the merge lists in this lesson will shift
with it — reach for the shipped file, not the regenerated one, if the two ever part.

Two reasons this is the corpus:

1. **No download, ever.** The bytes ship inside the lesson, and the generator reads a local
   Python install, so the lesson runs with the network unplugged and every student trains on
   an identical, byte-for-byte file. A merge list is only a testable object if the input is.
2. **Source code has structure worth merging.** Long repeated identifiers, four-space indents
   and `self.` everywhere give byte-pair merges something real to find in the first few dozen
   steps, which is what makes the printed merge table legible instead of random.

The trainers in this lesson drop whitespace during pre-tokenisation, so the indentation is not
itself merged — it only decides where word boundaries fall.
