# The lesson contract

Every lesson in this repository is something a student *does*, not something they read.
A lesson ships only when all 14 gates below are green. `tools/verify_all.py` machine-checks
what a machine can in gates 1-12 (its docstring lists each check); `tools/notebooks.py --check`
and `tools/verify_portable.py` check gates 13 and 14. What no tool can judge -- whether an
objective is measurable, whether a quote is on the live page, whether a number in the prose was
typed -- is checked by an independent reviewer and recorded in the lesson's `meta.yaml`.

## Pedagogy gates (the Coursera/Udacity shape)

1. **Objectives** — 3-5 measurable objectives in `meta.yaml`, each starting with a verb
   ("implement", "measure", "explain why"). No objective may be "understand X".
2. **Prerequisites** — explicit lesson ids. A student who has done the prerequisites can
   finish this lesson without reaching for anything else.
3. **Interleaving** — no more than 40 lines of prose (non-blank markdown lines, hint blocks
   included) before the next thing the student runs.
   Concept, then immediately a cell they execute or edit.
4. **Scaffolded exercises** — each exercise is a function stub with a docstring, a worked
   example in the docstring, `# YOUR CODE HERE`, and `raise NotImplementedError`.
   Never a blank cell.
5. **Instant feedback** — public `assert`-based checks in the notebook, runnable by the
   student, with failure messages that name the likely mistake, not just "assertion failed".
6. **Autograded** — `tests/test_lesson.py` defines a `RUBRIC` with points, hidden cases and
   a hint per test. Partial credit is mandatory; all-or-nothing grading is a bug.
7. **Solutions** — a complete reference implementation in `solutions/`, excluded from the
   student bundle by the build, never by `.gitignore` alone.
8. **Self-check** — 3+ multiple-choice questions probing the misconception the lesson exists
   to fix, answers in `solutions/`.

## Execution gates

9.  **Runs top to bottom, clean** — `python lesson.py` exits 0 in a fresh interpreter. No
    hidden state, no cell-order dependence, no network access unless declared.
10. **Budget honoured** — declared `tier` and `budget_seconds` in `meta.yaml`; the executor
    writes back *measured* wall time and peak RSS. `cpu8` means 8 GiB / 2 vCPU / <= 10 min.
    A lesson that exceeds its declared budget fails the gate; it does not get a bigger budget.
11. **Data is real and free** — every dataset named with its licence and a direct URL. No
    gated, non-commercial or registration-walled data on a required path.
12. **Claims sourced** — every factual claim about the world carries a primary source URL
    and an access date in `claims.yaml`. Numbers in prose are generated, never typed.

## Delivery gates (what a student actually meets)

Gates 1-12 are checked in this repository's environment, from this repository's directory. A
student meets neither. They click a badge, and a notebook service opens one file in an empty
directory, in a Python it chose. These two gates are about that moment.

13. **Opens anywhere** — `lesson.ipynb` is committed, generated from `lesson.py`, and never
    edited by hand (`tools/notebooks.py --check`). Its first two cells are the launcher that
    `tools/notebooks.py --inject` generates: Open in Colab / Kaggle / Binder / Codespaces
    badges, and a cell that installs only what the lesson requires and is missing — pinned to
    `requirements.txt` — and fetches only the sibling files it needs. Never hand-edit it; fix
    the generator. `tools/verify_portable.py` must pass: the notebook alone, in an empty
    directory, in a real kernel, in a minimal environment, on Python 3.11 and 3.12; and with
    `--completed`, the finished lesson must print the same results in both.
    A measurement of the student's machine — a time, a memory peak, a throughput — prints
    WITH ITS UNIT on the value itself (`0.134 s`, `1.45 MiB`, `418 MB/s`), never only in a
    column header. A number DERIVED from such a measurement — how many samples fit a control
    tick at the speed this machine ran — is itself a measurement and prints with `≈` in front of
    it. Both tell a student which numbers will differ on their machine, and they are how the
    comparison tells a measurement from a result: inputs and results are compared exactly,
    measurements are not.
14. **Run all is a good experience** — a student who opens the notebook and presses Run all
    before writing a line must reach the last cell with no unhandled exception:
    - every check and every demo that consumes an exercise's result runs through the lesson's
      `_try` guard, so an unfilled stub reports that it is not implemented yet, a wrong answer
      prints its check's hint and the notebook carries on, and a demo that needs an unfinished
      exercise names that exercise and skips;
    - each exercise offers progressive hints in collapsed `<details>` blocks — first what to
      think about, then the approach in words. A hint never contains code that solves the
      exercise, a value the rubric checks, or a path under `solutions/`;
    - the notebook ends with a progress board: one line per exercise marked ✅ passed,
      ❌ failed or ⏳ not started, then how many of how many are complete;
    - `sys.exit` / `raise SystemExit` happen only outside a notebook kernel. In a script or
      under CI, a failed check still ends the run non-zero; in a kernel it is a printed line,
      never a traceback at the foot of the page.
    `tools/verify_portable.py` (without `--completed`) is the check: every cell runs.

## Language policy

- **Python notebooks** are the default teaching surface (jupytext `py:percent` is the source
  of truth; `.ipynb` is generated, never committed by hand).
- **C and C++** carry the systems lessons, where the point is memory, cache, precision or an
  engine's real API. Built with `clang`/`make`, graded by a test binary.
- **C#** lessons are deferred until a .NET runner exists in CI. Authoring a C# exercise that
  has never been executed would violate gate 9.

## Known trap: compiled lessons and absolute library paths

A C/C++ lesson links against `libmujoco` inside the virtualenv, and the linker bakes that
ABSOLUTE path into the binary. Move the checkout, or rebuild the venv somewhere else, and the
binary still points at the old path: `dyld: Library not loaded`. `make` will not save you —
the binary is newer than its source, so it looks up to date.

`tools/execute.py` therefore deletes compiled artefacts before gating any `language: c` or
`language: cpp` lesson. A STUDENT who moves their checkout hits the same trap and has no such
guard, so every compiled lesson must tell them, in its "common mistakes" section, to run
`make clean` after moving or rebuilding. Authors: check this before marking a compiled lesson
`reviewed`.
