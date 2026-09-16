# The lesson contract

Every lesson in this repository is something a student *does*, not something they read.
A lesson ships only when all 12 gates below are green. Gates 1-8 are machine-checked by
`tools/execute.py` and `tools/grade.py`; 9-12 are human-reviewed and recorded in `meta.yaml`.

## Pedagogy gates (the Coursera/Udacity shape)

1. **Objectives** — 3-5 measurable objectives in `meta.yaml`, each starting with a verb
   ("implement", "measure", "explain why"). No objective may be "understand X".
2. **Prerequisites** — explicit lesson ids. A student who has done the prerequisites can
   finish this lesson without reaching for anything else.
3. **Interleaving** — no more than ~40 lines of prose before the next thing the student runs.
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
