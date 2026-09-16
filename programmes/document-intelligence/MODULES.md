# P02 · Module map — Document and contract intelligence for regulated operations

Eleven modules. **Five are built** — 1 to 4 and 10. The other six are specified here and have no
directory; if you go looking for `lessons/P02-L05-*` you will not find it, and that is
deliberate — this file does not pretend unbuilt work exists.

Every module is a lab a student *does*: a notebook they run and fill in, or a C exercise they
compile. Every module declares a compute tier and is held to it by `tools/execute.py`.

**Environment constraint, stated once.** The whole programme runs on Python 3.12 with numpy,
matplotlib, tokenizers and pytest, plus `clang` and `make`. There is no PDF library, no OCR
engine and no model API on any required path, so every corpus is generated deterministically
inside its lesson from a fixed seed. The labs below are designed within that, not around it.

| # | Status | Lesson id | Tier | Measured · rubric |
|---|---|---|---|---|
| 1 | **BUILT** | `P02-L01-extraction-evaluation` | `cpu8` | 0.2 s · 29 MiB · 84 pts / 21 cases |
| 2 | **BUILT** | `P02-L02-layout-reading-order` | `cpu8` | 0.2 s · 31 MiB · 94 pts / 23 cases |
| 3 | **BUILT** | `P02-L03-sequence-labelling` | `cpu8` | 0.7 s · 34 MiB · 90 pts / 20 cases |
| 4 | **BUILT** | `P02-L04-table-extraction` | `cpu8` | 7.3 s · 30 MiB · 94 pts / 21 cases |
| 5-9 | specified | — | `cpu8` | — |
| 10 | **BUILT** | `P02-L10-streaming-scanner-in-c` | `cpu8`, language **C** | 4.3 s · 65 MiB · 39 pts / 12 cases |
| 11 | specified | — | `cpu8` | — |

---

## Module 1 — Field extraction you can measure: build the evaluation harness before the extractor

**Status: BUILT.** `lessons/P02-L01-extraction-evaluation/` · tier `cpu8` · measured 0.2 s,
29 MiB · 84 rubric points across 21 autograded cases · prerequisite `T00-L01-the-8gb-track`.

**The lab.** The student is handed 180 short synthetic remittance advices with gold labels, and
a deliberately mediocre stand-in extractor with a seeded, documented error model — it reformats
dates and amounts into its own house style, drops fields, invents payment terms that were never
printed, smudges a supplier name the way a scan does, and substitutes a digit in an amount. They
then implement, in pure numpy and stdlib:

1. `normalise_value` — a per-field-type normalisation policy (money, date, id, integer, text),
   including the trap that a comma is a thousands separator in one convention and a decimal
   point in another.
2. `match_value` — exact, normalised and fuzzy matching, with fuzzy confined to text fields.
   They then run a cell that counts how many genuinely wrong amounts a fuzzy money matcher
   would wave through on this corpus.
3. `score_field` and `macro_f1` — per-field precision, recall and F1, where a wrong value counts
   as both a false positive and a false negative, and an unweighted macro average that refuses
   to let the field on every page bury the rare one.
4. `classify_cell` and `confusion_by_field_type` — five mutually exclusive labels per cell
   (correct, miss, spurious, wrong value, true negative), aggregated by field *type*, so one F1
   number becomes a work order.
5. `review_queue` and `apply_reviews` — a confidence-ordered routing policy under a fixed
   budget, with a deterministic tie-break, that must not mutate the records it scores.

They finish by sweeping the budget and reading a quality/cost table: macro F1 against a random
routing control, cost per document from placeholder rates they are told to replace, and the
marginal F1 bought per unit of cost. On this corpus the yield *rises* then falls, because the
lowest-confidence cells are all misses and the next band contains the wrong values — and fixing
a wrong value is worth strictly more than fixing a miss. That is measured in the notebook, not
asserted.

**Why this is module 1 rather than module 3.** Published evidence puts the residual difficulty
of contract extraction in exactly the fields normalisation governs (see `claims.yaml`). A
student who has not written the normaliser cannot tell a formatting difference from a defect,
and will spend the rest of the programme filing bugs against models that were already right.

---

## Module 2 — Ingestion and layout without a PDF library

**Status: BUILT.** `lessons/P02-L02-layout-reading-order/` · tier `cpu8` · measured 0.2 s,
31 MiB · 94 rubric points across 23 autograded cases.

**Lab:** reading order from geometry. The student is given synthetic pages as
token boxes (text, x, y, width, height, page) covering single-column, two-column and
table-bearing layouts. They implement recursive XY-cut segmentation, a reading-order sort within
each block, and a header/footer detector; then score their reading order against the gold
sequence with Kendall's tau and a block-level boundary F1, using the harness from module 1. They
finish by measuring how much field-extraction F1 moves when reading order is wrong, which is the
argument for caring about layout at all.

## Module 3 — From rules to a tagger: sequence labelling over document tokens

**Status: BUILT.** `lessons/P02-L03-sequence-labelling/` · tier `cpu8` · measured 0.7 s,
34 MiB · 90 rubric points across 20 autograded cases.

**Lab:** implement an averaged structured perceptron in numpy over BIO tags on
the module 2 token stream, with hand-built features (shape, prefix, neighbouring token,
line position). Train it, then score it with module 1's harness — same scorer, different
extractor, which is the whole point. They compare against the rule-based baseline on the same
axes and identify which field types the tagger actually improved.

## Module 4 — Table extraction: structure and content are two different scores

**Status: BUILT.** `lessons/P02-L04-table-extraction/` · tier `cpu8` · measured 7.3 s,
30 MiB · 94 rubric points across 21 autograded cases.

**Lab:** reconstruct rows and columns from token geometry by projection
profiling and clustering, handle a spanning header and a row split across a page break, then
implement two scorers — cell-content F1 and a structure score in the spirit of TEDS — and show a
case where content is nearly perfect while structure is wrong, and a case where the reverse is
true. Ends with a rule for which of the two your downstream consumer actually needs.

## Module 5 — Contract clause classification and calibrated abstention *(specified)*

**Tier `cpu8`.** **Lab:** using the `tokenizers` package to build a subword vocabulary over a
synthetic clause corpus (governing law, limitation of liability, termination for convenience,
assignment, indemnity, change of control), train a multinomial logistic regression in numpy,
then calibrate its probabilities and implement an abstention rule. They measure per-clause F1,
the coverage/accuracy curve, and what abstention costs in recall — feeding the review queue from
module 1.

## Module 6 — Human-in-the-loop review queues that survive contact with humans *(specified)*

**Tier `cpu8`.** **Lab:** module 1's reviewer was perfect. This one is not. The student
simulates reviewers with disagreement and fatigue, measures inter-annotator agreement (raw
agreement and Cohen's kappa), implements a two-tier escalation policy and an SLA-aware queue,
and measures throughput against quality. Ends with the uncomfortable measurement: the ceiling
your harness can see is your annotation agreement, not 1.0.

## Module 7 — Error analysis and slice discovery: is that regression real? *(specified)*

**Tier `cpu8`.** **Lab:** implement automatic slice finding over document attributes (vendor,
layout family, language, page count, scan quality), bootstrap confidence intervals on per-slice
F1, and apply a multiple-comparison correction. The student is given two model versions and must
decide, with intervals rather than point estimates, which apparent regressions are real. They
also compute the sample size a slice needs before it is allowed to block a release.

## Module 8 — Drift monitoring for a live document pipeline *(specified)*

**Tier `cpu8`.** **Lab:** build the monitor. Population stability index and KL divergence over
field-value and confidence distributions, a CUSUM chart on a daily quality proxy, and an alarm
threshold chosen against an explicit false-alarm budget rather than a round number. The student
replays a synthetic year containing a vendor template change, a seasonal mix shift and one real
model regression, and must catch the regression without paging anyone for the other two.

## Module 9 — Cost per document, and the routing budget that sets it *(specified)*

**Tier `cpu8`.** **Lab:** a three-tier router — cheap extractor, expensive extractor, human —
with per-tier cost and accuracy measured from the student's own earlier modules. They formulate
the budget allocation, solve it by sweep and then by a greedy marginal-yield rule, plot the
Pareto frontier of quality against cost per document, and write the one-paragraph
recommendation a finance partner would sign.

## Module 10 — The scanner in C: fixed memory over an unbounded export

**Status: BUILT.** `lessons/P02-L10-streaming-scanner-in-c/` · tier `cpu8`, language **C** ·
measured 4.3 s, 65 MiB · 39 rubric points across 12 autograded cases.

**Lab:** implement a streaming field scanner over a multi-gigabyte
synthetic document export in C, in constant memory, with explicit handling of quoted fields,
embedded newlines and a truncated final record; graded by a test binary built with `clang` and
`make`. They measure throughput and peak RSS against the Python version from module 1 and learn
where the factor comes from. Per `QUALITY.md`, this module's common-mistakes section must tell
students to run `make clean` after moving or rebuilding their checkout.

## Module 11 — Capstone: the conformity pack *(specified)*

**Tier `cpu8`.** **Lab:** assemble the harness, the slice analysis, the drift monitor, the
review policy and the cost model into a single evidence pack for one document pipeline: what it
does, how it was measured, on what data, with what residual error by class, what oversight it
has, and what it costs. Graded on whether an independent validator could reproduce every number
from the artefacts supplied — which is the only definition of "documented" that means anything.

---

## Build order for later waves

Modules 2, 3 and 4 next: they are the extractor side, and they are the ones module 1's harness
was built to judge. Then 6 and 7 (the human and statistical honesty pair), then 8 and 9 (the
operations pair), then 10, then the capstone. Modules 5 and 10 are the two with real
implementation risk inside this environment's package set and should be prototyped before they
are promised to anyone.
