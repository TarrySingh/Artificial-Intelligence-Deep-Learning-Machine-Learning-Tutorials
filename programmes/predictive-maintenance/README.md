# P03 · Industrial predictive maintenance: sensor physics to alarm economics

The programme that makes a data scientist useful on a plant floor.

Most predictive-maintenance teaching stops at a model with a good score on a held-out split.
A plant does not buy scores. It buys a decision — *pull this machine out on Thursday* — and
that decision is made by a threshold, a lead time and three prices, none of which a model
knows anything about. This programme is built backwards from the decision: the first thing you
do is price a confusion matrix, and only then go back and earn the feature that fills it in.

Every module is something you **do**: a notebook you run and fill in, or a C program you
compile. No module is only reading.

## Who it is for

By job title, the people this is written for:

- **Data scientists and ML engineers** who have been handed a pile of sensor history and a
  brief that says "predict failures", and who have worked out that the brief is under-specified
  but not yet what it is missing.
- **Reliability engineers** and **condition-monitoring analysts** who already set thresholds
  from experience and want to be able to defend one in a currency the plant manager uses.
- **Maintenance planners** and **maintenance engineers** who own the consequence of a false
  alarm and want to see it priced.
- **Controls and OT engineers** who will be asked to put a model somewhere on a network that
  was not designed for it, and who will be the ones saying no if the answer has to be no.
- **Asset managers** deciding whether a monitoring programme pays for itself.

You need Python to the level of writing a function and reading a traceback, numpy indexing,
and the idea of a confusion matrix. You do not need a machine-learning background: the opening
module's detector is a threshold on a single scalar, on purpose, so that no property of a model
can be blamed for the result it produces.

## Why this programme exists

Three pieces of evidence, each sourced in [`claims.yaml`](claims.yaml) with a URL and the date
it was read.

**The people are being hired.** The US Bureau of Labor Statistics projects employment of
industrial machinery mechanics, machinery maintenance workers and millwrights to grow 14
percent from 2025 to 2035 — "much faster than the average for all occupations" — with about
51,900 openings a year on average over the decade, from a 2025 base of 547,300 jobs. These are
the colleagues who will act on your alarms, and there are going to be more of them, not fewer.

**There is a regulatory clock.** Regulation (EU) 2023/1230 on machinery applies from
**20 January 2027**, replacing the Machinery Directive. Anything fitted to machinery sold into
the EU now has a documentation deadline attached to it. What the Regulation requires of a
monitoring system is a conformity question and belongs to track T10's conformity lesson, not
to this paragraph; the date is what matters here, and the date is fixed.

**Operators run this in production, and say so.** Siemens and Deutsche Bahn announced a
predictive maintenance pilot in 2016 in which "data from the vehicles will be received and
analyzed in a central diagnostics system to calculate failure predictions", with specialists
turning those predictions into instructions for technicians in DB workshops. A decade later
DB's own page on artificial intelligence states that "DB's goal here is the wholesale
implementation of condition-based maintenance for trains". That is the pipeline these modules
build, named and dated at one operator.

You will not find a market-size figure in this README. No primary source for one could be
verified, so none is quoted.

## What a graduate can do

- Build a **causal** degradation feature from raw sensor bursts — one that can be computed at
  hour *t* from data that exists at hour *t*, which rules out most of what is convenient
  offline — and say what is in it and what it is blind to.
- Turn a health index into an alarm decision under a **lead-time requirement**, so that an
  alarm which arrives too late to act on is counted as the missed failure it is, not as a
  prediction that came true.
- Sweep thresholds, build ROC and precision/recall curves, and explain why neither curve can
  choose a point on itself.
- Write an **expected-cost function** over planned interventions, unplanned failures and false
  alarms, find the cost-optimal threshold, and show how far it sits from the accuracy-optimal
  one on the same counts.
- Say *which* failures a chosen threshold deliberately gives up, and why paying to catch them
  would cost more than they do.
- Defend a threshold to a plant manager by handing over the three prices it came from, and
  re-derive it when the prices change — because the false-alarm budget is not infinite. The
  UK Health and Safety Executive's alarm-handling guidance, written after the 1994 Texaco
  Milford Haven explosion, records two operators facing 275 alarms in the last eleven minutes
  before it, and sets a normal-operation target of no more than one alarm every ten minutes.
  That is the ceiling a threshold is really being set against.

## The modules

The full map, with the lab in each one, is in [`MODULES.md`](MODULES.md). **One module is
built**: module 1, `lessons/P03-L01-alarm-economics`. The other eight are specified there for
later waves and do not exist as code. `MODULES.md` marks every one of them.

## Prerequisite track lessons

| lesson | why |
|---|---|
| `lessons/T00-L01-the-8gb-track` | **Required.** You build `measure()` and `tier_check()`, the profiler and the compute gate this whole repository is held to. Every module below declares a tier and is measured against it by `tools/execute.py`; T00-L01 is where you learn what those numbers mean and why a lesson that busts its budget gets rewritten rather than re-declared. |
| `lessons/T10-L01-ai-act-conformity-pack` | Recommended before module 8. Deployment on an OT network is where the documentation obligations stop being somebody else's problem. |

Nothing else in Synapsa Commons is assumed. No lesson here trains a neural network, and no lesson
here needs a GPU.

## Compute tier

**Every module in this programme is tier `cpu8`: at most 8 GiB of RAM, no GPU, and under ten
minutes of wall clock.** Nothing downloads on a required path. Nothing needs an API key.

The built module measures well inside that:

| | declared | measured by `tools/execute.py` |
|---|---|---|
| `P03-L01-alarm-economics` | `cpu8`, 90 s | 0.6-0.7 s, ~190 MiB |

Those figures are written back into `meta.yaml` by the execution gate on every run, not typed
by an author, which is why they wobble in the last digit between runs and the table gives a
range. The authoritative pair is whatever is in that module's `meta.yaml` right now.
The synthetic fleet the opening module generates is 300 machines x 360 hours x 64 samples —
53 MiB of float64 — which is deliberately large enough that a careless feature implementation
is noticeably slow and small enough that the whole notebook runs in under a second.

## Running any module

```bash
# from the repository root

# the execution gate: runs the reference solution, measures wall time and peak RSS
.venv/bin/python tools/execute.py programmes/predictive-maintenance/lessons/<id> --write-back

# the autograder, against your own work
.venv/bin/python tools/grade.py programmes/predictive-maintenance/lessons/<id>

# regenerate the student notebook from lesson.py (never run jupytext directly: its random
# cell ids make the committed notebook look stale to tools/notebooks.py --check)
.venv/bin/python tools/notebooks.py --build programmes/predictive-maintenance/lessons/<id>
```

`lesson.py` is the source of truth; the `.ipynb` is generated and never hand-edited.
`solutions/` and `tests/` are excluded from the student bundle by the build, not by
`.gitignore`.

## A note on data

The opening module's fleet is **synthetic**, declared as such in its `meta.yaml`, with the
generator in the notebook where the student reads it before running it. That is a deliberate
choice and not a shortcut: grading threshold logic needs ground truth a student can see —
known failure hours, known defect onsets, a controlled base rate — so that a bug in their own
lead-time accounting is distinguishable from a quirk of the data.

Later modules move onto real run-to-failure data. NASA's Prognostics Center of Excellence
maintains a free Prognostics Data Repository, including the Turbofan Engine Degradation
Simulation sets produced with C-MAPSS; `MODULES.md` names it against module 5. Every dataset a
module uses will carry its licence and a direct URL before that module ships, as gate 11 of
[`QUALITY.md`](../../QUALITY.md) requires. A module whose dataset cannot be named that way does
not get built.
