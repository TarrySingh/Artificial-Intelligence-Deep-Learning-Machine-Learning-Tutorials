# Model risk, AI assurance and audit analytics

**The programme for the people who have to sign.**

Somebody puts their name on the sentence "this model is fit for its intended use". This
programme is the analytics that make that sentence defensible — built by hand, in numpy and C,
on a laptop, with every figure computed by code the student runs and every document generated
from those figures rather than typed alongside them.

It is not a governance course. There are no policy templates here and no maturity models. Every
module is a notebook or a compiled exercise that produces evidence: a number, a table, a
generated report, a machine-checkable trace from a claim back to the run that supports it.

---

## Why this programme, and why now

Three things changed, and all three are dated.

**The foundational US guidance was rewritten in 2026.** On 17 April 2026 the Federal Reserve,
the OCC and the FDIC jointly issued revised model risk management guidance — SR 26-2 at the
Fed, Bulletin 2026-13 at the OCC. It supersedes SR 11-7, which had stood since 4 April 2011,
and SR 21-8 from 2021. The agencies say they updated it to "clarify model risk management
principles and to emphasize a risk-based approach to model risk management that is tailored to
a banking organization's model risk profile and the size and complexity of its operations".
The Fed expects it to be most relevant to banking organisations over 30 billion dollars in
total assets. The OCC's version names model validation and monitoring — including validating
conceptual soundness and outcomes analyses — among the practices it covers, and that pairing is
the spine of the module map below.

The same guidance puts generative and agentic AI models expressly outside its own scope:
"Generative AI and agentic AI models are novel and rapidly evolving. As such, they are not
within the scope of this guidance." That is a gap, not an exemption, and the assurance modules
in this programme are built for it rather than pretending the banking guidance already covers
it.

**A second regime arrived on a different clock.** The EU AI Act entered into force on 1 August
2024 and became applicable on 2 August 2026. After the AI Omnibus, the obligations for high-risk
use cases in the sensitive areas of Annex III apply from 2 December 2027, and Annex I systems
embedded in regulated products from 2 August 2028. Credit scoring and creditworthiness
assessment of natural persons sit in Annex III. A European bank's credit models are therefore
inside a prudential model-risk regime *and* a product-safety-style AI regime, with different
deadlines and different evidence expectations. Somebody has to produce both sets of evidence
from one set of runs.

**Assurance became a thing you certify, not just a thing you claim.** ISO/IEC 42001, the AI
management system standard, is now held by named companies under accredited certification —
Amazon Web Services announced on 25 November 2024 that it was the first major cloud provider to
hold it for AI services, covering Bedrock, Q Business, Textract and Transcribe. The practical
consequence for a validator is that the evidence you produce internally is increasingly the same
evidence an external auditor asks to see.

In the UK the direction was set earlier: the PRA's SS1/23, published 17 May 2023 and effective
17 May 2024, sets out five principles, and the *first* of them is "Model identification and
model risk classification" — inventory and tiering, before any metric. This programme's module
order follows that.

Every claim in the four paragraphs above is sourced, with a primary URL and an access date, in
[`claims.yaml`](claims.yaml).

---

## Who this is for

Name the job titles, because vagueness here wastes people's time.

- **Model validators** and **model validation analysts** in a second line of defence — the ones
  who have to reproduce a first-line team's results and then say something about them.
- **Model risk managers** and **heads of model risk**, who own the inventory, the tiering and
  the aggregate model risk position.
- **Quantitative analysts in model risk management**, including the credit, market and
  counterparty-credit validation desks.
- **Internal auditors** covering models and AI, and **IT audit** analysts who are being handed
  AI systems and asked to form a view.
- **AI governance** and **responsible AI leads** who need to turn a framework into artefacts.
- **AI assurance** and **conformity assessment** practitioners working to the EU AI Act, ISO/IEC
  42001 or a client's assurance scheme.
- **Data scientists and ML engineers in regulated firms** who keep getting validation findings
  back and would like to know what the other side is actually running.
- **Regulatory examiners and supervisors** who want to run the analytics themselves rather than
  read about them.

For scale rather than for promise: the US Bureau of Labor Statistics records that financial risk
specialists — the occupation code most US bank model-risk roles sit under — held about 66,000
jobs in 2025, and projects overall employment of financial analysts to grow 7 percent from 2025
to 2035, much faster than the average for all occupations. The median annual wage for financial
and investment analysts was 102,740 dollars in May 2025. That is an occupational projection, not
a claim about this programme. No salary figure from a job board appears anywhere in this
programme, and `claims.yaml` records why.

---

## What a graduate can do

Concretely, and demonstrably, because each one is an artefact they built:

1. **Measure whether a model means what it says.** Build a reliability table and a
   support-weighted expected calibration error, and explain why a model can rank beautifully
   and still be indefensible at a decision threshold.
2. **Detect and quantify population drift** against a baseline that cannot move, with per-bin
   contributions so a breach can be traced to the bin that caused it — and recognise the
   monitoring pack whose bins are re-cut each month and therefore can never report anything.
3. **Report subgroup performance under a minimum-support rule** that suppresses metrics without
   hiding groups, and defend both halves of that rule.
4. **Stand up a model inventory and a defensible tiering**, and reconcile it against what is
   actually running in production.
5. **Review conceptual soundness mechanically** — assumptions tested, specification diffed
   against implementation, variables checked against their stated purpose.
6. **Build a challenger from scratch** and run a paired comparison with a significance test
   rather than a leaderboard.
7. **Produce explainability evidence that reproduces** — same seed, same inputs, same
   attributions, with the reproduction itself checked by code.
8. **Design monitoring thresholds with a measured false-alarm rate**, instead of adopting a
   number because it appeared in a vendor deck.
9. **Generate the validation report and the committee pack from the results**, with a gate that
   fails the document if any figure in it does not trace back to a computed result.

What a graduate cannot do, and this programme will not claim: sign anything. Signing is a role,
not a skill, and the judgement it needs is not teachable in ten notebooks.

---

## Prerequisites

- **[`lessons/T00-L01-the-8gb-track`](../../lessons/T00-L01-the-8gb-track)** — the profiler and
  the tier gate. Required. Every module here declares a compute budget, and this is the lesson
  that teaches what a declared budget means and how it is measured. It is also the house
  introduction to how these lessons are shaped: stubs, public checks, an autograded rubric.
- **Python and numpy**, at the level of writing an indexing expression without looking it up.
  No pandas, no scikit-learn, no statsmodels: every estimator in this programme is built.
- **No finance background is assumed.** Where a lesson needs a credit, market or capital
  concept, it defines it at the point of use and in a sentence.
- **No prior model risk experience is assumed** either, but the programme moves quickly and
  assumes the reader has met a model that turned out to be wrong.

Two modules are compiled C rather than Python (marked in
[`MODULES.md`](MODULES.md)); they need `clang` and `make`, and nothing else.

---

## Compute tier

**Every module in this programme is `cpu8`: 8 GiB of RAM, 2 vCPU, no GPU, no network, under ten
minutes.** That is not a limitation the programme apologises for — it is a design constraint
that matches the work. Validation analytics run on extracts, not on training clusters, and an
examiner should be able to re-run your evidence on the machine in front of them.

The opening lesson measures well under a second and under 40 MiB. The exact figures are
written into its `meta.yaml` by `tools/execute.py` from an actual run, never typed by its
author, which is the point of the prerequisite lesson. The per-module budgets are in
[`MODULES.md`](MODULES.md), and the gate that enforces them is in `QUALITY.md` at the repository
root.

---

## Status

[`MODULES.md`](MODULES.md) is this programme's map. Its status table is generated from the
lessons themselves by `tools/status.py`: a module reads **built** only when its lesson exists,
passes all 14 gates in `QUALITY.md` and has been independently reviewed, and every other row
says `specified`. No count is typed here, so none can go stale.

```
programmes/model-risk/
├── README.md        this file
├── MODULES.md       the module map, with its generated status table
├── claims.yaml      every claim above, with a primary source
└── lessons/         one directory per built module
```

Run a lesson's gates from the repository root, for example the opening one:

```
python tools/execute.py programmes/model-risk/lessons/P04-L01-validation-suite --write-back
python tools/grade.py  programmes/model-risk/lessons/P04-L01-validation-suite --solution
python tools/grade.py  programmes/model-risk/lessons/P04-L01-validation-suite
```
