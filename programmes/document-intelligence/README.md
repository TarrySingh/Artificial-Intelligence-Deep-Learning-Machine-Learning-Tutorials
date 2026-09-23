# Document and contract intelligence for regulated operations

**Programme id:** `P02` · **Compute tier:** `cpu8` throughout (8 GiB, 2 vCPU, no GPU, no
network on any required path) · **Status:** generated in [`MODULES.md`](./MODULES.md).

This is the programme most enterprises actually buy. Not a chatbot, not a demo: a pipeline that
reads invoices, contracts, claims, KYC packs and remittance advices, pulls structured fields out
of them, and hands the uncertain ones to a person — under a budget, with an audit trail, in a
setting where being wrong has a named cost and sometimes a named regulator.

It is also the programme where most courses cheat. They teach you to call an extraction API and
eyeball the output. This one starts at the other end: **you build the measuring instrument
before you build the thing it measures**, because without it you cannot tell an improvement
from a relabelling, and you certainly cannot tell a regulator anything.

---

## Who this is for

Named job titles, because "AI practitioner" is not one:

- **Machine learning engineer, document AI / IDP** — owns the extraction models and the
  evaluation harness behind them.
- **Data scientist, operations or finance shared services** — accounts payable, claims intake,
  trade-document processing, onboarding.
- **ML platform / MLOps engineer** — owns the pipeline, the drift monitors and the cost per
  document.
- **Model risk analyst / model validator** (banking, insurance) — has to sign off on a system
  someone else built, and needs to know what a defensible evaluation looks like.
- **AI governance and compliance lead** — writes the human-oversight policy and has to show it
  is real, not decorative.
- **Annotation / data operations lead** — owns the labels the whole programme depends on, and
  the review queue this programme teaches you to size.
- **Solutions architect or forward-deployed engineer** at an IDP vendor — has to defend an
  accuracy number in front of a customer who will test it.

**Prerequisite track lessons:** `T00-L01-the-8gb-track` (the profiler and tier gate every
lesson here is measured by — it is where you learn that a declared number and a measured number
are different things). Comfortable Python and basic probability are assumed. No prior NLP.

---

## What a graduate can do

1. Write an evaluation harness for a field-extraction system: normalisation policy per field
   type, exact/normalised/fuzzy matching, per-field precision/recall/F1, and a confusion
   analysis that says which defect class dominates.
2. Explain, with numbers they generated, why a fuzzy matcher pointed at a money field is a
   payment incident rather than a convenience.
3. Reconstruct reading order and table structure from token geometry, and score table
   extraction on both content and structure.
4. Build and calibrate a contract-clause classifier, and decide where it should abstain.
5. Size a human review queue against a fixed budget, prove the routing policy beats random
   routing at the same budget, and report the quality bought per unit of cost.
6. Monitor a live document pipeline for drift and distinguish a real regression from noise.
7. Produce a cost per document that survives contact with a finance team, and an evidence pack
   that survives contact with a model validator.

---

## Why this programme, and why now

Everything in this section carries a primary source and an access date in
[`claims.yaml`](./claims.yaml). Nothing here is asserted without one.

**The document layer is being made machine-readable by law, on a published timetable.** The
European Commission's own eInvoicing guidance records that Germany's Growth Opportunities Act
made eInvoicing the default for domestic B2B invoices from 1 January 2025, with the obligation
to *issue* structured invoices reaching businesses above EUR 800,000 turnover on 1 January 2027
and everyone else on 1 January 2028. Structured formats do not make document intelligence
unnecessary — they raise the bar, because the mixed estate of structured feeds and scanned
legacy documents is exactly where extraction errors hide.

**Human oversight of high-risk AI is a design obligation, not a convention.** EU AI Act
Article 14(1) requires high-risk systems to be designed so that they can be effectively
overseen by natural persons while in use. The Digital Omnibus on AI, Regulation (EU) 2026/1744,
which entered into force on 27 July 2026, moved the application date for stand-alone Annex III
high-risk systems to **2 December 2027** — a deferral, not a repeal. Whether a given document
pipeline is high risk depends on what it decides; where it is, the review queue this programme
makes you build and measure is the mechanism the obligation asks for.

**Confidence-gated human review is already the documented design of shipping products.**
Microsoft's Azure AI Document Intelligence documentation tells operators, in as many words,
that confidence can be used to decide whether to accept a prediction automatically or flag it
for human review. Module 1 asks you to implement that policy and then measure what it buys.

**The extraction problem is not solved, and the residual errors cluster by field type.** A May
2026 comparison of frontier LLMs against domain-trained small models on structured contract
extraction reports performance strongest on short-text identifiers and weakest on currency
fields requiring normalisation or aggregation. A March 2026 benchmark of multi-agent
architectures for financial document processing scores systems on five axes — field-level F1,
document-level accuracy, end-to-end latency, cost per document and token efficiency — and finds
the most accurate architecture costing 2.3 times the sequential baseline. A July 2026 benchmark
of schema-guided enterprise extraction finds commercial vision-language models performing well
on short documents but truncating record lists on long ones, with coding agents keeping higher
accuracy at much higher cost. Accuracy and cost are the same conversation, and both of them are
measurements.

**The labour market for people who can do this is growing.** The US Bureau of Labor Statistics
projects employment of data scientists to grow 35% from 2025 to 2035, "much faster than the
average for all occupations" (page last modified 27 August 2026). That is the closest thing to
a primary statistic for this population; it is a claim about data scientists, not about
document-AI engineers specifically, and this README does not stretch it further.

---

## The environment, and the constraint that shaped the curriculum

Every lesson runs on a laptop CPU in under ten minutes and under 8 GiB, on Python 3.12 with
**numpy, matplotlib, tokenizers and pytest, and nothing else** — plus `clang` and `make` for the
one systems module. There is no PDF library, no OCR engine and no model API on any required
path.

That is a real constraint and the curriculum treats it as one. Corpora are generated
deterministically inside each lesson from a fixed seed, so every student sees the same
documents, the same error mix and therefore the same numbers, and every number a student reads
was computed by code they ran. Where a real system would call a vision-language model, these
lessons hand you a stand-in extractor with a documented, seeded error model — because the
skill being taught is *judging* an extractor, and that skill transfers to whatever you put
behind the harness at work.

---

## What is built

[`MODULES.md`](./MODULES.md) is this programme's map. Its status table is generated from the
lessons themselves by `tools/status.py`: a module reads **built** only when its lesson exists,
passes all 14 gates in `QUALITY.md` and has been independently reviewed, and every other row
says `specified`. No count is typed here, so none can go stale.

It is also the full map: every module, the lab a student does in it, and its compute tier.
