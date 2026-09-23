# EU AI Act conformity engineering

**The programme that turns a regulation into running code.**

Every module is something you *do*: a notebook you run and fill in, with scaffolded stubs, an
autograded rubric with partial credit, and a worked solution. Nothing here is a lecture about
compliance. You leave each module holding an artefact — a checker, a log, a document
generator, a monitoring feed — that a reviewer can run against a real system.

> **This is engineering, not legal advice.** The programme teaches the data structures,
> checks and evidence trails a regulation implies. It is not a substitute for the Official
> Journal text or for professional advice on a specific system. Every module says so in its
> opening cell and again at the end.

---

## Who this is for

Named by the job titles that actually do this work:

- **AI/ML engineers and MLOps engineers** on a system that is, or is about to be, classified
  high-risk — the people who have to make the logging, the versioning and the evidence exist
  in the codebase rather than in a policy document.
- **AI governance leads, AI compliance managers and model risk managers** who own the
  evidence pack and are tired of receiving spreadsheets they cannot verify.
- **Data protection officers and privacy engineers** extending a GDPR practice into AI, where
  a retention floor and an erasure duty can point in opposite directions.
- **Product and engineering managers** who have to schedule conformity work against a release
  plan and need to know what it actually costs in engineering days.
- **Internal auditors and assurance staff** who will be asked to test these controls and need
  to have built one before they inspect one.

The programme assumes you can write Python: dicts, lists, comprehensions, `datetime`. It does
not assume you have read a regulation before. Each module quotes what it relies on and sources
it.

## What a graduate can do

1. Classify a system, and say what applies to it, from when, and on which route.
2. Build an append-only, tamper-evident event log, prove its integrity against a published
   anchor, and reconstruct a single inference decision end to end from it.
3. Generate Annex IV technical documentation from the system's own artefacts, so it cannot
   silently drift out of date.
4. Test a data-governance claim rather than assert one.
5. Instrument human oversight so that "a human reviewed it" is a measurable property.
6. Produce accuracy, robustness and cybersecurity evidence that states its own uncertainty.
7. Stand up a post-market monitoring feed and a serious-incident path off the same log.
8. Pick and justify a conformity assessment route, and assemble the declaration pack.

And, across all of it: tell the difference between a control that works and a control that
merely reports that it works.

## Why now

The AI Act entered into force on 1 August 2024 and became applicable on 2 August 2026. The
Digital Omnibus on AI, Regulation (EU) 2026/1744, entered into force on 27 July 2026 and moved
the Chapter III high-risk duties — risk management, data governance, technical documentation,
record-keeping, human oversight, accuracy and robustness — to **2 December 2027** for Annex III
stand-alone high-risk systems and **2 August 2028** for AI embedded in products covered by
Annex I product-safety law.

That is the whole window. It is long enough to engineer these controls properly and far too
short to retrofit them, because most of them — logging in particular — have to be designed
into the system rather than bolted on. Article 50's transparency duties, meanwhile, already
applied from 2 August 2026: "high-risk was deferred" has never meant "the AI Act was deferred".

On what is at stake, the programme is careful about a number that is usually quoted wrongly.
The headline EUR 35 000 000 or 7 % of worldwide turnover ceiling in Article 99 is for
**Article 5 prohibited practices**. The obligations this programme builds — the provider and
deployer duties in Articles 16 and 26, and Article 50 — sit under the **EUR 15 000 000 or 3 %**
ceiling. Quoting the larger figure for a logging defect is the kind of error this programme
exists to stop.

## Who is hiring, and for what

Two named studies, with their dates and sample sizes, because a market claim without a
denominator is not evidence:

- The **IAPP AI Governance Profession Report 2025** (published 16 April 2025) found 77 % of
  surveyed organisations working on AI governance, rising to near 90 % among those already
  using AI. Of 671 respondents, **10 — 1.5 % — said they would not need additional AI
  governance staff in the next 12 months**. Finding qualified people was named as part of the
  challenge by 23.5 %.
- **Axipro Technology's EU AI Act governance hiring-gap study** analysed 3,519 AI-related job
  postings across eight EU countries — Belgium, France, Germany, Ireland, Italy, the
  Netherlands, Spain and Sweden — over a 30-day window in mid-2026. It counted **3,004 builder
  roles against 446 governance roles**, a spread running from 16:1 in Sweden to 3.5:1 in
  Ireland. This is a vendor study, not a statistical office, and it is labelled as one.

The second study also carries the finding that matters most for a job search, and it is not
the ratio. The EU AI Act is named in **7.6 % of all AI postings**, and in **fewer than three in
ten even of the governance roles**. The work is being hired for under adjacent titles — model
risk, model validation, AI assurance, responsible AI, data protection engineering, ML platform
— rather than against the Act by name. Search for the work, not for the statute.

Both sources, their quotes and their access dates are in `claims.yaml` at this directory. The
figures deliberately excluded from this README, and why, are recorded there too.

## Prerequisite track lessons

- **`lessons/T10-L01-ai-act-conformity-pack`** — required before module 1 of this programme.
  It builds the evidence table this programme's modules attach to: risk classification, the
  cumulative obligation set, the per-system dates, and the rule that a "planned" row is not
  evidence. Every module here writes into an evidence id that lesson defines.
- **`lessons/T00-L01-the-8gb-track`** — recommended, not required. It builds the profiler and
  the tier gate that decide whether any lesson in Synapsa Commons may ship, so it explains what the
  compute tier below actually promises you.

## Compute tier

**`cpu8`** for every module: 8 GiB of RAM, 2 vCPU, no GPU, no network, no download. Each lesson
declares a `budget_seconds` and the repository's executor writes back the wall time and peak
RSS it measured, so the tier is a checked promise rather than an aspiration. The built module
measures well under its budget on a laptop CPU.

Dependencies across the whole programme are the Python standard library plus numpy. That is
deliberate: a compliance control you cannot run without a vendor's SDK is a control you cannot
audit, and a lesson that needs a package the student cannot install is a lesson they never
finish.

## Status

[`MODULES.md`](MODULES.md) is this programme's map. Its status table is generated from the
lessons themselves by `tools/status.py`: a module reads **built** only when its lesson exists,
passes all 14 gates in `QUALITY.md` and has been independently reviewed, and every other row
says `specified`. No count is typed here, so none can go stale.
