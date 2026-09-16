# Asset provenance

## `systems.yaml`

| | |
|---|---|
| Origin | Authored for this lesson. Not a third-party dataset. |
| Licence | CC0-1.0 — public domain dedication, same terms as the lesson text |
| Gated? | No. Nothing is downloaded, so there is nothing to gate. |
| Retrieved | Not applicable: it is written, not fetched |

### What it is

A fictional organisation's AI system registry: eight systems, each with the fields a
conformity pack needs before anyone can decide what applies to it, plus an `evidence` map of
document references with a status and a date.

Every system, product name, document reference, training record and date in the file is
invented. It describes no real product, no real company and no real audit. The eight entries
were chosen so that between them they exercise every branch of the lesson's rulebook:

| system | what it exercises |
|---|---|
| `cv-screener` | Annex III stand-alone high-risk; a document whose `review_due` has lapsed |
| `support-copilot` | Article 50 only, placed on the market before 2 August 2026 — the transitional marking date |
| `citizen-score` | an Article 5 prohibited practice, which outranks everything else |
| `xray-triage` | Annex I embedding in a medical device — the 2028 route |
| `atlas-7b` | a general-purpose AI model, dated to the 2025 GPAI milestone |
| `weld-inspector` | minimal risk, which is one duty rather than none |
| `loan-copilot` | high-risk *and* interactive *and* synthetic — obligations cumulate, dates differ |
| `cobot-guard` | both an Annex III area and an Annex I embedding: the precedence case |

### Why there is no download

The lesson's subject is a rulebook, not a corpus. Everything it needs is either this file or a
date in `claims.yaml`, so `lesson.py` reads from `assets/` and never opens a socket — the
execution gate runs with no connectivity and nothing changes.

### The legal texts are not in here

The AI Act and the Digital Omnibus are cited, not shipped. Their EUR-Lex and European
Commission URLs, the exact quotes relied on, and the date each was read are recorded in
`../claims.yaml`. The lesson encodes dates from those sources into a rulebook the student
runs; it reproduces no substantial part of either text.
