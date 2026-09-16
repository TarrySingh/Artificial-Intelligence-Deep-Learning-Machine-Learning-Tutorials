# %% [markdown]
# # P01-L02 · Risk classification as a decision procedure
#
# **You will build:** a classifier that does not merely return a tier. It returns the tier,
# the ordered trail of questions that produced it, the obligations that follow, the date each
# one starts to bind, and a sealed record you can append to the log you built in P01-L01.
# Then you point it at four analysts who described the same system four different ways, and
# find out which single question they actually disagreed about.
#
# **Time:** ~75 minutes · **Runs on:** a laptop CPU, no download, no network
# · **Prerequisites:** `T10-L01-ai-act-conformity-pack`, `P01-L01-article-12-logging`
#
# T10-L01 gave you a `classify()` that read flags off a dictionary. This lesson makes you
# earn those flags. A tier with no trail behind it is an opinion; a tier with a trail is a
# decision you can hand to someone who disagrees with you.
#
# By the end you will be able to:
#
# 1. Implement the Article 6(3) derogation, including the profiling rule that overrides it and
#    the Article 6(4) documentation duty that comes with claiming it.
# 2. Implement an ordered interview that reaches a tier in the fewest questions and records
#    which question was decisive.
# 3. Implement the obligation and date lookup that follows from a tier, and separate what
#    binds today from what binds later.
# 4. Seal a classification into a hash-covered record, and diff two records across a system
#    change to report which obligations appeared and which vanished.
# 5. Report a panel's disagreement as the earliest question they split on, rather than as a
#    spread of verdicts.
#
# > **This is engineering, not legal advice.** It is an exercise in turning a set of legal
# > criteria into an auditable decision procedure. Every article, quotation and date below is
# > sourced in `claims.yaml` with its URL and access date. Two of the dates are labelled as
# > *this lesson's reading* rather than as sourced fact, and section 5 says which and why. The
# > systems you classify are fictional. For a real system, read the Official Journal text and
# > take professional advice.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from typing import Any, Callable, NamedTuple

print("python", sys.version.split()[0], "· standard library only")

# The date this classification is run. Fixed, so every number below is reproducible: a
# classification record that changes silently with the calendar is not an artefact.
AS_OF = date(2026, 9, 16)
print("as of", AS_OF.isoformat())


def canonical_bytes(obj: Any) -> bytes:
    """One deterministic serialisation of an object. Given to you — you built it in P01-L01."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def parse_date(value: Any) -> date | None:
    """Accept an ISO string, a date, or None. Given to you; not graded."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


_FAILED_CHECKS: list[str] = []


def _try(label: str, check: Callable[[], None]) -> None:
    """Run a check, or a demo that depends on your code, without derailing the notebook.

    A stub you have not filled in yet simply says so. A wrong answer prints the check's own
    message — which names the likely mistake — and the notebook carries on, so one broken
    exercise never hides the feedback on the other five.
    """
    try:
        check()
    except NotImplementedError:
        print(f"{label}: not implemented yet — fill in the stub above, then re-run this cell.")
    except AssertionError as exc:
        _FAILED_CHECKS.append(label)
        print(f"{label}: FAILED — {exc}")
    except Exception as exc:  # a half-finished implementation raising something else
        _FAILED_CHECKS.append(label)
        print(f"{label}: raised {type(exc).__name__}: {exc}")


# %% [markdown]
# ## 1. The ladder, and what is actually deferred
#
# Everything below hangs off dates, so start with them. Each is sourced in `claims.yaml`.
# Read the table the next cell prints rather than this paragraph.
#
# The one sentence worth carrying: **the high-risk duties were deferred; the AI Act was not.**
# Article 50's transparency duties have applied since general application, and a system you
# classify today may owe something today even though its Chapter III duties are years out.

# %%
MILESTONES = {
    "2025-02-02": "Article 5 prohibited practices, and Article 4 AI literacy",
    "2025-08-02": "General-purpose AI model obligations, and the governance rules",
    "2026-08-02": "General application, including Article 50 transparency",
    "2026-12-02": "Article 50(2) marking, for systems on the market before 2026-08-02",
    "2027-12-02": "Chapter III duties for Annex III stand-alone high-risk systems",
    "2028-08-02": "Chapter III duties for Annex I product-embedded high-risk systems",
}

print(f"{'date':12s} {'':5s} what starts to bind")
for _iso, _what in sorted(MILESTONES.items()):
    print(f"{_iso:12s} {'PAST' if date.fromisoformat(_iso) <= AS_OF else '->':5s} {_what}")
_past = sum(date.fromisoformat(d) <= AS_OF for d in MILESTONES)
print(f"\n{_past} of {len(MILESTONES)} rungs already bind on {AS_OF.isoformat()}; "
      f"{len(MILESTONES) - _past} do not")

# %% [markdown]
# ## 2. The six tiers, and what "borderline" means
#
# A tier is not a severity score. It is the name of a route: it decides which obligations
# attach, which dates apply, and in module 8, which conformity assessment procedure you owe.
#
# `annex_iii_derogated` is the tier most classifiers do not have, and it is the one this
# lesson is about. It is **not** the same as `minimal`. A system that escapes the high-risk
# route under Article 6(3) picks up two duties of its own for having escaped it.

# %%
# Ordered most severe first. The order is load-bearing: it breaks ties in the panel report.
TIER_SEVERITY = (
    "prohibited",
    "high_risk_annex_i",
    "high_risk_annex_iii",
    "annex_iii_derogated",
    "gpai_model",
    "transparency_only",
    "minimal",
)
print(f"{len(TIER_SEVERITY)} tiers, most severe first:")
for _rank, _tier in enumerate(TIER_SEVERITY):
    print(f"  {_rank}  {_tier}")

# %% [markdown]
# ## 3. The systems
#
# Nine fictional systems, written as the dictionaries an intake form would produce. Nothing
# is read from disk and nothing is downloaded. Look at the flags, not at the names — the
# names are there so you can talk about them, and two of them are deliberately misleading.

# %%
SYSTEMS = {
    "mood-meter": {
        "id": "mood-meter", "name": "Infers how engaged each employee looks on a call",
        "prohibited_practice": "emotion inference in the workplace (Article 5(1)(f))",
        "placed_on_market": "2026-03-01",
    },
    "lift-door-sensor": {
        "id": "lift-door-sensor", "name": "Decides when a lift door may close",
        "annex_i_safety_component": True, "annex_iii_area": None,
        "placed_on_market": "2026-05-12",
    },
    "credit-scorer": {
        "id": "credit-scorer", "name": "Scores consumer loan applications",
        "annex_iii_area": "creditworthiness (Annex III point 5(b))",
        "derogation_claimed": False, "performs_profiling": True,
        "placed_on_market": "2026-04-01",
    },
    "cv-ranker": {
        "id": "cv-ranker", "name": "Ranks applicants for a vacancy and shortlists ten",
        "annex_iii_area": "employment (Annex III point 4(a))",
        "derogation_claimed": True, "performs_profiling": True,
        "significant_risk_of_harm": False, "improves_human_activity": True,
        "assessment_documented": True, "registered_in_eu_database": True,
        "justification": "A recruiter reads every shortlist before anyone is contacted.",
        "documented_on": "2026-05-01", "placed_on_market": "2026-06-01",
    },
    "shift-note-tidier": {
        "id": "shift-note-tidier", "name": "Reformats free-text shift notes into a fixed schema",
        "annex_iii_area": "workers' management (Annex III point 4(b))",
        "derogation_claimed": True, "performs_profiling": False,
        "significant_risk_of_harm": False, "narrow_procedural_task": True,
        "assessment_documented": True, "registered_in_eu_database": True,
        "justification": "Reformatting only; no field is added, scored, ranked or removed.",
        "documented_on": "2026-02-10", "placed_on_market": "2026-03-15",
    },
    "exam-prep-sorter": {
        "id": "exam-prep-sorter", "name": "Groups exam scripts by question for human markers",
        "annex_iii_area": "education (Annex III point 3(b))",
        "derogation_claimed": True, "performs_profiling": False,
        "significant_risk_of_harm": False, "preparatory_task": True,
        "assessment_documented": False, "registered_in_eu_database": False,
        "justification": "   ", "documented_on": None, "placed_on_market": "2026-01-09",
    },
    "house-lm": {
        "id": "house-lm", "name": "A general-purpose language model trained in house",
        "is_gpai_model": True, "placed_on_market": "2026-02-02",
    },
    "policy-drafter": {
        "id": "policy-drafter", "name": "Drafts internal policy text in a chat window",
        "interacts_with_humans": True, "generates_synthetic_content": True,
        "placed_on_market": "2025-11-04",
    },
    "demand-forecaster": {
        "id": "demand-forecaster", "name": "Forecasts warehouse demand from sales history",
        "placed_on_market": "2026-01-20",
    },
}

print(f"{len(SYSTEMS)} systems; "
      f"{sum(1 for s in SYSTEMS.values() if s.get('annex_iii_area'))} name an Annex III area, "
      f"{sum(1 for s in SYSTEMS.values() if s.get('derogation_claimed'))} claim the derogation")
print(json.dumps(SYSTEMS["cv-ranker"], indent=2))

# %% [markdown]
# ## 4. Exercise 1 — `derogation_assessment(system)`
#
# Article 6(2) says Annex III systems **shall be considered to be high-risk**. Article 6(3)
# opens one door out of that, and the door has a bar across it.
#
# The derogation holds where the system "does not pose a significant risk of harm to the
# health, safety or fundamental rights of natural persons" **and** it meets at least one of
# four conditions. But an Annex III system "shall always be considered to be high-risk where
# the AI system performs profiling of natural persons" — that sentence overrides the whole
# paragraph, whichever condition you met. And Article 6(4) attaches a duty to the claim
# itself: document the assessment *before* the system is placed on the market, register under
# Article 49(2), and produce the documentation on request.
#
# So there are three ways to fail a derogation you were entitled to on the merits: profile,
# fail to document, or fail to register.

# %%
# The four conditions of Article 6(3), in the order the article lists them. The order is
# graded: when two hold, the record names the first, so two reviewers agree on the citation.
DEROGATION_CONDITIONS = (
    ("narrow_procedural_task", "Article 6(3)(a) a narrow procedural task"),
    ("improves_human_activity", "Article 6(3)(b) improving a previously completed human activity"),
    ("detects_decision_patterns", "Article 6(3)(c) detecting decision patterns without "
                                  "replacing the prior human assessment"),
    ("preparatory_task", "Article 6(3)(d) a preparatory task to an Annex III assessment"),
)

# Reasons are ids, not sentences, so the rubric grades the outcome rather than your prose.
REASON_NOT_CLAIMED = "not_claimed"
REASON_PROFILING_OVERRIDE = "profiling_override"
REASON_SIGNIFICANT_RISK = "significant_risk_of_harm"
REASON_NO_CONDITION_MET = "no_condition_met"
REASON_DOCUMENTATION_INCOMPLETE = "documentation_incomplete"
REASON_GRANTED = "granted"

# The four things Article 6(4) and Article 49(2) ask of a provider who claims the derogation.
DOCUMENTATION_DUTIES = ("assessment_documented", "justification",
                        "documented_before_placing", "registered_in_eu_database")

print(f"{len(DEROGATION_CONDITIONS)} conditions, any one of which suffices:")
for _field, _label in DEROGATION_CONDITIONS:
    print(f"  {_field:26s} {_label}")
print(f"\n{len(DOCUMENTATION_DUTIES)} documentation duties attach to claiming it: "
      f"{', '.join(DOCUMENTATION_DUTIES)}")


# %%
def derogation_assessment(system: dict) -> dict:
    """Decide whether the Article 6(3) derogation holds for an Annex III system.

    Work in this order, and stop at the first rule that applies. The order matters: a
    profiling system must be refused even when its paperwork is perfect, and a system that
    never claimed the derogation must not be reported as having failed its documentation.

      1. `system["derogation_claimed"]` falsy   -> not granted, REASON_NOT_CLAIMED,
         and `documentation_gaps` is [] — no duty arises from a claim nobody made.
      2. `system["performs_profiling"]` truthy  -> not granted, REASON_PROFILING_OVERRIDE.
      3. `system["significant_risk_of_harm"]` truthy -> not granted, REASON_SIGNIFICANT_RISK.
      4. no condition in DEROGATION_CONDITIONS is truthy -> not granted, REASON_NO_CONDITION_MET.
      5. `documentation_gaps` is non-empty      -> not granted, REASON_DOCUMENTATION_INCOMPLETE.
      6. otherwise                              -> granted, REASON_GRANTED.

    `condition` is the FIELD NAME of the first condition in DEROGATION_CONDITIONS that is
    truthy, or None when none is. Report it even when the derogation is refused: "you met
    6(3)(b) and it did not save you" is the finding a reviewer needs.

    `documentation_gaps` is the sorted list of failed duties, drawn from DOCUMENTATION_DUTIES,
    computed whenever the derogation is claimed (steps 2 to 6), and it fails when:
      - "assessment_documented"        : the field is falsy;
      - "justification"                : the field is missing, or blank after .strip();
      - "documented_before_placing"    : either date is missing, or documented_on is not
                                         STRICTLY earlier than placed_on_market — Article 6(4)
                                         says before the system is placed on the market;
      - "registered_in_eu_database"    : the field is falsy.
    Use parse_date() for the two dates; it returns None for a missing one.

    Example:
        >>> derogation_assessment({"derogation_claimed": False})["reason"]
        'not_claimed'

    Returns:
        dict with exactly these five keys:
          "claimed"             bool
          "granted"             bool
          "condition"           str | None
          "reason"              str, one of the six REASON_* constants
          "documentation_gaps"  list[str], sorted
    """
    # YOUR CODE HERE
    raise NotImplementedError


# Public checks — run these as often as you like.
def _check_derogation() -> None:
    got = derogation_assessment(SYSTEMS["credit-scorer"])
    assert set(got) == {"claimed", "granted", "condition", "reason", "documentation_gaps"}, (
        f"keys were {sorted(got)} — return exactly the five documented names.")
    assert got["reason"] == REASON_NOT_CLAIMED and got["granted"] is False, (
        f"credit-scorer never claimed the derogation, so the reason is "
        f"{REASON_NOT_CLAIMED!r}, not {got['reason']!r}.")
    assert got["documentation_gaps"] == [], (
        f"got {got['documentation_gaps']} — a provider who never claimed the derogation owes "
        "no Article 6(4) documentation, so do not report gaps against them.")

    got = derogation_assessment(SYSTEMS["cv-ranker"])
    assert got["reason"] == REASON_PROFILING_OVERRIDE and got["granted"] is False, (
        f"cv-ranker got {got['reason']!r}. Its paperwork is perfect and it meets 6(3)(b); it "
        "profiles natural persons, and that sentence of Article 6(3) overrides the rest. "
        "Check profiling BEFORE you check the conditions and the documentation.")
    assert got["condition"] == "improves_human_activity", (
        f"condition was {got['condition']!r} — still name the condition that was met. A "
        "refusal that hides what the provider argued is not a reviewable refusal.")

    got = derogation_assessment(SYSTEMS["exam-prep-sorter"])
    assert got["reason"] == REASON_DOCUMENTATION_INCOMPLETE, (
        f"exam-prep-sorter got {got['reason']!r} — it is entitled to 6(3)(d) on the merits "
        "and has documented none of it, so the documentation duty is what refuses it.")
    assert got["documentation_gaps"] == sorted(
        ["assessment_documented", "justification", "documented_before_placing",
         "registered_in_eu_database"]), (
        f"got {got['documentation_gaps']} — all four duties fail here, and the list is "
        "sorted. A justification of whitespace is not a justification: use .strip().")

    got = derogation_assessment(SYSTEMS["shift-note-tidier"])
    assert got["granted"] is True and got["reason"] == REASON_GRANTED, (
        f"shift-note-tidier got {got['reason']!r} with gaps {got['documentation_gaps']} — it "
        "meets 6(3)(a), does not profile, and documented on 2026-02-10 before placing on "
        "2026-03-15.")

    late = dict(SYSTEMS["shift-note-tidier"], documented_on="2026-03-15")
    assert derogation_assessment(late)["documentation_gaps"] == ["documented_before_placing"], (
        "documenting ON the day of placing is not documenting BEFORE it — the comparison is "
        "strict.")

    bare = dict(SYSTEMS["shift-note-tidier"], narrow_procedural_task=False)
    assert derogation_assessment(bare)["reason"] == REASON_NO_CONDITION_MET, (
        "with no condition truthy the derogation fails on the merits, before documentation.")
    print("exercise 1 looks right")


# %% [markdown]
# ## 5. Exercise 2 — `ask_tier(system)`: the interview
#
# A classifier that evaluates every flag and then picks a winner cannot tell you what decided
# the outcome, because nothing did. An interview can: it asks questions in a fixed order and
# **stops at the first answer that decides**. The trail it leaves is the explanation.
#
# Two properties the rubric checks. **Minimality:** a prohibited system needs one question,
# and asking the other five is a defect, not thoroughness. **Exactly one decisive answer,**
# and it is the last one in the trail. Question 3 is never decisive by itself — naming an
# Annex III area opens the derogation question, it does not settle the tier.

# %%
QUESTIONS = (
    ("q1_prohibited", "Does it carry out a practice prohibited by Article 5?"),
    ("q2_annex_i_safety_component",
     "Is it a safety component of a product covered by Annex I harmonisation law?"),
    ("q3_annex_iii_area", "Does its intended purpose fall in one of the eight Annex III areas?"),
    ("q4_article_6_3", "Does the Article 6(3) derogation hold, documentation and all?"),
    ("q5_gpai_model", "Is it a general-purpose AI model?"),
    ("q6_transparency", "Does it interact with people or generate synthetic content?"),
)
QUESTION_IDS = tuple(q for q, _ in QUESTIONS)


class Answer(NamedTuple):
    """One line of the trail: the question, what it was answered, and whether it decided."""
    question: str
    answer: Any
    decisive: bool


class Interview(NamedTuple):
    """The outcome of an interview: the tier, and the trail of questions that reached it."""
    tier: str
    trail: list


print(f"{len(QUESTIONS)} questions, asked in this order until one decides:")
for _qid, _text in QUESTIONS:
    print(f"  {_qid:28s} {_text}")


# %%
def ask_tier(system: dict) -> Interview:
    """Interview a system description and return its tier and the trail that reached it.

    Ask in order, appending one Answer per question asked, and RETURN as soon as a question
    decides. Never ask a question after the tier is settled.

      q1_prohibited              answer bool(system.get("prohibited_practice")).
                                 True -> decisive, tier "prohibited".
      q2_annex_i_safety_component answer bool(system.get("annex_i_safety_component")).
                                 True -> decisive, tier "high_risk_annex_i".
      q3_annex_iii_area          answer system.get("annex_iii_area") or None. NEVER decisive.
                                 If truthy, go on to q4; if falsy, go on to q5.
      q4_article_6_3             answer derogation_assessment(system)["reason"]. ALWAYS
                                 decisive. Granted -> "annex_iii_derogated",
                                 otherwise -> "high_risk_annex_iii".
      q5_gpai_model              answer bool(system.get("is_gpai_model")).
                                 True -> decisive, tier "gpai_model".
      q6_transparency            answer bool(interacts_with_humans or
                                 generates_synthetic_content). ALWAYS decisive:
                                 True -> "transparency_only", False -> "minimal".

    Note what q6 being always decisive buys you: "minimal" is an answered question, not a
    fall-through, so every trail ends on the question that produced the tier.

    Example:
        >>> iv = ask_tier({"id": "x", "prohibited_practice": "social scoring"})
        >>> iv.tier, len(iv.trail), iv.trail[0].question
        ('prohibited', 1, 'q1_prohibited')

    Returns:
        Interview(tier=str, trail=list[Answer]) — the trail in the order asked.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_ask_tier() -> None:
    iv = ask_tier(SYSTEMS["mood-meter"])
    assert isinstance(iv, Interview) and iv.tier == "prohibited", (
        f"mood-meter is prohibited; you returned {getattr(iv, 'tier', iv)!r}.")
    assert len(iv.trail) == 1, (
        f"the trail has {len(iv.trail)} entries — a prohibited practice is settled by the "
        "first question, so return immediately instead of asking the other five.")
    assert all(isinstance(a, Answer) for a in iv.trail), "the trail holds Answer tuples."

    for _sid, _want_tier, _want_len in (("lift-door-sensor", "high_risk_annex_i", 2),
                                        ("cv-ranker", "high_risk_annex_iii", 4),
                                        ("shift-note-tidier", "annex_iii_derogated", 4),
                                        ("house-lm", "gpai_model", 4),
                                        ("policy-drafter", "transparency_only", 5),
                                        ("demand-forecaster", "minimal", 5)):
        iv = ask_tier(SYSTEMS[_sid])
        assert iv.tier == _want_tier, f"{_sid}: got {iv.tier!r}, expected {_want_tier!r}."
        assert len(iv.trail) == _want_len, (
            f"{_sid}: asked {[a.question for a in iv.trail]} — expected {_want_len} "
            "questions. Ask in order and stop at the first decisive answer; q3 is skipped "
            "onto q5 when no Annex III area is named, and q4 is only asked when one is.")
        assert [a.question for a in iv.trail] == sorted(
            (a.question for a in iv.trail), key=QUESTION_IDS.index), (
            f"{_sid}: the trail is out of order — ask the questions as QUESTIONS lists them.")
        decisive = [a for a in iv.trail if a.decisive]
        assert len(decisive) == 1 and decisive[0] is iv.trail[-1], (
            f"{_sid}: {len(decisive)} decisive answers. Exactly one, and it is the last: it "
            "is the question that ended the interview.")

    trail = {a.question: a for a in ask_tier(SYSTEMS["credit-scorer"]).trail}
    assert trail["q3_annex_iii_area"].decisive is False, (
        "q3 is never decisive on its own. Naming an Annex III area opens question 4; "
        "Article 6(3) is what settles the tier.")
    assert trail["q4_article_6_3"].answer == REASON_NOT_CLAIMED, (
        f"q4's recorded answer is the derogation reason id, not {trail['q4_article_6_3'].answer!r} "
        "— the trail has to say WHY the derogation failed, not just that it did.")
    assert ask_tier(SYSTEMS["policy-drafter"]).trail[-1].answer is True, (
        "q6's answer is the boolean OR of the two transparency flags.")
    print("exercise 2 looks right")


# %% [markdown]
# ## 6. Exercise 3 — `obligations_for(tier, system, as_of)`
#
# The tier is a route, not a verdict, and the route decides the bill. Three things people get
# wrong, and all three are graded.
#
# **Obligations cumulate.** A high-risk system that also chats owes Chapter III *and*
# Article 50, on different dates. Being high-risk does not absorb the transparency duties.
#
# **The derogation swaps duties, it does not delete them.** An `annex_iii_derogated` system
# owes the Article 6(4) documentation and the Article 49(2) registration. It is not `minimal`.
#
# **Two of these dates are not sourced fact.** The table below carries a `basis` field. The
# Chapter III dates come from the Commission's own page; the date on which the *derogation*
# duties start to bind is this lesson's reading — Article 6 sits in the chapter the Omnibus
# deferred, so the lesson puts them on the Annex III date and labels the inference as one.

# %%
GENERAL_APPLICATION = "2026-08-02"
ART50_MARK_LEGACY = "2026-12-02"    # transitional, for systems placed before general application

OBLIGATIONS = {
    "art4_ai_literacy": {
        "article": "Article 4", "applies_from": "2025-02-02", "basis": "sourced",
        "label": "AI literacy for the people who build and operate it"},
    "art5_prohibited_practice": {
        "article": "Article 5", "applies_from": "2025-02-02", "basis": "sourced",
        "label": "Prohibited: it must not be placed on the market or used"},
    "art53_gpai_model": {
        "article": "Article 53", "applies_from": "2025-08-02", "basis": "sourced",
        "label": "GPAI documentation, copyright policy, training-data summary"},
    "art50_inform_interaction": {
        "article": "Article 50(1)", "applies_from": GENERAL_APPLICATION, "basis": "sourced",
        "label": "Tell people they are interacting with an AI system"},
    "art50_mark_synthetic": {
        "article": "Article 50(2)", "applies_from": GENERAL_APPLICATION, "basis": "sourced",
        "label": "Mark synthetic content in a machine-readable format"},
    "ch3_high_risk_annex_iii": {
        "article": "Article 6(2)", "applies_from": "2027-12-02", "basis": "sourced",
        "label": "Chapter III duties, stand-alone high-risk"},
    "ch3_high_risk_annex_i": {
        "article": "Article 6(1)", "applies_from": "2028-08-02", "basis": "sourced",
        "label": "Chapter III duties, product-embedded high-risk"},
    "art6_4_derogation_documentation": {
        "article": "Article 6(4)", "applies_from": "2027-12-02", "basis": "lesson's reading",
        "label": "Document the not-high-risk assessment, and produce it on request"},
    "art49_2_registration": {
        "article": "Article 49(2)", "applies_from": "2027-12-02", "basis": "lesson's reading",
        "label": "Register the derogated system in the EU database"},
}

_inferred = [o for o, v in OBLIGATIONS.items() if v["basis"] != "sourced"]
print(f"{len(OBLIGATIONS)} obligations; {len(_inferred)} carry a date this lesson inferred "
      f"rather than sourced: {', '.join(sorted(_inferred))}")
for _oid, _v in sorted(OBLIGATIONS.items(), key=lambda kv: kv[1]["applies_from"]):
    print(f"  {_v['applies_from']}  {_oid:32s} {_v['article']:16s} [{_v['basis']}]")


# %%
def obligations_for(tier: str, system: dict, as_of: date = AS_OF) -> dict:
    """Return the obligations a tier and a system description imply, with their dates.

    Start with "art4_ai_literacy", which applies to every system here.

    If the tier is "prohibited", add "art5_prohibited_practice" and STOP: a system that must
    not be on the market does not also owe the duties of one that is. (That is this lesson's
    modelling choice, carried over from T10-L01, not a quotation.)

    Otherwise add, independently of one another:
      tier "high_risk_annex_i"     -> "ch3_high_risk_annex_i"
      tier "high_risk_annex_iii"   -> "ch3_high_risk_annex_iii"
      tier "annex_iii_derogated"   -> "art6_4_derogation_documentation" AND "art49_2_registration"
      tier "gpai_model"            -> "art53_gpai_model"
      system["interacts_with_humans"] truthy        -> "art50_inform_interaction"
      system["generates_synthetic_content"] truthy  -> "art50_mark_synthetic"

    The Article 50 duties are read off the SYSTEM, not off the tier, so a high-risk system
    that also chats collects them too.

    Dates are OBLIGATIONS[oid]["applies_from"], with one exception: "art50_mark_synthetic" is
    ART50_MARK_LEGACY when system["placed_on_market"] is strictly earlier than
    GENERAL_APPLICATION, and the table's date otherwise.

    Example:
        >>> r = obligations_for("minimal", {"id": "x", "placed_on_market": "2026-01-20"})
        >>> r["obligations"], r["binding_now"]
        (['art4_ai_literacy'], ['art4_ai_literacy'])

    Returns:
        dict with exactly these four keys:
          "obligations"   list[str], sorted
          "deadlines"     dict[str, str], every obligation id -> its ISO date for THIS system
          "binding_now"   list[str], sorted, the ids whose date is on or before as_of
          "next_deadline" str | None, the earliest date STRICTLY after as_of, or None
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_obligations() -> None:
    got = obligations_for("minimal", SYSTEMS["demand-forecaster"])
    assert set(got) == {"obligations", "deadlines", "binding_now", "next_deadline"}, (
        f"keys were {sorted(got)} — return exactly the four documented names.")
    assert got["obligations"] == ["art4_ai_literacy"], (
        f"got {got['obligations']} — minimal risk is one duty, not none.")
    assert got["next_deadline"] is None, (
        f"got {got['next_deadline']!r} — nothing is still ahead of this system, so None.")

    got = obligations_for("prohibited", SYSTEMS["mood-meter"])
    assert got["obligations"] == ["art4_ai_literacy", "art5_prohibited_practice"], (
        f"got {got['obligations']} — once prohibited, stop.")

    got = obligations_for("annex_iii_derogated", SYSTEMS["shift-note-tidier"])
    assert set(got["obligations"]) == {"art4_ai_literacy", "art6_4_derogation_documentation",
                                       "art49_2_registration"}, (
        f"got {got['obligations']} — a derogated system is not a minimal one. Escaping the "
        "high-risk route under Article 6(3) buys you the 6(4) documentation duty and the "
        "49(2) registration duty.")

    scorer = dict(SYSTEMS["credit-scorer"], interacts_with_humans=True,
                  generates_synthetic_content=True)
    got = obligations_for("high_risk_annex_iii", scorer)
    assert "art50_inform_interaction" in got["obligations"], (
        f"got {got['obligations']} — obligations cumulate. Read the Article 50 duties off the "
        "system's own flags; being high-risk does not absorb them.")
    assert got["deadlines"]["art50_mark_synthetic"] == ART50_MARK_LEGACY, (
        f"credit-scorer was placed {scorer['placed_on_market']}, before "
        f"{GENERAL_APPLICATION}, so marking is due {ART50_MARK_LEGACY}, not "
        f"{got['deadlines']['art50_mark_synthetic']}.")
    assert got["binding_now"] == ["art4_ai_literacy", "art50_inform_interaction"], (
        f"got {got['binding_now']} — binding_now is every id whose date is on or before "
        f"{AS_OF.isoformat()}. Chapter III is not; Article 50(1) is.")
    assert got["next_deadline"] == ART50_MARK_LEGACY, (
        f"got {got['next_deadline']!r} — the next deadline is the earliest date STRICTLY "
        "after as_of, which is the transitional marking date, not the Chapter III date.")

    late = dict(SYSTEMS["policy-drafter"], placed_on_market=GENERAL_APPLICATION)
    assert obligations_for("transparency_only", late)["deadlines"]["art50_mark_synthetic"] \
        == GENERAL_APPLICATION, (
        "a system placed ON the general application date is not 'before' it: the comparison "
        "is strict, so it gets the standard date.")
    print("exercise 3 looks right")


# %% [markdown]
# ## 7. Exercise 4 — `classification_record(system, as_of)`
#
# Now seal it. A record that can be edited after the fact is a note, not evidence. The seal
# is the same construction you built in P01-L01: a SHA-256 over one canonical serialisation
# of the record, covering everything except the digest field itself.
#
# The point of sealing the *trail* along with the tier is that the two can disagree. A record
# whose tier says `minimal` and whose trail says the derogation was refused is a record
# somebody edited, and the digest is what catches it.

# %%
def classification_record(system: dict, as_of: date = AS_OF) -> dict:
    """Assemble and seal one classification record for a system.

    Build the record with exactly these eight keys, in any order:

      "system_id"      system["id"]
      "as_of"          as_of.isoformat()
      "risk_tier"      the tier from ask_tier(system)
      "question_trail" the trail as a list of dicts, one per Answer, each with exactly the
                       keys "question", "answer" and "decisive" — plain JSON, because a
                       NamedTuple does not survive a round trip through a log
      "derogation"     derogation_assessment(system) when an Annex III area was named
                       (i.e. q4 was asked), otherwise None
      "obligations"    obligations_for(tier, system, as_of)["obligations"]
      "deadlines"      obligations_for(tier, system, as_of)["deadlines"]
      "record_hash"    hashlib.sha256(canonical_bytes(everything above)).hexdigest()

    Compute the digest over the record WITHOUT "record_hash" in it, then add the field. A
    digest that covered itself could not be recomputed by anyone checking the record.

    Example:
        >>> r = classification_record(SYSTEMS["demand-forecaster"])
        >>> r["risk_tier"], len(r["record_hash"]), r["derogation"]
        ('minimal', 64, None)

    Returns:
        dict with exactly those eight keys.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_record() -> None:
    rec = classification_record(SYSTEMS["cv-ranker"])
    assert set(rec) == {"system_id", "as_of", "risk_tier", "question_trail", "derogation",
                        "obligations", "deadlines", "record_hash"}, (
        f"keys were {sorted(rec)} — return exactly the eight documented names.")
    assert rec["risk_tier"] == "high_risk_annex_iii" and rec["system_id"] == "cv-ranker", (
        f"got tier {rec['risk_tier']!r} for {rec['system_id']!r} — take the tier from "
        "ask_tier() and the id from system['id'].")
    assert all(set(step) == {"question", "answer", "decisive"} for step in rec["question_trail"]), (
        "each trail step is a plain dict with exactly question, answer and decisive — "
        "serialise the Answer tuples rather than storing them.")
    assert json.loads(canonical_bytes(rec).decode()) == rec, (
        "the record must survive a JSON round trip; a NamedTuple or a date object in it "
        "will not.")

    bare = dict(rec)
    digest = bare.pop("record_hash")
    assert hashlib.sha256(canonical_bytes(bare)).hexdigest() == digest, (
        "the digest must be recomputable by anyone: hash the record WITHOUT record_hash "
        "in it, then add the field.")

    tampered = json.loads(json.dumps(bare))
    tampered["risk_tier"] = "minimal"
    assert hashlib.sha256(canonical_bytes(tampered)).hexdigest() != digest, "the tier is covered."
    tampered = json.loads(json.dumps(bare))
    tampered["question_trail"][-1]["answer"] = REASON_GRANTED
    assert hashlib.sha256(canonical_bytes(tampered)).hexdigest() != digest, (
        "the trail must be covered too, or a record can be re-explained after the fact.")

    assert classification_record(SYSTEMS["demand-forecaster"])["derogation"] is None, (
        "no Annex III area means q4 was never asked, so there is no derogation to record.")
    assert classification_record(SYSTEMS["shift-note-tidier"])["derogation"]["granted"] is True, (
        "store the whole derogation_assessment dict, not just its reason — a reviewer needs "
        "the condition and the gaps as well.")
    assert classification_record(SYSTEMS["cv-ranker"], date(2027, 1, 1))["record_hash"] != \
        rec["record_hash"], "as_of is part of the record, so it is part of the digest."
    print("exercise 4 looks right")


# %% [markdown]
# ## 8. Exercise 5 — `diff_records(before, after)`
#
# Systems change. A new deployment context, a new user population, a feature that turns a
# reformatter into a ranker — and the classification that was true last quarter is now
# wrong. The diff is the artefact that tells a release manager what a change cost.
#
# What you want out of it is not "the tier changed". It is: which obligations **appeared**,
# which **vanished**, from what date, and which question changed its answer to cause it.

# %%
def diff_records(before: dict, after: dict) -> dict:
    """Diff two classification records for the SAME system and report what moved.

    Raise ValueError when before["system_id"] != after["system_id"]: diffing two different
    systems produces a plausible report about nothing, which is the worst kind.

    "changed_answers" covers questions asked in BOTH trails whose recorded "answer" differs.
    A question asked in only one trail is not a changed answer — an earlier question must have
    changed for the interview to take a different path, and that one is reported instead.

    "requires_reassessment" is True when the tier changed, or any obligation appeared, or any
    vanished. A merely rescheduled date does not trigger it.

    Example:
        >>> a = classification_record(SYSTEMS["demand-forecaster"])
        >>> diff_records(a, a)["requires_reassessment"]
        False

    Returns:
        dict with exactly these eight keys:
          "system_id"            str
          "tier_before"          str
          "tier_after"           str
          "tier_changed"         bool
          "added"                dict[str, str], obligation id -> its date in `after`
          "removed"              dict[str, str], obligation id -> its date in `before`
          "changed_answers"      list[str], sorted question ids
          "requires_reassessment" bool
    """
    # YOUR CODE HERE
    raise NotImplementedError


# A change of user population: the tidier starts scoring the notes it used to reformat.
SHIFT_TIDIER_AFTER = dict(
    SYSTEMS["shift-note-tidier"], performs_profiling=True, narrow_procedural_task=False,
    name="Scores free-text shift notes and ranks workers by reliability")


def _check_diff() -> None:
    before = classification_record(SYSTEMS["shift-note-tidier"])
    after = classification_record(SHIFT_TIDIER_AFTER)
    got = diff_records(before, after)
    assert set(got) == {"system_id", "tier_before", "tier_after", "tier_changed", "added",
                        "removed", "changed_answers", "requires_reassessment"}, (
        f"keys were {sorted(got)} — return exactly the eight documented names.")
    assert (got["tier_before"], got["tier_after"]) == \
        ("annex_iii_derogated", "high_risk_annex_iii"), (
        f"got {got['tier_before']!r} -> {got['tier_after']!r}.")
    assert got["tier_changed"] is True and got["requires_reassessment"] is True
    assert set(got["added"]) == {"ch3_high_risk_annex_iii"}, (
        f"added {sorted(got['added'])} — the profiling flag pushed it back onto the high-risk "
        "route.")
    assert got["added"]["ch3_high_risk_annex_iii"] == "2027-12-02", (
        "the value is the obligation's date in the AFTER record, so a release manager can "
        "read the deadline off the diff.")
    assert set(got["removed"]) == {"art6_4_derogation_documentation", "art49_2_registration"}, (
        f"removed {sorted(got['removed'])} — the derogation duties go when the derogation "
        "does. Both sides matter: obligations vanish as well as appear.")
    assert got["changed_answers"] == ["q4_article_6_3"], (
        f"got {got['changed_answers']} — q1, q2 and q3 answered identically; only the "
        "derogation reason moved.")

    same = diff_records(before, before)
    assert same["requires_reassessment"] is False and same["changed_answers"] == [] \
        and same["added"] == {} and same["removed"] == {}, (
        "a record diffed against itself moves nothing.")

    drafter = classification_record(SYSTEMS["policy-drafter"])
    got = diff_records(drafter, classification_record(
        dict(SYSTEMS["policy-drafter"], generates_synthetic_content=False)))
    assert got["tier_changed"] is False and set(got["removed"]) == {"art50_mark_synthetic"}, (
        f"got tier_changed={got['tier_changed']}, removed={sorted(got['removed'])} — the tier "
        "held at transparency_only while an obligation vanished, which is exactly the case a "
        "tier-only diff misses.")
    assert got["requires_reassessment"] is True, (
        "an obligation vanished, so the classification has to be redone and re-documented "
        "even though the tier held.")
    assert got["changed_answers"] == [], (
        f"got {got['changed_answers']} — q6 answers the OR of two flags and the other is "
        "still True, so the trail is identical while an obligation vanished. Report what the "
        "trails say; do not reverse-engineer changed_answers from the obligation diff.")

    try:
        diff_records(before, drafter)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "diffing two different systems must raise ValueError, not quietly produce a "
            "report about nothing.")
    print("exercise 5 looks right")


# %% [markdown]
# ## 9. Exercise 6 — `disagreement_report(panel)`
#
# Four analysts fill in the same intake form about the same system and hand you four
# different tiers. The useless report says "opinions ranged from minimal to high-risk". The
# useful one says: *the earliest question you split on is question 3, and until you settle it
# nothing downstream is worth arguing about.*
#
# Find the **pivot**: the first question, in `QUESTIONS` order, on which the analysts who
# were asked it did not all give the same answer. Everything after the pivot is downstream of
# an unresolved disagreement.

# %%
# One system, four intake forms. Same id, because it is one system.
PANEL = {
    "analyst_a": dict(SYSTEMS["shift-note-tidier"], id="shift-allocator",
                      annex_iii_area="workers' management (Annex III point 4(b))"),
    "analyst_b": dict(SYSTEMS["shift-note-tidier"], id="shift-allocator",
                      annex_iii_area="workers' management (Annex III point 4(b))",
                      performs_profiling=True),
    "analyst_c": dict(SYSTEMS["shift-note-tidier"], id="shift-allocator",
                      annex_iii_area="workers' management (Annex III point 4(b))",
                      significant_risk_of_harm=True),
    # Analyst D does not think allocating shifts is workers' management at all.
    "analyst_d": dict(SYSTEMS["shift-note-tidier"], id="shift-allocator", annex_iii_area=None),
}
print(f"{len(PANEL)} intake forms for one system: {', '.join(sorted(PANEL))}")


def disagreement_report(panel: dict) -> dict:
    """Interview every analyst's description of one system and report where they split.

    Raise ValueError when the descriptions do not all carry the same "id": a panel is several
    views of ONE system.

    "pivot_question" is the first question id in QUESTION_IDS order for which the analysts who
    were asked it gave more than one distinct answer, or None when nobody disagreed anywhere.
    "pivot_answers" maps only those analysts who were asked the pivot to their answer, and is
    {} when there is no pivot.

    "majority_tier" is the tier the most analysts reached. Break a tie by severity, taking the
    tier that appears EARLIEST in TIER_SEVERITY — when a panel is split down the middle, an
    engineering team does not get to average the answer down.

    Example:
        >>> one = {"only": SYSTEMS["demand-forecaster"]}
        >>> r = disagreement_report(one)
        >>> r["unanimous"], r["pivot_question"], r["majority_tier"]
        (True, None, 'minimal')

    Returns:
        dict with exactly these six keys:
          "system_id"      str
          "unanimous"      bool, True when every analyst reached the same tier
          "tiers"          dict[str, list[str]], tier -> sorted analyst ids that reached it
          "majority_tier"  str
          "pivot_question" str | None
          "pivot_answers"  dict[str, Any], analyst id -> answer to the pivot question
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_disagreement() -> None:
    got = disagreement_report(PANEL)
    assert set(got) == {"system_id", "unanimous", "tiers", "majority_tier", "pivot_question",
                        "pivot_answers"}, (
        f"keys were {sorted(got)} — return exactly the six documented names.")
    assert got["system_id"] == "shift-allocator" and got["unanimous"] is False
    assert got["tiers"] == {"annex_iii_derogated": ["analyst_a"],
                            "high_risk_annex_iii": ["analyst_b", "analyst_c"],
                            "minimal": ["analyst_d"]}, (
        f"got {got['tiers']} — group analysts by the tier they reached, ids sorted, and "
        "include no tier nobody reached.")
    assert got["majority_tier"] == "high_risk_annex_iii", (
        f"got {got['majority_tier']!r} — two of the four reached it.")
    assert got["pivot_question"] == "q3_annex_iii_area", (
        f"got {got['pivot_question']!r}. The panel thinks it is arguing about profiling at q4, "
        "but analyst_d never reached q4: the earliest split is over whether this is an "
        "Annex III area at all. Walk QUESTION_IDS in order and stop at the first disagreement.")
    assert set(got["pivot_answers"]) == set(PANEL), (
        "every analyst was asked q3, so every one of them appears in pivot_answers.")
    assert got["pivot_answers"]["analyst_d"] is None, (
        "analyst_d's answer to q3 is None — that is the answer, and it is what splits them.")

    agreed = {k: v for k, v in PANEL.items() if k in ("analyst_a",)}
    got = disagreement_report(agreed)
    assert got["unanimous"] is True and got["pivot_question"] is None \
        and got["pivot_answers"] == {}, (
        "one analyst cannot disagree with anyone: no pivot, and pivot_answers is empty.")

    tie = {"x": PANEL["analyst_a"], "y": PANEL["analyst_b"]}
    assert disagreement_report(tie)["majority_tier"] == "high_risk_annex_iii", (
        "one each: break the tie by TIER_SEVERITY order, which puts high_risk_annex_iii "
        "ahead of annex_iii_derogated. Do not let Counter.most_common pick by insertion "
        "order — that makes the verdict depend on who filled the form in first.")

    try:
        disagreement_report({"a": SYSTEMS["credit-scorer"], "b": SYSTEMS["house-lm"]})
    except ValueError:
        pass
    else:
        raise AssertionError("two different systems are not a panel: raise ValueError.")
    print("exercise 6 looks right")


# %% [markdown]
# ## 10. The artefact
#
# Run the whole procedure over all nine systems, then look at one record in full and at the
# panel report. This is the thing that goes in the pack, and it is appendable to the P01-L01
# log as one event per record: the digest is already there.

# %%
def classification_table(as_of: date = AS_OF) -> None:
    """Print one row per system: tier, questions asked, what binds now, what binds next."""
    print(f"{'system':20s} {'tier':22s} {'asked':6s} {'now':4s} next")
    for sid, system in SYSTEMS.items():
        interview = ask_tier(system)
        duties = obligations_for(interview.tier, system, as_of)
        print(f"{sid:20s} {interview.tier:22s} {len(interview.trail):<6d} "
              f"{len(duties['binding_now']):<4d} {duties['next_deadline'] or '-'}")


_try("the table", classification_table)

# %%
def _show_record() -> None:
    record = classification_record(SYSTEMS["cv-ranker"])
    print(json.dumps(record, indent=2))
    decisive = [s for s in record["question_trail"] if s["decisive"]][0]
    print(f"\ndecided by {decisive['question']} answering {decisive['answer']!r} after "
          f"{len(record['question_trail'])} questions")


_try("the record", _show_record)

# %%
def _show_panel() -> None:
    report = disagreement_report(PANEL)
    print(f"system {report['system_id']}: "
          f"{'unanimous' if report['unanimous'] else 'split'} across {len(PANEL)} analysts")
    for tier in TIER_SEVERITY:
        if tier in report["tiers"]:
            print(f"  {tier:22s} {', '.join(report['tiers'][tier])}")
    print(f"  majority: {report['majority_tier']}")
    print(f"\npivot: {report['pivot_question']}")
    for analyst, answer in sorted(report["pivot_answers"].items()):
        print(f"  {analyst:12s} {answer!r}")
    print("\nEverything after the pivot is downstream of an unresolved disagreement.")


_try("the panel", _show_panel)

# %%
def _show_diff() -> None:
    diff = diff_records(classification_record(SYSTEMS["shift-note-tidier"]),
                        classification_record(SHIFT_TIDIER_AFTER))
    print(f"{diff['system_id']}: {diff['tier_before']} -> {diff['tier_after']}")
    for label, block in (("+ appeared", diff["added"]), ("- vanished", diff["removed"])):
        for oid, when in sorted(block.items()):
            print(f"  {label} {oid:34s} from {when}")
    print(f"  changed answers: {diff['changed_answers']}")
    print(f"  requires reassessment: {diff['requires_reassessment']}")


_try("the diff", _show_diff)

# %% [markdown]
# ## 11. Common mistakes
#
# - **Evaluating every flag, then picking a winner.** It produces the right tier and no
#   explanation, because nothing decided anything. Ask in order and return early.
# - **Checking profiling after the conditions.** A profiling system that meets 6(3)(b) with
#   perfect paperwork is still high-risk. Put the override where the article puts it.
# - **Treating `annex_iii_derogated` as `minimal`.** Claiming the derogation creates duties:
#   document the assessment before placing, and register under Article 49(2).
# - **Accepting a justification of whitespace.** `if system.get("justification")` is True for
#   `"   "`. Strip it.
# - **Documenting on the day of placing.** Article 6(4) says *before*. The comparison is
#   strict, and a non-strict one silently passes the most common real failure.
# - **Reading the Article 50 duties off the tier.** They are properties of the system. A
#   high-risk system that also chats owes both, on two different dates.
# - **Breaking a tie with `Counter.most_common()`.** Its order depends on insertion, so the
#   verdict depends on which analyst filled the form in first. Break ties by severity.
# - **Hashing the record with `record_hash` already in it.** Then nobody can recompute it.
# - **Deriving `changed_answers` from the obligation diff.** They are different things. A
#   question can answer identically while an obligation vanishes — `q6_transparency` records
#   the OR of two flags, so one of them can flip without the trail moving at all. Report what
#   the trail says, and let the obligation diff say the rest.
# - **Diffing two systems.** Without the id check the report looks plausible and means nothing.

# %% [markdown]
# ## 12. Self-check
#
# 1. An Annex III recruitment system performs profiling of candidates, meets Article 6(3)(b),
#    poses no significant risk on the provider's own assessment, and has documented and
#    registered everything. What is it?
#    - (a) not high-risk: it met a condition and documented it
#    - (b) high-risk: profiling of natural persons overrides the derogation entirely
#    - (c) not high-risk until an authority objects
#    - (d) high-risk only if the documentation is later found wanting
#
# 2. A provider correctly concludes under Article 6(3) that its Annex III system is not
#    high-risk. What does it now owe?
#    - (a) nothing; the derogation removes the system from the Act
#    - (b) documentation of the assessment before placing on the market, and registration
#    - (c) the full Chapter III duties, from 2 December 2027
#    - (d) only an entry in its internal risk register
#
# 3. Your classifier answers all six questions for every system and then returns the most
#    severe tier any answer implies. What has it lost?
#    - (a) nothing; it is the same tier
#    - (b) speed only
#    - (c) the explanation — no single question decided, so the trail cannot name one
#    - (d) the ability to handle prohibited practices
#

# %%
# Before questions 4 and 5: two facts you can run rather than recall.
def _self_check_aids() -> None:
    report = disagreement_report(PANEL)
    print(f"pivot {report['pivot_question']}, reached by "
          f"{len(report['pivot_answers'])} of {len(PANEL)} analysts, "
          f"who between them reached {len(report['tiers'])} different tiers")
    quieter = dict(SYSTEMS["policy-drafter"], generates_synthetic_content=False)
    diff = diff_records(classification_record(SYSTEMS["policy-drafter"]),
                        classification_record(quieter))
    print(f"policy-drafter: tier_changed={diff['tier_changed']}, "
          f"vanished={sorted(diff['removed'])}, "
          f"requires_reassessment={diff['requires_reassessment']}")


_try("self-check aids", _self_check_aids)

# %% [markdown]
# 4. Four analysts describe one system and reach three different tiers. The most useful thing
#    to report is:
#    - (a) the range of tiers
#    - (b) the average severity
#    - (c) the earliest question in the procedure on which they gave different answers
#    - (d) the most severe tier anyone reached
#
# 5. A system's tier does not change, but one obligation vanishes from its record. What
#    follows?
#    - (a) nothing; the tier is what matters
#    - (b) the classification still has to be redone and re-documented — a tier-only diff
#          would have missed it
#    - (c) the record was tampered with
#    - (d) the obligation was never real
#
# Mark them in the next cell. The key is not written in this file — only a salted hash of it —
# so you find out which are wrong without reading the answers off the page. The reasoning for
# each is published in the course solution bundle.

# %%
# Salted hashes of the answers, not the answers. Nothing here tells you which letter is right.
_SELF_CHECK_KEY = {
    1: "1384ca6ba3efde65",
    2: "00d02e57d8c5ea91",
    3: "f5a5e04bdcceae3b",
    4: "353d24937d8f7203",
    5: "03e515490f779b2b",
}

_SELF_CHECK_HINT = {
    1: "re-read the second paragraph of section 4, and look at what _check_derogation asserts "
       "about cv-ranker.",
    2: "run exercise 3 against shift-note-tidier and read the obligation list it prints.",
    3: "re-read the two properties named at the top of section 5.",
    4: "run the panel demo in section 10 and read the line under 'pivot'.",
    5: "look at the last assertion in _check_diff, about policy-drafter.",
}


def check_self_check(answers: dict) -> None:
    """Mark your self-check answers. Pass a dict of question number -> letter.

    Example:
        >>> check_self_check({1: "a"})          # doctest: +SKIP
          q1  not 'a' — re-read the second paragraph of section 4 ...
          q2  no answer given
        ...
    """
    right = 0
    for question in sorted(_SELF_CHECK_KEY):
        given = str(answers.get(question, "")).strip().lower()
        digest = hashlib.sha256(f"P01-L02:q{question}:{given}".encode()).hexdigest()[:16]
        if digest == _SELF_CHECK_KEY[question]:
            right += 1
            print(f"  q{question}  correct")
        elif not given:
            print(f"  q{question}  no answer given")
        else:
            print(f"  q{question}  not {given!r} — {_SELF_CHECK_HINT[question]}")
    print(f"\n{len(_SELF_CHECK_KEY)} questions, {right} right")


# Put your own letters in, then run this cell:
# check_self_check({1: "a", 2: "a", 3: "a", 4: "a", 5: "a"})

# %% [markdown]
# ## What you built, and where it goes next
#
# A decision procedure that returns its own reasoning, a derogation with the override and the
# paperwork duty that really attach to it, a sealed record, a diff that prices a system
# change in obligations and dates, and a panel report that names the one question worth
# arguing about first.
#
# That record is the `risk_classification_record` evidence id in the pack, and it is the input
# to module 8: the conformity assessment route is a decision procedure over exactly this
# object. Append each record to your P01-L01 log — the digest is already computed — and the
# classification history becomes as tamper-evident as the decision history.
#
# **Again, and finally: this is engineering, not legal advice.**

# %%
if __name__ == "__main__":
    for _name, _check in (("exercise 1", _check_derogation),
                          ("exercise 2", _check_ask_tier),
                          ("exercise 3", _check_obligations),
                          ("exercise 4", _check_record),
                          ("exercise 5", _check_diff),
                          ("exercise 6", _check_disagreement)):
        _try(_name, _check)
    # A stub nobody has reached yet is not a failure. A check that ran and came back wrong is,
    # and it ends this run non-zero rather than letting a green exit code paper over it.
    if _FAILED_CHECKS:
        raise SystemExit("checks failed: " + ", ".join(dict.fromkeys(_FAILED_CHECKS)))
