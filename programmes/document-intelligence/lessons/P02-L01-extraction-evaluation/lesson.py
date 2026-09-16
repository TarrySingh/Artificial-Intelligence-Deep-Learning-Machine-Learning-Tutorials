# %% [markdown]
# # P02-L01 · Field extraction you can measure
#
# **You will build:** the evaluation harness for a field-extraction system — a normaliser, a
# per-field precision/recall/F1 scorer, a confusion analysis over field types, and a review
# queue that spends a fixed human budget where it buys the most quality.
#
# **Time:** ~55 minutes · **Runs on:** a laptop CPU, 8 GiB RAM, no GPU, no download, no model
# API · **Prerequisites:** T00-L01 (the 8 GB track).
#
# There is no PDF library in this environment and no extractor worth the name. That is the
# point. The harness is the artefact that outlives every extractor you will put behind it, and
# the only one that can tell you whether the next extractor is an improvement or a relabelling.
#
# By the end you will be able to:
#
# 1. Implement `normalise_value` so "the same value" is a written policy, per field type.
# 2. Implement exact, normalised and fuzzy matching, and confine fuzzy to where it is safe.
# 3. Implement a per-field precision/recall/F1 scorer that counts a wrong value as both an FP
#    and an FN.
# 4. Build a confusion analysis over field types that turns one F1 number into a work order.
# 5. Implement a confidence-ordered review queue and measure the quality it buys per unit cost.
# 6. Explain why the harness is built before the extractor, not after it.

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import random
import re
import sys
import time
from typing import Iterable, Mapping, NamedTuple, Sequence

import numpy as np

_LESSON_T0 = time.perf_counter()
print("python", sys.version.split()[0], "· numpy", np.__version__)
print("no PDF library, no OCR engine, no model API — and none of that is needed to build")
print("the instrument that decides whether an extractor is fit to ship.\n")

# The schema: the six fields this programme's first corpus is annotated for, and the TYPE of
# each one. Field type is the unit of policy here — money is normalised and compared one way,
# free text another — so it travels with the field everywhere in this lesson.
SCHEMA: dict[str, str] = {
    "invoice_id": "id",
    "invoice_date": "date",
    "total_amount": "money",
    "currency": "id",
    "counterparty": "text",
    "payment_terms_days": "integer",
}

# The five things that can happen to one (document, field) cell. Every cell lands in exactly
# one of them, which is what makes the confusion table add up.
ERROR_LABELS = ("correct", "miss", "spurious", "wrong_value", "true_negative")

MATCH_MODES = ("exact", "normalised", "fuzzy")

# Character-level similarity at or above this counts as a fuzzy match. Section 3 measures what
# this number costs you when it is applied to the wrong field type.
FUZZY_THRESHOLD = 0.85

MONTH_NAMES = ("January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December")
MONTH_INDEX = {name.lower(): i + 1 for i, name in enumerate(MONTH_NAMES)}

# Legal-form tokens that carry no identifying information. "Nordwind Logistik GmbH" and
# "Nordwind Logistik" are the same counterparty; a matcher that disagrees invents a defect.
COMPANY_SUFFIXES = frozenset({
    "gmbh", "bv", "nv", "ag", "kg", "ltd", "limited", "inc", "incorporated",
    "plc", "llc", "sa", "sas", "srl", "spa", "oy", "ab", "as", "co",
})


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings: insertions, deletions and substitutions, cost 1."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def similarity(a: str, b: str) -> float:
    """1.0 for identical strings, 0.0 for nothing in common. Edit distance over the longer one."""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / longest if longest else 1.0


class FieldScore(NamedTuple):
    """What one field scored over the whole corpus, under one matching mode."""
    field: str
    mode: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


_FAILED_CHECKS: list[str] = []


def _try(label: str, check) -> None:
    """Run a check, or a demo that depends on your code, without derailing the notebook.

    A stub you have not filled in yet simply says so. A wrong answer prints the check's own
    message — which names the likely mistake — and the notebook carries on to the next cell,
    so one broken exercise never hides the feedback on the other five.
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
# ## 1. The corpus, and why it is synthetic on purpose
#
# There is no PDF reader in this environment. That is the lesson's starting condition, not an
# apology: the harness you are about to build has to exist before any extractor does, and it
# is cheaper to prove that on documents whose ground truth you generated than on a scanned
# stack whose ground truth you would have to pay someone to produce.
#
# `build_corpus()` writes short remittance advices and stores, for each one, the exact string
# a human annotator would have typed into the label field. Run it and read one document.

# %%
_COMPANIES = (
    "Nordwind Logistik GmbH", "Vantor Marine B.V.", "Helix Pharma Limited",
    "Caldera Energy PLC", "Brightwater Analytics Ltd", "Orsini Costruzioni SRL",
    "Kestrel Freight Inc", "Aalto Terveys Oy", "Meridian Custody AG",
    "Sable & Finch LLP", "Dunbar Reinsurance Ltd", "Petrarca Chimica SpA",
    "Lindqvist Verkstad AB", "Hollandse Kaasunie N.V.", "Argent Clearing SA",
    "Torrent Robotics Inc", "Vesper Maritime AS", "Kaneko Precision KK",
)
_CURRENCIES = {"EUR": "€", "USD": "$", "GBP": "£"}


def _surface_date(year: int, month: int, day: int, style: int) -> str:
    """The same date as an annotator would have found it printed on the page."""
    if style == 0:
        return f"{year:04d}-{month:02d}-{day:02d}"
    if style == 1:
        return f"{day:02d}/{month:02d}/{year:04d}"       # day first: the corpus is European
    return f"{day} {MONTH_NAMES[month - 1]} {year}"


def _surface_money(cents: int, currency: str, style: int) -> str:
    whole, part = divmod(cents, 100)
    if style == 0:
        return f"{_CURRENCIES[currency]}{whole:,}.{part:02d}"
    if style == 1:
        return f"{whole}.{part:02d} {currency}"
    return f"{currency} {whole:,}.{part:02d}"


def _surface_id(year: int, serial: int, style: int) -> str:
    if style == 0:
        return f"INV-{year}-{serial:04d}"
    if style == 1:
        return f"INV/{year}/{serial:04d}"
    return f"Inv {year} {serial:04d}"


def build_corpus(n_docs: int = 180, seed: int = 20260916) -> list[dict]:
    """Deterministic corpus of short documents, each with the gold values an annotator typed.

    Gold is the SURFACE string, exactly as it appears on the page — that is what a human
    labeller gives you. An empty string means the field is genuinely absent from the document,
    which is different from an extractor failing to find it.
    """
    rng = random.Random(seed)
    corpus = []
    for i in range(n_docs):
        year = rng.choice((2025, 2026))
        month, day, serial = rng.randint(1, 12), rng.randint(1, 28), rng.randint(1, 9999)
        currency = rng.choice(tuple(_CURRENCIES))
        cents = rng.randint(1_50, 480_000_00)
        company = rng.choice(_COMPANIES)
        terms = rng.choice(("14", "30", "30", "45", "60", "", ""))  # 2 in 7 say nothing
        d_style, m_style, i_style = rng.randrange(3), rng.randrange(3), rng.randrange(3)
        gold = {
            "invoice_id": _surface_id(year, serial, i_style),
            "invoice_date": _surface_date(year, month, day, d_style),
            "total_amount": _surface_money(cents, currency, m_style),
            "currency": currency,
            "counterparty": company,
            "payment_terms_days": terms,
        }
        terms_line = f"Payment terms: net {terms} days\n" if terms else ""
        text = (
            f"REMITTANCE ADVICE\n"
            f"Invoice {gold['invoice_id']}\n"
            f"Issued {gold['invoice_date']}\n"
            f"Supplier: {company}\n"
            f"{terms_line}"
            f"Amount due: {gold['total_amount']}\n"
        )
        corpus.append({"doc_id": f"DOC-{i:04d}", "text": text, "gold": gold})
    return corpus


CORPUS = build_corpus()
print(f"{len(CORPUS)} documents, {len(SCHEMA)} fields each "
      f"= {len(CORPUS) * len(SCHEMA)} cells to be judged\n")
print(CORPUS[3]["text"])
print("gold labels an annotator typed for that document:")
for _f, _v in CORPUS[3]["gold"].items():
    print(f"  {_f:20s} {_v!r}")
_absent = sum(1 for d in CORPUS if not d["gold"]["payment_terms_days"])
print(f"\n{_absent} of {len(CORPUS)} documents state no payment terms at all — "
      f"for those, the right answer is nothing.")

# %% [markdown]
# ## 2. The extractor, and why it is not your problem today
#
# `run_extractor()` reads each document and returns a predicted value and a confidence per
# field. It is deliberately mediocre in the ways real systems are mediocre: it reformats dates
# and amounts into its own house style, it drops fields, it invents payment terms that were
# never on the page, it mangles a supplier name the way an OCR pass does, and now and then it
# transposes two digits in an amount. Run it and read one prediction.

# %%
def _reformat(value: str, field_type: str) -> str:
    """The extractor's house style: same meaning, different surface. Exact match hates this."""
    if field_type == "date":
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
        if m:
            return f"{int(m.group(3))}/{int(m.group(2))}/{m.group(1)}"
        m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", value)
        if m:
            return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        m = re.match(r"^(\d{1,2}) (\w+) (\d{4})$", value)
        if m:
            return f"{m.group(3)}-{MONTH_INDEX[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
        return value
    if field_type == "money":
        digits = re.sub(r"[^\d.]", "", value.replace(",", ""))
        return digits
    if field_type == "id":
        return re.sub(r"[^A-Za-z0-9]", "", value).upper()
    if field_type == "text":
        return value.replace(" B.V.", "").replace(" GmbH", "").replace(" Ltd", "")
    if field_type == "integer":
        return f"net {value}" if value else value
    return value


def _ocr_noise(value: str, rng: random.Random) -> str:
    """One substituted character, the way a scan smudges a letter."""
    letters = [i for i, c in enumerate(value) if c.isalpha()]
    if not letters:
        return value
    i = rng.choice(letters)
    chars = list(value)
    chars[i] = rng.choice("aeoirnmcl")
    return "".join(chars)


def _corrupt(value: str, field_type: str, rng: random.Random) -> str:
    """A genuinely wrong answer: same shape, different meaning. This is the dangerous one."""
    if field_type == "money":
        # One substituted digit: the classic scan error, and the one that survives every
        # sanity check a human would apply to the total at a glance.
        digits = [i for i, c in enumerate(value) if c.isdigit()]
        if digits:
            i = rng.choice(digits)
            chars = list(value)
            chars[i] = rng.choice([d for d in "0123456789" if d != chars[i]])
            return "".join(chars)
        return value + "1"
    if field_type in {"integer", "id"}:
        digits = [i for i, c in enumerate(value) if c.isdigit()]
        if len(digits) >= 2:
            i, j = rng.sample(digits, 2)
            chars = list(value)
            chars[i], chars[j] = chars[j], chars[i]
            if "".join(chars) != value:
                return "".join(chars)
        return value + "1"
    if field_type == "date":
        m = re.search(r"\b(\d{1,2})\b", value)
        if m:
            bumped = str((int(m.group(1)) % 28) + 1).zfill(len(m.group(1)))
            return value[:m.start(1)] + bumped + value[m.end(1):]
        return value
    return rng.choice([c for c in _COMPANIES if c != value])


def run_extractor(corpus: Sequence[Mapping], seed: int = 7) -> list[dict]:
    """Predictions and confidences for every (document, field) cell. Deterministic.

    The error mix is drawn per document from a fixed seed, so every student sees the same
    corpus, the same mistakes and the same numbers. Confidence is informative but not
    reliable — high on most correct cells, lower on most wrong ones, with real overlap.
    """
    records = []
    for doc in corpus:
        rng = random.Random(seed * 100_003 + int(doc["doc_id"][4:]))
        pred, conf = {}, {}
        for field, field_type in SCHEMA.items():
            truth = doc["gold"][field]
            roll = rng.random()
            if not truth:
                if roll < 0.22:                       # invents terms that were never printed
                    pred[field] = rng.choice(("30", "14", "60"))
                    conf[field] = round(rng.uniform(0.30, 0.72), 3)
                else:
                    pred[field], conf[field] = "", round(rng.uniform(0.80, 0.98), 3)
                continue
            if roll < 0.09:                           # found nothing
                pred[field], conf[field] = "", 0.0
            elif roll < 0.16:                         # genuinely wrong value
                pred[field] = _corrupt(truth, field_type, rng)
                conf[field] = round(rng.uniform(0.34, 0.86), 3)
            elif roll < 0.23 and field_type == "text":  # smudged scan
                pred[field] = _ocr_noise(truth, rng)
                conf[field] = round(rng.uniform(0.40, 0.78), 3)
            elif roll < 0.72:                         # right answer, house-style surface
                pred[field] = _reformat(truth, field_type)
                conf[field] = round(rng.uniform(0.62, 0.95), 3)
            else:                                     # right answer, verbatim
                pred[field] = truth
                conf[field] = round(rng.uniform(0.86, 0.99), 3)
        records.append({"doc_id": doc["doc_id"], "gold": dict(doc["gold"]),
                        "pred": pred, "conf": conf})
    return records


RECORDS = run_extractor(CORPUS)
print("document DOC-0003, gold vs predicted vs confidence:")
for _f in SCHEMA:
    _r = RECORDS[3]
    print(f"  {_f:20s} gold={_r['gold'][_f]!r:28s} pred={_r['pred'][_f]!r:28s} "
          f"conf={_r['conf'][_f]}")
print("\nSome of those differ only in formatting. Some of them are wrong. Nothing in this")
print("output tells you which is which — that is the instrument you are about to build.")

# %% [markdown]
# ## 3. Exercise 1 — `normalise_value`
#
# A matcher that compares raw strings will report the extractor's house style as a defect.
# Normalisation is where you write down, per field type, what "the same value" means. It is a
# policy decision, so it belongs in code your reviewers can read and argue with.
#
# It is also where the published evidence puts the difficulty. A May 2026 comparison of
# frontier LLMs against domain-trained models on structured contract extraction found
# performance strongest on short-text identifiers and weakest on currency fields requiring
# normalisation or aggregation (arXiv 2605.05532). Normalisation is not a tidying step you
# bolt on at the end; it is where a third of your reported defects come from.

# %%
def normalise_value(value: str, field_type: str) -> str:
    """Canonical form of `value` for its `field_type`. Empty in, empty out.

    Rules, one per field type:

    * ``money``    → plain digits, always two decimals: ``"€1,234.50"`` and ``"1234.5"`` and
                     ``"EUR 1.234,50"`` all become ``"1234.50"``. When both a comma and a dot
                     appear, the LAST of the two is the decimal separator and the other is a
                     thousands separator. When only a comma appears it is a decimal comma if
                     exactly two digits follow it, and a thousands separator otherwise.
    * ``date``     → ISO ``YYYY-MM-DD``. The corpus is European: ``"04/03/2026"`` is the
                     fourth of March, not the third of April. ``"4 March 2026"`` parses too.
    * ``integer``  → the first run of digits, as a plain int: ``"net 30 days"`` → ``"30"``.
    * ``id``       → upper case, every non-alphanumeric character dropped.
    * ``text``     → lower case, punctuation to spaces, whitespace collapsed, then trailing
                     legal-form tokens in ``COMPANY_SUFFIXES`` dropped.

    A value that does not parse is NOT silently discarded: fall back to the raw string, upper
    cased with whitespace collapsed, so a normaliser can never invent or destroy a value.
    An empty or whitespace-only value normalises to the empty string. A `field_type` that is
    not one of the types in `SCHEMA` is a bug in the caller, so raise ``ValueError``.

    Examples:
        >>> normalise_value("€1,234.50", "money")
        '1234.50'
        >>> normalise_value("04/03/2026", "date")
        '2026-03-04'
        >>> normalise_value("Inv 2026 0042", "id")
        'INV20260042'
        >>> normalise_value("Vantor Marine B.V.", "text")
        'vantor marine'
        >>> normalise_value("net 30 days", "integer")
        '30'
        >>> normalise_value("  ", "money")
        ''
        >>> normalise_value("on receipt", "integer")
        'ON RECEIPT'
    """
    # YOUR CODE HERE
    raise NotImplementedError


# Public checks — run these as often as you like.
def _check_normalise() -> None:
    money = {"€1,234.50": "1234.50", "1234.5": "1234.50", "EUR 1,234.50": "1234.50",
             "1.234,50": "1234.50", "1234.50 USD": "1234.50"}
    for raw, want in money.items():
        got = normalise_value(raw, "money")
        assert got == want, (f"normalise_value({raw!r}, 'money') gave {got!r}, expected {want!r}"
                             " — strip the symbol, decide what a comma means, then format to 2dp")
    dates = {"2026-03-04": "2026-03-04", "04/03/2026": "2026-03-04", "4 March 2026": "2026-03-04"}
    for raw, want in dates.items():
        got = normalise_value(raw, "date")
        assert got == want, (f"normalise_value({raw!r}, 'date') gave {got!r}, expected {want!r}"
                             " — this corpus is day-first, so 04/03 is 4 March")
    assert normalise_value("Inv 2026 0042", "id") == "INV20260042", "id: upper case, alnum only"
    assert normalise_value("Nordwind Logistik GmbH", "text") == "nordwind logistik", \
        "text: drop the trailing legal form, it identifies nobody"
    assert normalise_value("net 30 days", "integer") == "30", "integer: first run of digits"
    assert normalise_value("   ", "date") == "", "whitespace only must normalise to empty"
    assert normalise_value("on receipt", "integer") == "ON RECEIPT", \
        "an unparseable value falls back to the raw string, upper cased — never to empty"
    print("exercise 1 looks right")


_try("exercise 1", _check_normalise)

# %% [markdown]
# Run this to see what normalisation is worth before you have scored anything. It counts the
# cells where predicted and gold differ as raw strings but agree once normalised.

# %%
def _show_surface_gap() -> None:
    raw_equal = norm_equal = 0
    for record in RECORDS:
        for field, field_type in SCHEMA.items():
            p, g = record["pred"][field], record["gold"][field]
            if not p or not g:
                continue
            raw_equal += p == g
            norm_equal += normalise_value(p, field_type) == normalise_value(g, field_type)
    both = sum(1 for r in RECORDS for f in SCHEMA if r["pred"][f] and r["gold"][f])
    print(f"of {both} cells where both sides said something:")
    print(f"  identical as raw strings : {raw_equal:4d}  ({100 * raw_equal / both:.1f}%)")
    print(f"  identical once normalised: {norm_equal:4d}  ({100 * norm_equal / both:.1f}%)")
    print(f"\n{norm_equal - raw_equal} cells are the same value in a different costume.")
    print("Report the first number as accuracy and you will send an engineer to fix a")
    print("model that was already right.")


_try("surface gap", _show_surface_gap)

# %% [markdown]
# ## 4. Exercise 2 — `match_value`
#
# Three modes, one function. `exact` is raw string equality. `normalised` compares canonical
# forms. `fuzzy` additionally accepts near-identical **text**, and nothing else: an amount that
# is one digit out looks 86% similar to the right answer and is a payment incident.

# %%
def match_value(pred: str, gold: str, field_type: str, mode: str) -> bool:
    """Does `pred` count as a match for `gold` under `mode`?

    * An empty side never matches. "Nothing" is not a value; a cell where either side is
      empty is counted elsewhere, as a miss or as a spurious extraction.
    * ``exact``      → raw string equality.
    * ``normalised`` → equality of ``normalise_value`` on both sides.
    * ``fuzzy``      → normalised equality, OR, for ``text`` fields only,
      ``similarity`` of the normalised forms at or above ``FUZZY_THRESHOLD``.

    An unknown mode is a programming error, not a non-match: raise ``ValueError``.

    Examples:
        >>> match_value("€1,234.50", "1234.50", "money", "exact")
        False
        >>> match_value("€1,234.50", "1234.50", "money", "normalised")
        True
        >>> match_value("Nordwind Logistlk", "Nordwind Logistik GmbH", "text", "fuzzy")
        True
        >>> match_value("1284.50", "1234.50", "money", "fuzzy")
        False
        >>> match_value("", "1234.50", "money", "normalised")
        False
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_match() -> None:
    assert match_value("€1,234.50", "1234.50", "money", "normalised"), \
        "normalised mode must compare canonical forms, not raw strings"
    assert not match_value("€1,234.50", "1234.50", "money", "exact"), \
        "exact mode is raw string equality — no normalising allowed"
    assert match_value("Nordwind Logistlk", "Nordwind Logistik GmbH", "text", "fuzzy"), \
        "one smudged character in a supplier name is what fuzzy mode is for"
    assert not match_value("1284.50", "1234.50", "money", "fuzzy"), (
        "fuzzy mode must refuse money: 1284.50 and 1234.50 are 86% similar and 50 currency "
        "units apart. Gate fuzzy on field_type == 'text'.")
    assert not match_value("INV20260041", "INV20260042", "id", "fuzzy"), \
        "fuzzy mode must refuse ids too — a near-identical id is a different invoice"
    for mode in MATCH_MODES:
        assert not match_value("", "x", "text", mode), "an empty prediction never matches"
        assert not match_value("x", "", "text", mode), "an empty gold never matches"
    try:
        match_value("a", "a", "text", "approximately")
        raise AssertionError("an unknown mode must raise ValueError, not return a verdict")
    except ValueError:
        pass
    print("exercise 2 looks right")


_try("exercise 2", _check_match)

# %% [markdown]
# ## 5. Exercise 3 — `score_field`
#
# Now the scorer. One field, one mode, the whole corpus. The convention that matters:
# **a wrong value is both a false positive and a false negative.** You claimed something that
# was not true (precision) and you failed to produce the value that was (recall). Counting it
# once flatters whichever number you happened to put in the deck.

# %%
def score_field(records: Sequence[Mapping], field: str, mode: str) -> FieldScore:
    """Precision, recall and F1 for one field over `records`, under one matching mode.

    Per (document, field) cell, with `p` the prediction and `g` the gold value:

    * both non-empty and matching → one true positive
    * prediction non-empty, and either gold is empty or it does not match → false positive
    * gold non-empty, and either prediction is empty or it does not match → false negative
    * both empty → nothing; the extractor was right to stay quiet

    A wrong value therefore scores one FP *and* one FN. Guard the divisions: with no
    predictions at all precision is 0.0, not a crash, and F1 is 0.0 when both sides are 0.0.

    Example:
        >>> rows = [{"doc_id": "D1", "gold": {"currency": "EUR"}, "pred": {"currency": "EUR"}},
        ...         {"doc_id": "D2", "gold": {"currency": "USD"}, "pred": {"currency": ""}}]
        >>> s = score_field(rows, "currency", "normalised")
        >>> (s.tp, s.fp, s.fn, round(s.f1, 3))
        (1, 0, 1, 0.667)
    """
    # YOUR CODE HERE
    raise NotImplementedError


def macro_f1(records: Sequence[Mapping], mode: str) -> float:
    """Unweighted mean of the per-field F1 scores over every field in `SCHEMA`.

    Unweighted on purpose: `payment_terms_days` appears on fewer documents than
    `total_amount`, and a macro average refuses to let the common fields drown out the rare
    one. Return 0.0 for an empty schema rather than dividing by zero.

    Example:
        >>> rows = [{"doc_id": "D1", "gold": {f: "" for f in SCHEMA},
        ...          "pred": {f: "" for f in SCHEMA}}]
        >>> macro_f1(rows, "normalised")
        0.0
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_score() -> None:
    rows = [
        {"doc_id": "D1", "gold": {"currency": "EUR"}, "pred": {"currency": "eur"}},   # tp
        {"doc_id": "D2", "gold": {"currency": "USD"}, "pred": {"currency": ""}},      # fn
        {"doc_id": "D3", "gold": {"currency": ""}, "pred": {"currency": "GBP"}},      # fp
        {"doc_id": "D4", "gold": {"currency": "EUR"}, "pred": {"currency": "USD"}},   # fp + fn
        {"doc_id": "D5", "gold": {"currency": ""}, "pred": {"currency": ""}},         # nothing
    ]
    s = score_field(rows, "currency", "normalised")
    assert (s.tp, s.fp, s.fn) == (1, 2, 2), (
        f"expected tp=1 fp=2 fn=2, got tp={s.tp} fp={s.fp} fn={s.fn} — a wrong value (D4) is "
        "BOTH a false positive and a false negative, and D5 is neither")
    assert abs(s.precision - 1 / 3) < 1e-9, f"precision should be tp/(tp+fp), got {s.precision}"
    assert abs(s.recall - 1 / 3) < 1e-9, f"recall should be tp/(tp+fn), got {s.recall}"
    empty = score_field([{"doc_id": "D", "gold": {"currency": ""}, "pred": {"currency": ""}}],
                        "currency", "exact")
    assert empty.precision == 0.0 and empty.f1 == 0.0, \
        "no predictions and no gold: return 0.0, do not divide by zero"
    print("exercise 3 looks right")


_try("exercise 3", _check_score)

# %% [markdown]
# Now the first table worth showing anyone: the same extractor, the same corpus, judged three
# ways. Nothing about the model changes between these rows.

# %%
def _show_mode_table() -> None:
    print(f"{'field':22s}{'type':9s}" + "".join(f"{m:>12s}" for m in MATCH_MODES))
    for field, field_type in SCHEMA.items():
        cells = "".join(f"{score_field(RECORDS, field, m).f1:12.3f}" for m in MATCH_MODES)
        print(f"{field:22s}{field_type:9s}{cells}")
    print(f"{'MACRO F1':22s}{'':9s}" + "".join(f"{macro_f1(RECORDS, m):12.3f}" for m in MATCH_MODES))
    gain = macro_f1(RECORDS, "normalised") - macro_f1(RECORDS, "exact")
    print(f"\nnormalisation alone moved macro F1 by {gain:+.3f} without touching the extractor.")
    print("fuzzy moves only the text field, because you confined it to the text field.")


_try("mode table", _show_mode_table)

# %% [markdown]
# ## 6. Exercise 4 — the confusion analysis
#
# An F1 of 0.84 does not tell you what to fix. Four hundred defects split as *mostly misses*
# and *mostly wrong values* demand opposite responses: the first is a recall problem you solve
# with a better reader, the second is a precision problem you solve by making the model abstain.

# %%
def classify_cell(pred: str, gold: str, field_type: str, mode: str) -> str:
    """Label one (document, field) cell as exactly one of `ERROR_LABELS`.

    * ``"true_negative"`` — both empty. The field is not on the page and nothing was claimed.
    * ``"miss"``          — gold has a value, the extractor produced nothing.
    * ``"spurious"``      — the extractor produced a value for a field that is not on the page.
    * ``"correct"``       — both present and matching under `mode`.
    * ``"wrong_value"``   — both present and not matching.

    Examples:
        >>> classify_cell("", "30", "integer", "normalised")
        'miss'
        >>> classify_cell("30", "", "integer", "normalised")
        'spurious'
        >>> classify_cell("net 30", "30", "integer", "normalised")
        'correct'
    """
    # YOUR CODE HERE
    raise NotImplementedError


def confusion_by_field_type(records: Sequence[Mapping], mode: str) -> dict[str, dict[str, int]]:
    """Counts of every label in `ERROR_LABELS`, grouped by field TYPE rather than field name.

    Every field type in `SCHEMA` appears as a key, and every label in `ERROR_LABELS` appears
    under it, zero included — a table with holes in it cannot be read across rows. Two fields
    of the same type (`invoice_id` and `currency` are both ``id``) add into the same row.

    Example:
        >>> table = confusion_by_field_type([], "normalised")
        >>> sorted(table) == sorted(set(SCHEMA.values()))
        True
        >>> table["money"]["miss"]
        0
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_confusion() -> None:
    assert classify_cell("", "", "text", "exact") == "true_negative", "both empty is not an error"
    assert classify_cell("", "30", "integer", "normalised") == "miss", "gold only is a miss"
    assert classify_cell("30", "", "integer", "normalised") == "spurious", "pred only is spurious"
    assert classify_cell("net 30", "30", "integer", "normalised") == "correct", \
        "classify_cell must route the both-present case through match_value"
    assert classify_cell("45", "30", "integer", "normalised") == "wrong_value", \
        "both present and not matching is wrong_value, not a miss"
    table = confusion_by_field_type([], "normalised")
    assert set(table) == set(SCHEMA.values()), f"one row per field TYPE, got {sorted(table)}"
    assert all(set(row) == set(ERROR_LABELS) for row in table.values()), \
        "every row carries every label, zeros included"
    real = confusion_by_field_type(RECORDS, "normalised")
    total = sum(sum(row.values()) for row in real.values())
    assert total == len(RECORDS) * len(SCHEMA), (
        f"the table must account for every cell: {total} counted, "
        f"{len(RECORDS) * len(SCHEMA)} exist")
    print("exercise 4 looks right")


_try("exercise 4", _check_confusion)

# %% [markdown]
# Read the table below as a work order, not a score.

# %%
def _show_confusion() -> None:
    table = confusion_by_field_type(RECORDS, "normalised")
    header = f"{'field type':12s}" + "".join(f"{lab:>14s}" for lab in ERROR_LABELS)
    print(header)
    for field_type in sorted(table):
        row = table[field_type]
        print(f"{field_type:12s}" + "".join(f"{row[lab]:14d}" for lab in ERROR_LABELS))
    worst = max(table, key=lambda ft: table[ft]["wrong_value"])
    leaky = max(table, key=lambda ft: table[ft]["spurious"])
    print(f"\nmost wrong values: {worst!r} ({table[worst]['wrong_value']}) — a silent-defect")
    print("problem; the extractor is confidently producing a value that is not on the page.")
    print(f"most spurious: {leaky!r} ({table[leaky]['spurious']}) — an abstention problem;")
    print("the field is often absent and the extractor answers anyway.")


_try("confusion table", _show_confusion)

# %% [markdown]
# ## 7. Exercise 5 — the review queue
#
# You cannot review everything. Given a budget of *B* cells a human can check in a shift, the
# policy decides which *B*. Sort by confidence ascending, break ties on `(doc_id, field)` so
# two runs of the same policy queue the same work, and stop at the budget.
#
# Where a document pipeline falls inside one of the EU AI Act's high-risk classes, this queue
# is also the mechanism the law asks for: Article 14(1) requires a high-risk system to be
# designed so that it can be effectively overseen by natural persons while it is in use.
# Whether your particular pipeline is high risk depends on what it decides. Whether your
# oversight is real depends on whether you measured it, which is the next section.

# %%
def review_queue(records: Sequence[Mapping], budget: int) -> tuple[tuple[str, str], ...]:
    """The `(doc_id, field)` cells to send to a human, lowest confidence first.

    * Every cell of every record is a candidate, including the ones where the extractor
      predicted nothing — a confident silence and a missed field look identical from outside.
    * Ties break on ``(doc_id, field)`` ascending — document id first, then the field name
      alphabetically — so two runs of the same policy queue the same work in the same order.
    * A budget of 0 sends nothing; a budget larger than the corpus sends everything.
    * A negative budget is a bug in the caller: raise ``ValueError``.

    Example:
        >>> rows = [{"doc_id": "D2", "gold": {}, "pred": {}, "conf": {f: 0.5 for f in SCHEMA}},
        ...         {"doc_id": "D1", "gold": {}, "pred": {}, "conf": {f: 0.5 for f in SCHEMA}}]
        >>> review_queue(rows, 1)          # every confidence equal, so the tie-break decides
        (('D1', 'counterparty'),)
    """
    # YOUR CODE HERE
    raise NotImplementedError


def apply_reviews(records: Sequence[Mapping], routed: Iterable[tuple[str, str]]) -> list[dict]:
    """A NEW list of records with every routed cell corrected by a perfect human reviewer.

    A reviewed cell takes the gold value and a confidence of 1.0. Everything else is copied
    unchanged. `records` must come back untouched: the whole point of the exercise below is to
    score the same corpus at several budgets, and a function that edits its input in place
    makes every budget after the first one a measurement of the previous one.

    Example:
        >>> rows = [{"doc_id": "D1", "gold": {"currency": "EUR"},
        ...          "pred": {"currency": ""}, "conf": {"currency": 0.0}}]
        >>> after = apply_reviews(rows, [("D1", "currency")])
        >>> (after[0]["pred"]["currency"], rows[0]["pred"]["currency"])
        ('EUR', '')
    """
    # YOUR CODE HERE
    raise NotImplementedError


def _check_queue() -> None:
    rows = [{"doc_id": "D2", "gold": {f: "" for f in SCHEMA}, "pred": {f: "" for f in SCHEMA},
             "conf": {f: 0.5 for f in SCHEMA}},
            {"doc_id": "D1", "gold": {f: "" for f in SCHEMA}, "pred": {f: "" for f in SCHEMA},
             "conf": {f: 0.5 for f in SCHEMA}}]
    assert review_queue(rows, 1) == (("D1", "counterparty"),), (
        "all confidences equal, so the tie-break decides: (doc_id, field) ascending puts D1 "
        "before D2, and 'counterparty' first alphabetically among the fields")
    assert review_queue(rows, 0) == (), "a budget of 0 routes nothing"
    assert len(review_queue(rows, 10_000)) == len(rows) * len(SCHEMA), \
        "a budget larger than the corpus routes every cell, and no more"
    try:
        review_queue(rows, -1)
        raise AssertionError("a negative budget must raise ValueError")
    except ValueError:
        pass
    queue = review_queue(RECORDS, 40)
    confs = {(r["doc_id"], f): r["conf"][f] for r in RECORDS for f in SCHEMA}
    routed = [confs[c] for c in queue]
    assert routed == sorted(routed), "the queue must come out lowest confidence FIRST"
    assert max(routed) <= min(v for c, v in confs.items() if c not in set(queue)), \
        "every routed cell must be at least as uncertain as every cell you left behind"

    before = [{"doc_id": "D1", "gold": {"currency": "EUR"}, "pred": {"currency": ""},
               "conf": {"currency": 0.0}}]
    after = apply_reviews(before, [("D1", "currency")])
    assert after[0]["pred"]["currency"] == "EUR", "a reviewed cell takes the gold value"
    assert after[0]["conf"]["currency"] == 1.0, "a reviewed cell is certain afterwards"
    assert before[0]["pred"]["currency"] == "", (
        "apply_reviews must not mutate its input — copy the dicts, or every later budget "
        "silently inherits the reviews of the earlier one")
    print("exercise 5 looks right")


_try("exercise 5", _check_queue)

# %% [markdown]
# ## 8. The quality/cost curve
#
# Two numbers you will be asked for in every review of this kind of system: what does it score,
# and what does a document cost. Both are standard axes: a March 2026 benchmark of multi-agent
# document pipelines scores systems on field-level F1, document-level accuracy, end-to-end
# latency, cost per document and token efficiency (arXiv 2603.22651). The two constants below
# are the only figures in this lesson
# you should replace with your own — they are placeholders for your organisation's rates, and
# nothing here claims they are typical of anything.

# %%
SECONDS_PER_REVIEWED_CELL = 40.0      # placeholder: time your own reviewers, do not guess
REVIEWER_COST_PER_HOUR = 30.0         # placeholder: your loaded cost, in your own currency
MACHINE_COST_PER_DOCUMENT = 0.004     # placeholder: your inference bill divided by volume


def cost_per_document(n_reviews: int, n_docs: int) -> float:
    """Machine cost plus human review cost, per document, from the placeholders above."""
    if n_docs <= 0:
        raise ValueError("n_docs must be positive")
    human = n_reviews * (SECONDS_PER_REVIEWED_CELL / 3600.0) * REVIEWER_COST_PER_HOUR
    return MACHINE_COST_PER_DOCUMENT + human / n_docs


def random_queue(records: Sequence[Mapping], budget: int, seed: int = 11) -> tuple:
    """The control policy: route `budget` cells chosen at random. Deterministic."""
    cells = [(r["doc_id"], f) for r in records for f in SCHEMA]
    rng = random.Random(seed)
    return tuple(rng.sample(cells, k=min(budget, len(cells))))


def _show_quality_cost_curve() -> None:
    n_docs, n_cells = len(RECORDS), len(RECORDS) * len(SCHEMA)
    budgets = [0, 60, 120, 240, 480, n_cells]
    print(f"{n_cells} cells total. Routing by confidence, against a random control.\n")
    print(f"{'budget':>8s}{'% cells':>9s}{'macro F1':>10s}{'random F1':>11s}"
          f"{'cost/doc':>10s}{'marginal F1/cost':>18s}")
    rows = []
    for budget in budgets:
        by_conf = macro_f1(apply_reviews(RECORDS, review_queue(RECORDS, budget)), "normalised")
        by_rand = macro_f1(apply_reviews(RECORDS, random_queue(RECORDS, budget)), "normalised")
        cost = cost_per_document(budget, n_docs)
        if rows:
            prev = rows[-1]
            marginal = (by_conf - prev[1]) / (cost - prev[3])
        else:
            marginal = 0.0
        rows.append((budget, by_conf, by_rand, cost, marginal))
        step = f"{marginal:18.3f}" if budget else f"{'—':>18s}"
        print(f"{budget:8d}{100 * budget / n_cells:8.1f}%{by_conf:10.3f}{by_rand:11.3f}"
              f"{cost:10.4f}{step}")

    best = max(rows[1:], key=lambda r: r[4])
    print(f"\nthe best marginal return on this corpus is the step ending at a budget of "
          f"{best[0]} cells ({best[4]:.3f} F1 per unit of cost per document), and every step "
          f"after it is worse.")

    # WHY the first step is not the best one, counted rather than asserted.
    first = review_queue(RECORDS, budgets[1])
    labels = {(r["doc_id"], f): classify_cell(r["pred"][f], r["gold"][f], SCHEMA[f],
                                              "normalised")
              for r in RECORDS for f in SCHEMA}
    head = [labels[c] for c in first]
    band = [labels[c] for c in review_queue(RECORDS, budgets[2])[len(first):]]
    print(f"first {len(head)} cells routed: " +
          ", ".join(f"{lab}={head.count(lab)}" for lab in ERROR_LABELS if head.count(lab)))
    print(f"next  {len(band)} cells routed: " +
          ", ".join(f"{lab}={band.count(lab)}" for lab in ERROR_LABELS if band.count(lab)))
    print("Fixing a miss converts one FN into one TP. Fixing a wrong value converts an FP AND")
    print("an FN into one TP, so it is worth strictly more — which is why the yield can climb")
    print("before it falls. Read across to the random column to see what the POLICY is worth,")
    print("as opposed to what the reviewers are worth.")


_try("quality/cost curve", _show_quality_cost_curve)

# %% [markdown]
# ## 9. Common mistakes
#
# - **Scoring exact string equality and calling it accuracy.** The mode table above shows the
#   gap. You will file bugs against a model that was already right.
# - **Counting a wrong value once.** It is a false positive and a false negative. Counting it
#   as one or the other inflates whichever of precision and recall you are being judged on.
# - **Letting fuzzy matching near an amount, a date or an identifier.** `1284.50` scores 0.86
#   against `1234.50`. A threshold of 0.85 turns a payment error into a green dashboard.
# - **Ignoring the empty-gold cells.** Fields that are legitimately absent are where spurious
#   extractions live, and a harness that skips them cannot see precision collapse.
# - **Reporting micro-averaged F1 only.** Pooling counts lets the fields that appear on every
#   page bury the rare field that the regulator actually asks about.
# - **A review queue that mutates the records it scores.** Every later budget then measures the
#   earlier one, and the curve bends upwards for free.
# - **Routing by field instead of by confidence.** Run the control column: a policy has to beat
#   random routing at the same budget before it is worth its complexity.
# - **Tuning the extractor before the harness exists.** You cannot tell an improvement from a
#   relabelling without a scorer you trust, and the scorer is the cheaper artefact.

# %% [markdown]
# The third one is worth seeing rather than believing. Run this.

# %%
def _show_what_fuzzy_money_would_cost() -> None:
    for p, g in (("1284.50", "1234.50"), ("1234.05", "1234.50"), ("2234.50", "1234.50")):
        sim = similarity(p, g)
        print(f"  similarity({p!r}, {g!r}) = {sim:.3f}   a 0.85 threshold accepts it: "
              f"{sim >= FUZZY_THRESHOLD}")
    waved = 0
    for record in RECORDS:
        p, g = record["pred"]["total_amount"], record["gold"]["total_amount"]
        if not p or not g:
            continue
        np_, ng_ = normalise_value(p, "money"), normalise_value(g, "money")
        if np_ != ng_ and similarity(np_, ng_) >= FUZZY_THRESHOLD:
            waved += 1
    print(f"\non this corpus, a matcher that fuzzed the money field would report {waved} "
          f"genuinely\nwrong amounts as correct — and your dashboard would go greener as it "
          f"did so.\nEach one is an invoice that does not reconcile.")


_try("fuzzy on money", _show_what_fuzzy_money_would_cost)

# %% [markdown]
# ## 10. Self-check
#
# 1. Exact-mode macro F1 is far below normalised-mode macro F1 on the same predictions. This
#    means:
#    - (a) the extractor got better between the two measurements
#    - (b) most of the disagreement was surface formatting, not wrong values
#    - (c) normalised mode is more lenient and therefore less trustworthy
#
# 2. A colleague proposes running fuzzy matching on `total_amount` "to stop penalising small
#    formatting differences". The strongest objection is:
#    - (a) it is slower, because edit distance is quadratic
#    - (b) formatting differences in amounts are already handled by normalisation, and fuzzy
#          matching would accept genuinely different amounts as correct
#    - (c) fuzzy matching only works on ASCII
#
# 3. `payment_terms_days` is absent from many documents and the extractor answers anyway. In
#    the confusion table those cells appear as:
#    - (a) misses, and they hurt recall
#    - (b) spurious extractions, and they hurt precision
#    - (c) true negatives, and they hurt nothing
#
# 4. In the quality/cost table, the F1 bought per unit of cost falls as the budget grows. The
#    right reading is:
#    - (a) human review stops working past a certain volume
#    - (b) confidence routing puts the cells most likely to be wrong at the front, so later
#          budget is spent re-checking cells that were already right
#    - (c) the reviewers get tired
#
# 5. You are handed a new extractor that scores 0.91 macro F1 where the old one scored 0.86.
#    Before shipping it, the first thing to check is:
#    - (a) whether the confusion table moved defects out of `wrong_value`, or only out of
#          `miss` — the two failures cost very different amounts downstream
#    - (b) the inference latency
#    - (c) nothing; 0.91 beats 0.86
#
# Answers are published in the course solution bundle.

# %% [markdown]
# One last cell: the scorecard. Five lines, every one of them computed, which is the shape of
# the summary a model-risk reviewer will ask you for.

# %%
def _show_scorecard() -> None:
    n_docs, n_cells = len(RECORDS), len(RECORDS) * len(SCHEMA)
    scores = {f: score_field(RECORDS, f, "normalised") for f in SCHEMA}
    worst = min(scores, key=lambda f: scores[f].f1)
    table = confusion_by_field_type(RECORDS, "normalised")
    defects = {lab: sum(row[lab] for row in table.values())
               for lab in ("miss", "spurious", "wrong_value")}
    dominant = max(defects, key=defects.get)
    best_budget, best_yield, previous = 0, 0.0, (0, macro_f1(RECORDS, "normalised"), 0.0)
    for budget in (60, 120, 240, 480, n_cells):
        f1 = macro_f1(apply_reviews(RECORDS, review_queue(RECORDS, budget)), "normalised")
        cost = cost_per_document(budget, n_docs)
        step = (f1 - previous[1]) / (cost - previous[2])
        if step > best_yield:
            best_budget, best_yield = budget, step
        previous = (budget, f1, cost)
    print(f"corpus                 {n_docs} documents, {n_cells} annotated cells")
    print(f"macro F1 (normalised)  {macro_f1(RECORDS, 'normalised'):.3f}")
    print(f"weakest field          {worst} (F1 {scores[worst].f1:.3f}, "
          f"{scores[worst].fn} false negatives)")
    print(f"dominant defect class  {dominant} ({defects[dominant]} of "
          f"{sum(defects.values())} defects)")
    print(f"recommended budget     {best_budget} cells "
          f"({100 * best_budget / n_cells:.0f}% of them), the highest-yield step measured "
          f"above,\n                       at {cost_per_document(best_budget, n_docs):.4f} "
          f"per document in the placeholder cost model")


_try("scorecard", _show_scorecard)

# %% [markdown]
# ## What you built, and where it goes next
#
# You now own the instrument the rest of this programme is measured with: a normaliser whose
# policy is readable, a scorer that refuses to count a wrong value once, a confusion table that
# names the defect class, and a routing policy that has to beat random before it earns its
# place. Every later module — table extraction, clause classification, drift monitoring — is
# scored by this harness rather than by a new one invented for the occasion.

# %%
if __name__ == "__main__":
    for _name, _check in (("exercise 1", _check_normalise),
                          ("exercise 2", _check_match),
                          ("exercise 3", _check_score),
                          ("exercise 4", _check_confusion),
                          ("exercise 5", _check_queue)):
        _try(_name, _check)
    print(f"\nlesson wall time so far: {time.perf_counter() - _LESSON_T0:.1f}s")
    # A stub you have not reached yet is not a failure — it prints "not implemented yet" and
    # the notebook carries on. A check that RAN and came back wrong is a failure, and it ends
    # this run non-zero rather than letting a green exit code paper over it.
    if _FAILED_CHECKS:
        raise SystemExit("checks failed: " + ", ".join(dict.fromkeys(_FAILED_CHECKS)))
