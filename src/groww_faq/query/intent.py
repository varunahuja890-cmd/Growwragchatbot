"""Phase 7 — rules-first intent classifier.

Runs before retrieval (PRD FR-5) so refusals never spend a retrieval or an
LLM generation. Checked in this order: advice, comparison, prediction,
portfolio, performance, otherwise factual. Two scheme names in one
question is a comparison even when the wording is mild.
"""

from __future__ import annotations

import re

from groww_faq.query.scheme import scheme_page_mentions

FACTUAL = "FACTUAL"
ADVICE = "ADVICE"
COMPARISON = "COMPARISON"
PREDICTION = "PREDICTION"
PORTFOLIO = "PORTFOLIO"
PERFORMANCE = "PERFORMANCE"

REFUSAL_CLASSES = (ADVICE, COMPARISON, PREDICTION, PORTFOLIO)

_ADVICE = [
    r"should i", r"shall i", r"should we", r"can i (invest|buy|sell)",
    r"do you recommend", r"recommend", r"\bsuggest", r"\badvise\b", r"\badvice\b",
    r"is it (a )?(good|great|wise|smart|bad)", r"is it worth", r"worth (it|investing)",
    r"is it safe", r"safe to", r"good option", r"right (fund|choice|scheme)",
    r"best pick", r"\bbuy\b", r"\bsell\b", r"\bhold\b", r"\bavoid\b",
]
_COMPARISON = [
    r"\bbetter\b", r"\bvs\b", r"\bversus\b", r"compar", r"difference between",
    r"outperform", r"which is (best|better)", r"which one",
]
_PREDICTION = [
    r"\bwill\b", r"next year", r"next \d+ years", r"in the future", r"in future",
    r"\bexpect", r"forecast", r"predict", r"how much will", r"growth rate",
    r"double my money", r"multiply my money",
]
_PORTFOLIO = [
    r"portfolio", r"allocat", r"diversif", r"my (investments?|money|savings)",
    r"how much should i (invest|put|allocate)",
]
_PERFORMANCE = [
    r"\breturns?\b", r"\bcagr\b", r"perform", r"\bpast\b", r"nav history",
    r"\bprofits?\b", r"\bgains?\b", r"\bloss(es)?\b", r"\byield\b",
    r"track record", r"how has it (done|fared|performed)",
]


def _matches(patterns: list[str], text_lower: str) -> bool:
    return any(re.search(p, text_lower) for p in patterns)


def classify_intent(question: str) -> str:
    """Advice -> comparison -> prediction -> portfolio -> performance -> factual."""
    q = (question or "").lower()

    if _matches(_ADVICE, q):
        return ADVICE

    # Two scheme pages named in one question: comparison, even if mild.
    if len(scheme_page_mentions(question)) >= 2:
        return COMPARISON
    if _matches(_COMPARISON, q):
        return COMPARISON

    if _matches(_PREDICTION, q):
        return PREDICTION
    if _matches(_PORTFOLIO, q):
        return PORTFOLIO
    if _matches(_PERFORMANCE, q):
        return PERFORMANCE
    return FACTUAL
