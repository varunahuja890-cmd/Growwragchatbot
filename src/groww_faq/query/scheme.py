"""Phase 7 — scheme resolver.

Alias map to scheme slug (architecture.md §6.3). When a scheme-specific
fact question (expense ratio, exit load, SIP, lock-in, riskometer,
benchmark) has no scheme, the pipeline asks one clarifying question that
lists the four schemes — no LLM call.
"""

from __future__ import annotations

from groww_faq.registry import TIER_A

AMC_SLUG = "hdfc-mutual-funds"

# Alias cues -> scheme_slug. Matched case-insensitively as substrings.
_ALIASES: dict[str, list[str]] = {
    "hdfc-elss-tax-saver-fund-direct-plan-growth": [
        "elss", "tax saver", "tax-saver", "taxsaver",
    ],
    "hdfc-equity-fund-direct-growth": [
        "flexi cap", "flexicap", "flexi-cap", "flexi",
    ],
    "hdfc-large-cap-fund-direct-growth": [
        "large cap", "largecap", "large-cap",
    ],
    "hdfc-mid-cap-fund-direct-growth": [
        "mid cap", "midcap", "mid-cap",
    ],
    AMC_SLUG: [
        "hdfc mutual fund", "hdfc amc", "amc page", "hdfc fund house",
        "all hdfc funds", "hdfc schemes",
    ],
}

# Fact keys that only make sense for one scheme (implementation.md §7).
SCHEME_SPECIFIC_CUES = [
    "expense ratio", "exit load", "sip", "lumpsum", "lock", "riskometer",
    "risk level", "benchmark", "fund manager", "objective", "nav", "aum",
    "category", "minimum investment", "min sip",
]

STATEMENT_CUES = ["statement", "download statement", "account statement"]


def scheme_page_mentions(question: str) -> list[str]:
    """Every *scheme* page named (AMC aliases excluded)."""
    q = (question or "").lower()
    hits: list[str] = []
    for slug, aliases in _ALIASES.items():
        if slug == AMC_SLUG:
            continue
        if any(a in q for a in aliases):
            hits.append(slug)
    return hits


def resolve_scheme(question: str) -> str | None:
    """First scheme page named; AMC page only when nothing more specific."""
    schemes = scheme_page_mentions(question)
    if schemes:
        return schemes[0]
    q = (question or "").lower()
    if any(a in q for a in _ALIASES[AMC_SLUG]):
        return AMC_SLUG
    return None


def scheme_display_names() -> list[str]:
    """The four scheme display names, for the clarifying question."""
    return [e.scheme for e in TIER_A if e.page_type == "scheme"]


def is_scheme_specific_fact(question: str) -> bool:
    q = (question or "").lower()
    return any(cue in q for cue in SCHEME_SPECIFIC_CUES)
