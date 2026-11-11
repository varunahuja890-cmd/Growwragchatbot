"""Citation registry — the allowlist behind both scraping and citations.

Tier A is the corpus: exactly five Groww pages, fetched once each, depth 1.
No anchors are followed, no "related funds", no sitemaps.

Tier B URLs are official Groww / HDFC MF / SEBI / AMFI destinations that
are linked in answers, refusals and redirects — they are never fetched.
The combined registry (15 URLs) backs Docs/sources.md and
Docs/sources.csv (PRD deliverable D2).
"""

from __future__ import annotations

from groww_faq.models import CitationLink, RegistryEntry

AMC_NAME = "HDFC Mutual Fund"

TIER_A: list[RegistryEntry] = [
    RegistryEntry(
        slug="hdfc-mutual-funds",
        url="https://groww.in/mutual-funds/amc/hdfc-mutual-funds",
        page_type="amc",
        amc=AMC_NAME,
        scheme="",
        plan="",
    ),
    RegistryEntry(
        slug="hdfc-large-cap-fund-direct-growth",
        url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        page_type="scheme",
        amc=AMC_NAME,
        scheme="HDFC Large Cap Fund",
        plan="Direct Growth",
    ),
    RegistryEntry(
        slug="hdfc-equity-fund-direct-growth",
        url="https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
        page_type="scheme",
        amc=AMC_NAME,
        scheme="HDFC Flexi Cap Fund",
        plan="Direct Plan Growth",
    ),
    RegistryEntry(
        slug="hdfc-elss-tax-saver-fund-direct-plan-growth",
        url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        page_type="scheme",
        amc=AMC_NAME,
        scheme="HDFC ELSS Tax Saver Fund",
        plan="Direct Plan Growth",
    ),
    RegistryEntry(
        slug="hdfc-mid-cap-fund-direct-growth",
        url="https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
        page_type="scheme",
        amc=AMC_NAME,
        scheme="HDFC Mid Cap Fund",
        plan="Direct Growth",
    ),
]

TIER_B: list[CitationLink] = [
    CitationLink(
        slug="hdfc-mf-home",
        url="https://www.hdfcfund.com/",
        label="HDFC Mutual Fund — official website",
        purpose="Factsheet / SID redirects",
    ),
    CitationLink(
        slug="groww-help",
        url="https://groww.in/help",
        label="Groww Help & Support",
        purpose="Statement-download help and support",
    ),
    CitationLink(
        slug="groww-help-mf-statement",
        url="https://groww.in/help/my-account/searchable/how-to-download-mutual-fund-statement",
        label="How to download a mutual fund statement — Groww help",
        purpose="Statement-download answers (miss path)",
    ),
    CitationLink(
        slug="groww-help-contact",
        url="https://groww.in/help/my-account/searchable/how-to-call-groww-customer-care",
        label="How to call Groww customer care",
        purpose="Support contact",
    ),
    CitationLink(
        slug="groww-mf-landing",
        url="https://groww.in/mutual-funds",
        label="Mutual funds on Groww",
        purpose="Educational redirect",
    ),
    CitationLink(
        slug="groww-mf-knowledge",
        url="https://groww.in/blog/category/mutual-funds",
        label="Groww — mutual funds knowledge centre",
        purpose="Educational redirect",
    ),
    CitationLink(
        slug="amfi-home",
        url="https://www.amfiindia.com/",
        label="AMFI — Association of Mutual Funds in India",
        purpose="Investor education",
    ),
    CitationLink(
        slug="amfi-investor-corner",
        url="https://www.amfiindia.com/investor-corner",
        label="AMFI investor corner",
        purpose="Investor education",
    ),
    CitationLink(
        slug="sebi-home",
        url="https://www.sebi.gov.in/",
        label="SEBI — Securities and Exchange Board of India",
        purpose="Regulator",
    ),
    CitationLink(
        slug="sebi-scores",
        url="https://scores.sebi.gov.in/",
        label="SEBI SCORES — complaint registration",
        purpose="Grievance escalation",
    ),
]

TIER_A_URLS = [e.url for e in TIER_A]
ALL_URLS = TIER_A_URLS + [c.url for c in TIER_B]

_ENTRY_BY_SLUG = {e.slug: e for e in TIER_A}
_ENTRY_BY_URL = {e.url: e for e in TIER_A}
_LINK_BY_SLUG = {c.slug: c for c in TIER_B}


def entry_by_slug(slug: str) -> RegistryEntry | None:
    return _ENTRY_BY_SLUG.get(slug)


def entry_by_url(url: str) -> RegistryEntry | None:
    return _ENTRY_BY_URL.get(url)


def citation_by_slug(slug: str) -> CitationLink | None:
    return _LINK_BY_SLUG.get(slug)
