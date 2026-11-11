"""Phase 7 — PII guard.

Runs first, in-process, on the raw string — before any log line, embed
call, or artifact write (PRD FR: privacy). On match the pipeline returns
the fixed line below and returns *before* the retrieval trace is written,
so the raw string never lands under ``artifacts/``.
"""

from __future__ import annotations

import re

PII_MESSAGE = "I don't need or store personal details."

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # PAN: 5 uppercase letters, 4 digits, 1 uppercase letter
    ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    # Aadhaar: 12 digits, optionally in 4-4-4 groups
    ("aadhaar", re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
    # e-mail address
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # Indian mobile: optional +91, then a 10-digit number starting 6-9
    ("phone", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")),
    # OTP near a 3-8 digit code
    ("otp", re.compile(r"(?i)\botp\b\D{0,12}\d{3,8}")),
    # Account / folio numbers: keyword followed by 5+ digits
    (
        "account",
        re.compile(r"(?i)\b(account|a/c|folio)\s*(number|no\.?|id)?\s*[:#]?\s*\d{5,}"),
    ),
]


def detect_pii(text: str) -> str | None:
    """Return the matched PII kind, or None when the input is clean."""
    for kind, pattern in _PATTERNS:
        if pattern.search(text):
            return kind
    return None
