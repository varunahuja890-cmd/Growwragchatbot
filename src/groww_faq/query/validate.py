"""Phase 7 — post-validator.

Checks the model text before anything is shown (architecture.md §6.6):

| check          | pass condition                                     | on failure        |
|----------------|----------------------------------------------------|-------------------|
| length         | at most 3 sentences                                 | regenerate once   |
| URLs           | zero URLs in the model output                       | regenerate once   |
| advice words   | none of should/recommend/better/avoid/buy/sell/hold | regenerate once   |
| numbers        | every number appears verbatim in the context        | regenerate once   |

After one failed regenerate the pipeline emits a template built from the
top chunk. Decimal points inside numbers ("0.74%") do not count as
sentence breaks.
"""

from __future__ import annotations

import re

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
URL_RE = re.compile(r"(?i)https?://|www\.")
ADVICE_RE = re.compile(r"(?i)\b(should|recommend|better|avoid|buy|sell|hold)\b")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def sentence_count(text: str) -> int:
    return len([s for s in SENTENCE_RE.split(text.strip()) if s.strip()])


def validate_answer(text: str, context: str) -> tuple[bool, str]:
    """Return (ok, reason). ``context`` is the concatenated chunk text."""
    t = (text or "").strip()
    if not t:
        return False, "empty"
    if sentence_count(t) > 3:
        return False, "too_many_sentences"
    if URL_RE.search(t):
        return False, "url_in_answer"
    if ADVICE_RE.search(t):
        return False, "advice_vocabulary"
    for num in NUMBER_RE.findall(t):
        if num not in context:
            return False, "invented_number"
    return True, ""


def template_body(top_chunk_text: str, max_sentences: int = 2) -> str:
    """Template fallback: the top chunk's fact line, capped to the contract."""
    sentences = [s for s in SENTENCE_RE.split(top_chunk_text.strip()) if s.strip()]
    if not sentences:
        return top_chunk_text.strip()
    return " ".join(sentences[:max_sentences])
