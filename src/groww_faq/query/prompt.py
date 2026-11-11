"""Phase 7 — prompt builder.

The system prompt allows only the supplied context: no outside knowledge,
no arithmetic, no comparisons, no advice, at most 3 sentences, no URL. The
model is told to say "I don't know" when the context lacks the fact; the
citation is appended by the pipeline, never by the model.
"""

from __future__ import annotations

from groww_faq.models import Hit

SYSTEM_PROMPT = """You answer factual questions about HDFC mutual fund schemes using ONLY the supplied context chunks.

Rules:
- Use only facts stated in the context. If the answer is not in the context, reply exactly: I don't know.
- Do not use outside knowledge.
- Do not calculate, estimate, or project any number. Copy numbers exactly as written.
- Do not compare schemes and do not give investment advice.
- Answer in at most 3 short sentences.
- Never include a URL or link."""


def build_prompt(question: str, hits: list[Hit]) -> tuple[str, str]:
    """(system, user) — each chunk labeled with chunk_id and source_url."""
    context = "\n\n".join(
        f"[chunk {h.chunk_id} | {h.source_url}]\n{h.text}" for h in hits
    )
    user = (
        f"Question: {question}\n\n"
        "Context chunks (the only allowed source of facts):\n\n"
        f"{context}"
    )
    return SYSTEM_PROMPT, user
