"""Phase 7 — online query pipeline.

``answer_question(question) -> Answer`` is the only function the UI calls:

    question
      -> PII check          (block; store nothing)
      -> intent check       (refuse or redirect; no retrieval)
      -> scheme resolve     (ask which scheme if a fact needs one)
      -> embed + search     (same model, filter, top 6, threshold, MMR)
      -> grounded LLM call  (temperature 0, no URL in the model text)
      -> validate           (one retry, then a template from the top chunk)
      -> attach one citation from chunk metadata

Every non-PII turn writes ``artifacts/05_retrieval/{query_id}.json``.
Answer shape: body, then one link, then ``Last updated from sources:
{fetched_at}``. Paths: answer | refusal | redirect | clarify | miss |
template | pii_block.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from groww_faq.config import AppConfig, Secrets, get_secrets
from groww_faq.llm.factory import ChatClient, create_chat_client
from groww_faq.models import Answer, Hit, RetrievalTrace
from groww_faq.query.intent import (
    FACTUAL,
    PERFORMANCE,
    PORTFOLIO,
    REFUSAL_CLASSES,
    classify_intent,
)
from groww_faq.query.pii import PII_MESSAGE, detect_pii
from groww_faq.query.prompt import build_prompt
from groww_faq.query.retrieve import Retriever
from groww_faq.query.scheme import (
    AMC_SLUG,
    is_scheme_specific_fact,
    resolve_scheme,
    scheme_display_names,
    scheme_page_mentions,
)
from groww_faq.query.validate import template_body, validate_answer
from groww_faq.registry import ALL_URLS, citation_by_slug, entry_by_slug
from groww_faq.utils import sha256_text, utc_now_iso, write_json
from groww_faq.vectorstore import ChromaStore

logger = logging.getLogger(__name__)

REFUSAL_BODY = (
    "I can only share factual scheme details, so I can't advise whether "
    "this fund suits you. You can review the scheme's official page for "
    "its objective and risk level:"
)
REDIRECT_BODY = (
    "I don't provide returns or performance figures. You can view this "
    "scheme's data on its official page:"
)
MISS_BODY = (
    "I don't know that from the pages I have. You can check the official page:"
)
CLARIFY_BODY = "Which scheme do you mean — {names}?"

STATEMENT_CUES = ("statement",)


class QueryPipeline:
    """Online orchestrator. Store / LLM / encoder are injectable for tests."""

    def __init__(
        self,
        cfg: AppConfig,
        secrets: Secrets | None = None,
        store: Any = None,
        llm: ChatClient | None = None,
        encoder: Any = None,
        trace_dir: Any = None,
    ) -> None:
        self.cfg = cfg
        self.secrets = secrets or get_secrets()
        self.store = store or ChromaStore(cfg.paths.chroma, cfg.vectorstore.collection)
        self.retriever = Retriever(cfg, store=self.store, encoder=encoder)
        self._llm = llm
        self.trace_dir = trace_dir if trace_dir is not None else cfg.paths.retrieval
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ llm
    def ensure_llm(self) -> ChatClient:
        """Build the provider client on first use (public for startup checks)."""
        if self._llm is None:
            self._llm = create_chat_client(self.cfg.resolved_llm(self.secrets), self.secrets)
        return self._llm

    # ------------------------------------------------------------ entrance
    def answer_question(self, question: str) -> Answer:
        question = (question or "").strip()

        # 1) PII — before any log, embed, or artifact write.
        if detect_pii(question):
            logger.info("pii_block turn — question not stored")
            return Answer(body=PII_MESSAGE, path="pii_block")

        intent = classify_intent(question)
        resolved = resolve_scheme(question)
        schemes = scheme_page_mentions(question)
        query_id = self._query_id(question)

        # 2) Clarifying question — no LLM call.
        if intent == FACTUAL and is_scheme_specific_fact(question) and resolved is None:
            body = CLARIFY_BODY.format(names=", ".join(scheme_display_names()))
            answer = Answer(
                body=body, path="clarify", query_id=query_id, intent=intent
            )
            self._trace(query_id, question, intent, "", {}, [], answer)
            return answer

        # 3) Refusals — two sentences plus one link, no retrieval, no LLM.
        if intent in REFUSAL_CLASSES:
            url, label, related = self._refusal_link(question, intent, schemes, resolved)
            answer = Answer(
                body=REFUSAL_BODY, path="refusal", citation_url=url,
                citation_label=label, related_url=related,
                query_id=query_id, intent=intent, scheme_slug=resolved or "",
            )
            self._trace(query_id, question, intent, resolved or "", {}, [], answer)
            return answer

        # 4) Performance — one-line redirect, no CAGR computed.
        if intent == PERFORMANCE:
            url, label = self._page_link(schemes or ([resolved] if resolved else []))
            answer = Answer(
                body=REDIRECT_BODY, path="redirect", citation_url=url,
                citation_label=label, query_id=query_id, intent=intent,
                scheme_slug=resolved or "",
            )
            self._trace(query_id, question, intent, resolved or "", {}, [], answer)
            return answer

        # 5) Retrieve — filter by scheme_slug when a scheme was named.
        where = (
            {"scheme_slug": resolved}
            if resolved and self.cfg.retrieval.scheme_filter
            else None
        )
        hits, search_info = self.retriever.search(question, where=where)

        # 6) Miss — "I don't know" plus the official page link. No LLM call.
        if not hits:
            url, label = self._miss_link(question, resolved)
            answer = Answer(
                body=MISS_BODY, path="miss", citation_url=url,
                citation_label=label, query_id=query_id, intent=intent,
                scheme_slug=resolved or "",
            )
            self._trace(query_id, question, intent, resolved or "", where or {}, [], answer)
            return answer

        # 7) Grounded generation -> validate (one retry) -> template.
        body, path, llm_info = self._generate(question, hits)
        top = hits[0]
        citation_url, citation_label = self._citation(top, resolved)
        answer = Answer(
            body=body, path=path, citation_url=citation_url,
            citation_label=citation_label, fetched_at=top.fetched_at,
            query_id=query_id, intent=intent, scheme_slug=resolved or "",
        )
        self._trace(
            query_id, question, intent, resolved or "", where or {}, hits,
            answer, llm_info=llm_info, search_info=search_info,
        )
        return answer

    # ----------------------------------------------------------- generation
    def _generate(self, question: str, hits: list[Hit]) -> tuple[str, str, dict]:
        """LLM call at temperature 0; one regenerate; then the template."""
        system, user = build_prompt(question, hits)
        context = "\n".join(h.text for h in hits)
        # Record the *resolved* model (GROQ_MODEL in .env can override the
        # config.yaml default) — the client in ensure_llm uses the same one.
        llm_info: dict[str, Any] = {
            "model": self.cfg.resolved_llm(self.secrets).model,
            "validated": False,
        }

        try:
            llm = self.ensure_llm()
            text = llm.complete(
                system, user, self.cfg.llm.temperature, self.cfg.llm.max_tokens
            )
        except Exception as exc:  # key missing / timeout twice / provider down
            logger.warning("LLM call failed (%s) — template fallback", exc)
            llm_info["error"] = f"{type(exc).__name__}: {exc}"
            return template_body(hits[0].text), "template", llm_info

        ok, reason = validate_answer(text, context)
        if not ok:
            logger.info("validator rejected (%s) — one retry", reason)
            llm_info["rejected"] = reason
            try:  # one regenerate
                llm = self.ensure_llm()
                retry = llm.complete(
                    system, user, self.cfg.llm.temperature, self.cfg.llm.max_tokens
                )
                ok, reason = validate_answer(retry, context)
                if ok:
                    text = retry
            except Exception as exc:
                logger.warning("LLM retry failed (%s) — template fallback", exc)
                llm_info["retry_error"] = f"{type(exc).__name__}: {exc}"
                ok = False

        if ok:
            llm_info["validated"] = True
            return text, "answer", llm_info
        return template_body(hits[0].text), "template", llm_info

    # ---------------------------------------------------------------- links
    def _page_link(self, slugs: list[str]) -> tuple[str, str]:
        """Link to a named scheme page, else the AMC overview."""
        for slug in slugs:
            entry = entry_by_slug(slug)
            if entry is not None:
                return entry.url, entry.scheme or entry.amc
        amc = entry_by_slug(AMC_SLUG)
        return amc.url, amc.amc

    def _refusal_link(
        self, question: str, intent: str, schemes: list[str], resolved: str | None
    ) -> tuple[str, str, str]:
        """One primary link (plus the related URL on the trace only)."""
        # Portfolio advice gets the educational link (Tier B, never fetched).
        if intent == PORTFOLIO:
            link = citation_by_slug("groww-mf-knowledge")
            if link is not None:
                return link.url, link.label, ""
        if schemes:
            url, label = self._page_link(schemes[:1])
            related_url, _ = self._page_link(schemes[1:2]) if len(schemes) > 1 else ("", "")
            return url, label, related_url
        if resolved:
            url, label = self._page_link([resolved])
            return url, label, ""
        url, label = self._page_link([])
        return url, label, ""

    def _miss_link(self, question: str, resolved: str | None) -> tuple[str, str]:
        """Miss path link — the Tier B statement help for statement asks."""
        q = question.lower()
        if any(cue in q for cue in STATEMENT_CUES):
            link = citation_by_slug("groww-help-mf-statement")
            if link is not None:
                return link.url, link.label  # cited only, never scraped
        return self._page_link([resolved] if resolved else [])

    def _citation(self, top: Hit, resolved: str | None) -> tuple[str, str]:
        """The citation is data: top chunk's source_url, allowlist-checked."""
        if top.source_url in ALL_URLS:
            return top.source_url, top.scheme or top.amc
        # Not in the registry — substitute an allowlisted URL.
        url, label = self._page_link([resolved] if resolved else [])
        return url, label

    # ---------------------------------------------------------------- trace
    def _query_id(self, question: str) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{ts}_{sha256_text(question)[:8]}"

    def _trace(  # noqa: PLR0913 — one trace writer, fields per spec
        self,
        query_id: str,
        question: str,
        intent: str,
        scheme_slug: str,
        where: dict[str, Any],
        hits: list[Hit],
        answer: Answer,
        llm_info: dict[str, Any] | None = None,
        search_info: dict[str, Any] | None = None,
    ) -> None:
        trace = RetrievalTrace(
            query_id=query_id,
            created_at=utc_now_iso(),
            question=question,
            intent=intent,
            scheme_slug=scheme_slug,
            filter={**where, **(search_info or {})},
            path=answer.path,
            hits=[
                {
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "section": h.section,
                    "fact_key": h.fact_key,
                    "text": h.text,
                    "source_url": h.source_url,
                }
                for h in hits
            ],
            citation_url=answer.citation_url,
            citation_label=answer.citation_label,
            related_url=answer.related_url,
            answer_body=answer.body,
            llm=llm_info or {},
        )
        write_json(self.trace_dir / f"{query_id}.json", trace.model_dump(mode="json"))
