"""Vercel serverless function — facts-only answer API (mirrors the app).

This is the Vercel edition of ``src/groww_faq/query/pipeline.py``: the
same PII -> intent -> scheme -> retrieve -> grounded LLM -> validate ->
one-citation flow, with the retrieval stack swapped for a dependency-light
one (ONNX int8 MiniLM + JSON/npy chunk index instead of torch +
sentence-transformers + Chroma). The guardrail modules themselves are
imported from ``src/groww_faq`` — they are the source of truth, not a
copy: PII patterns, intent regexes, the scheme alias map, the system
prompt, the post-validator and the citation registry are identical to the
Streamlit app.

Differences vs the pipeline, on purpose:

- No trace files (the Vercel filesystem is read-only): the trace is
  returned in the JSON response instead.
- No config.yaml / .env: the model id and key come from project
  environment variables (GROQ_MODEL, GROQ_API_KEY).
- A missing key does not crash: generation falls back to the cited
  template, exactly like any other LLM failure.

Route: POST /api/ask  {"question": "..."}  ->  Answer JSON.
GET /api/ask returns a small health/status payload (no question data).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

# --- make the repo package importable (src/ ships in the function bundle) ---
try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    _SRC = _REPO_ROOT / "src"
    if _SRC.is_dir():
        sys.path.insert(0, str(_SRC))
except Exception:  # pragma: no cover - path resolution must never crash import
    _REPO_ROOT = Path(".")

import numpy as np  # noqa: E402
import onnxruntime  # noqa: E402

from groww_faq.models import Hit  # noqa: E402
from groww_faq.query.intent import (  # noqa: E402
    FACTUAL,
    PERFORMANCE,
    PORTFOLIO,
    REFUSAL_CLASSES,
    classify_intent,
)
from groww_faq.query.pii import PII_MESSAGE, detect_pii  # noqa: E402
from groww_faq.query.prompt import build_prompt  # noqa: E402
from groww_faq.query.scheme import (  # noqa: E402
    AMC_SLUG,
    is_scheme_specific_fact,
    resolve_scheme,
    scheme_display_names,
    scheme_page_mentions,
)
from groww_faq.query.validate import template_body, validate_answer  # noqa: E402
from groww_faq.registry import ALL_URLS, citation_by_slug, entry_by_slug  # noqa: E402
from groww_faq.utils import sha256_text, utc_now_iso  # noqa: E402

# ---------------------------------------------------------------------------
# Retrieval config — mirrors config/config.yaml (retrieval + llm sections).
# ---------------------------------------------------------------------------
TOP_K = 6
SCORE_THRESHOLD = 0.25
USE_MMR = True
MMR_LAMBDA = 0.5
SCHEME_FILTER = True

LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 220
LLM_TIMEOUT_S = 25.0  # two attempts must fit the 60 s function budget
DEFAULT_MODEL = "openai/gpt-oss-120b"
GROQ_BASE = "https://api.groq.com/openai/v1"

# Mirrors query/pipeline.py turn bodies.
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

_DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Encoder + index (lazy — loaded on the first request, kept warm after)
# ---------------------------------------------------------------------------
class MiniLMOnnx:
    """all-MiniLM-L6-v2 via ONNX runtime.

    Same contract as SentenceTransformer.encode(normalize_embeddings=True):
    tokenize (max 256 tokens, pad to longest in batch), transformer forward,
    attention-mask-weighted mean pooling, L2 normalize. The class is also the
    embedding code used by scripts/export_onnx.py to build the index.
    """

    def __init__(self, onnx_path: Path, tokenizer_path: Path) -> None:
        import tokenizers

        self.session = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[start:start + batch_size]]
            encodings = self.tokenizer.encode_batch(batch)
            ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
            hidden = self.session.run(
                None, {"input_ids": ids, "attention_mask": mask}
            )[0].astype(np.float32)
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            chunks.append(pooled / np.clip(norm, 1e-12, None))
        return np.vstack(chunks).astype(np.float32)


_ENCODER: MiniLMOnnx | None = None
_CHUNKS: list[dict[str, Any]] | None = None
_VECTORS: np.ndarray | None = None


def ensure_index() -> tuple[MiniLMOnnx, list[dict[str, Any]], np.ndarray]:
    """Load (once) the ONNX encoder, chunk metadata and the vector matrix."""
    global _ENCODER, _CHUNKS, _VECTORS
    if _CHUNKS is None or _VECTORS is None or _ENCODER is None:
        _CHUNKS = json.loads(
            (_DATA_DIR / "chunks.json").read_text(encoding="utf-8")
        )
        _VECTORS = np.load(_DATA_DIR / "vectors.npy").astype(np.float32)
        _ENCODER = MiniLMOnnx(
            _DATA_DIR / "minilm_int8.onnx", _DATA_DIR / "tokenizer.json"
        )
        if _VECTORS.shape != (len(_CHUNKS), 384):
            raise RuntimeError(
                f"index mismatch: vectors {_VECTORS.shape} vs "
                f"{len(_CHUNKS)} chunks"
            )
    return _ENCODER, _CHUNKS, _VECTORS


# ---------------------------------------------------------------------------
# Retrieval — mirror of query/retrieve.py + the Chroma cosine-distance
# contract (similarity = 1 - distance; the matrix rows are L2-normalized,
# so the dot product is the cosine similarity).
# ---------------------------------------------------------------------------
def mmr_order(
    query_vec: np.ndarray, cand_vecs: np.ndarray, k: int, lam: float
) -> list[int]:
    """Greedy maximal-marginal-relevance — mirrors query/retrieve.py."""
    q_sim = cand_vecs @ query_vec
    sim = cand_vecs @ cand_vecs.T
    n = cand_vecs.shape[0]
    selected: list[int] = []
    remaining = set(range(n))
    while remaining and len(selected) < k:
        best_i, best_score = None, -1e9
        for i in remaining:
            redundancy = max(sim[i][j] for j in selected) if selected else 0.0
            score = lam * float(q_sim[i]) - (1.0 - lam) * float(redundancy)
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


def search(
    question: str, where: dict[str, Any] | None
) -> tuple[list[Hit], dict[str, Any]]:
    encoder, chunks, vectors = ensure_index()
    qvec = encoder.encode([question])[0]

    # Metadata filter (scheme_slug), then top-k by cosine similarity.
    slug = (where or {}).get("scheme_slug")
    rows = [
        i for i, c in enumerate(chunks)
        if not slug or c.get("scheme_slug") == slug
    ]
    pool = vectors[rows] if rows else np.zeros((0, vectors.shape[1]), np.float32)
    sims = pool @ qvec

    n_results = max(TOP_K * 3, 12) if USE_MMR else TOP_K
    order = np.argsort(-sims)[:n_results]

    hits: list[Hit] = []
    hit_vecs: list[np.ndarray] = []
    for pos in order:
        score = float(sims[pos])
        score = max(0.0, min(1.0, score))
        if score < SCORE_THRESHOLD:
            continue
        idx = rows[int(pos)]
        c = chunks[idx]
        hits.append(
            Hit(
                chunk_id=c["chunk_id"],
                score=round(score, 4),
                text=c["text"],
                source_url=c.get("source_url", ""),
                fetched_at=c.get("fetched_at", ""),
                amc=c.get("amc", ""),
                scheme=c.get("scheme", ""),
                scheme_slug=c.get("scheme_slug", ""),
                plan=c.get("plan", ""),
                page_type=c.get("page_type", ""),
                chunk_type=c.get("chunk_type", ""),
                section=c.get("section", ""),
                fact_key=c.get("fact_key", ""),
            )
        )
        hit_vecs.append(pool[int(pos)])

    if USE_MMR and len(hits) > TOP_K:
        keep = mmr_order(qvec, np.vstack(hit_vecs), TOP_K, MMR_LAMBDA)
        hits = [hits[i] for i in keep]
    else:
        hits = hits[:TOP_K]

    info = {
        "where": where or {},
        "n_results": n_results,
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
        "use_mmr": USE_MMR,
        "mmr_lambda": MMR_LAMBDA,
    }
    return hits, info


# ---------------------------------------------------------------------------
# LLM — mirror of llm/factory.py GroqClient (OpenAI-compatible endpoint,
# one internal retry on timeout / 5xx / 429, then raise -> template).
# ---------------------------------------------------------------------------
def llm_complete(system: str, user: str) -> str:
    import httpx

    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set on this deployment")
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    last_error = "unknown"
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(LLM_TIMEOUT_S),
    ) as client:
        for attempt in (1, 2):
            try:
                resp = client.post(f"{GROQ_BASE}/chat/completions", json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"http_{resp.status_code}"
                    if attempt == 1:
                        time.sleep(min(2 ** attempt, 8))
                        continue
                    break
                if resp.status_code >= 400:  # bad key / bad request — no retry
                    last_error = f"http_{resp.status_code}: {resp.text[:160]}"
                    break
                resp.raise_for_status()
                return str(resp.json()["choices"][0]["message"]["content"]).strip()
            except (httpx.HTTPError, KeyError, TypeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 1:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
    raise RuntimeError(f"LLM call failed after 2 attempts ({last_error})")


def _generate(
    question: str, hits: list[Hit]
) -> tuple[str, str, dict[str, Any]]:
    """LLM call at temperature 0; one regenerate; then the template."""
    system, user = build_prompt(question, hits)
    context = "\n".join(h.text for h in hits)
    llm_info: dict[str, Any] = {
        "model": os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        "validated": False,
    }
    try:
        text = llm_complete(system, user)
    except Exception as exc:  # key missing / timeout twice / provider down
        llm_info["error"] = f"{type(exc).__name__}: {exc}"
        return template_body(hits[0].text), "template", llm_info

    ok, reason = validate_answer(text, context)
    if not ok:
        llm_info["rejected"] = reason
        try:  # one regenerate
            retry = llm_complete(system, user)
            ok, reason = validate_answer(retry, context)
            if ok:
                text = retry
        except Exception as exc:
            llm_info["retry_error"] = f"{type(exc).__name__}: {exc}"
            ok = False

    if ok:
        llm_info["validated"] = True
        return text, "answer", llm_info
    return template_body(hits[0].text), "template", llm_info


# ---------------------------------------------------------------------------
# Links / citations — mirror of pipeline.py helpers (registry-allowlisted).
# ---------------------------------------------------------------------------
def _page_link(slugs: list[str]) -> tuple[str, str]:
    for slug in slugs:
        entry = entry_by_slug(slug)
        if entry is not None:
            return entry.url, entry.scheme or entry.amc
    amc = entry_by_slug(AMC_SLUG)
    return amc.url, amc.amc


def _refusal_link(
    question: str, intent: str, schemes: list[str], resolved: str | None
) -> tuple[str, str, str]:
    if intent == PORTFOLIO:
        link = citation_by_slug("groww-mf-knowledge")
        if link is not None:
            return link.url, link.label, ""
    if schemes:
        url, label = _page_link(schemes[:1])
        related_url, _ = (
            _page_link(schemes[1:2]) if len(schemes) > 1 else ("", "")
        )
        return url, label, related_url
    if resolved:
        url, label = _page_link([resolved])
        return url, label, ""
    url, label = _page_link([])
    return url, label, ""


def _miss_link(question: str, resolved: str | None) -> tuple[str, str]:
    q = question.lower()
    if any(cue in q for cue in STATEMENT_CUES):
        link = citation_by_slug("groww-help-mf-statement")
        if link is not None:
            return link.url, link.label  # cited only, never scraped
    return _page_link([resolved] if resolved else [])


def _citation(top: Hit, resolved: str | None) -> tuple[str, str]:
    if top.source_url in ALL_URLS:
        return top.source_url, top.scheme or top.amc
    return _page_link([resolved] if resolved else [])


def _query_id(question: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{sha256_text(question)[:8]}"


# ---------------------------------------------------------------------------
# The answer flow — mirror of QueryPipeline.answer_question, stage by stage.
# ---------------------------------------------------------------------------
def answer_question(question: str) -> dict[str, Any]:
    question = (question or "").strip()

    # 1) PII — before any log, embed, or artifact write.
    if detect_pii(question):
        return {"body": PII_MESSAGE, "path": "pii_block"}

    intent = classify_intent(question)
    resolved = resolve_scheme(question)
    schemes = scheme_page_mentions(question)
    query_id = _query_id(question)

    # 2) Clarifying question — no LLM call.
    if intent == FACTUAL and is_scheme_specific_fact(question) and resolved is None:
        body = CLARIFY_BODY.format(names=", ".join(scheme_display_names()))
        return _respond(
            question, query_id, intent, "", {}, [], body, "clarify", "", ""
        )

    # 3) Refusals — two sentences plus one link, no retrieval, no LLM.
    if intent in REFUSAL_CLASSES:
        url, label, related = _refusal_link(question, intent, schemes, resolved)
        return _respond(
            question, query_id, intent, resolved or "", {}, [], REFUSAL_BODY,
            "refusal", url, label, related_url=related,
        )

    # 4) Performance — one-line redirect, no CAGR computed.
    if intent == PERFORMANCE:
        url, label = _page_link(schemes or ([resolved] if resolved else []))
        return _respond(
            question, query_id, intent, resolved or "", {}, [], REDIRECT_BODY,
            "redirect", url, label,
        )

    # 5) Retrieve — filter by scheme_slug when a scheme was named.
    where = {"scheme_slug": resolved} if resolved and SCHEME_FILTER else None
    hits, search_info = search(question, where)

    # 6) Miss — "I don't know" plus the official page link. No LLM call.
    if not hits:
        url, label = _miss_link(question, resolved)
        return _respond(
            question, query_id, intent, resolved or "", where or {}, [],
            MISS_BODY, "miss", url, label,
        )

    # 7) Grounded generation -> validate (one retry) -> template.
    body, path, llm_info = _generate(question, hits)
    top = hits[0]
    citation_url, citation_label = _citation(top, resolved)
    return _respond(
        question, query_id, intent, resolved or "", where or {}, hits, body,
        path, citation_url, citation_label, fetched_at=top.fetched_at,
        llm_info=llm_info, search_info=search_info,
    )


def _respond(  # noqa: PLR0913 — one response builder, pipeline trace fields
    question: str,
    query_id: str,
    intent: str,
    scheme_slug: str,
    where: dict[str, Any],
    hits: list[Hit],
    body: str,
    path: str,
    citation_url: str,
    citation_label: str,
    fetched_at: str = "",
    related_url: str = "",
    llm_info: dict[str, Any] | None = None,
    search_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The Answer + the query-time trace, returned instead of written."""
    return {
        "body": body,
        "path": path,
        "citation_url": citation_url,
        "citation_label": citation_label,
        "related_url": related_url,
        "fetched_at": fetched_at,
        "query_id": query_id,
        "intent": intent,
        "scheme_slug": scheme_slug,
        "question": question,  # PII turns never reach this builder
        "created_at": utc_now_iso(),
        "llm": llm_info or {},
        "filter": {**where, **(search_info or {})},
        "hits": [
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
        "disclaimer": "Facts-only. No investment advice.",
    }


# ---------------------------------------------------------------------------
# HTTP handler — Vercel file-based Python function.
# ---------------------------------------------------------------------------
class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel entrypoint name

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            _, chunks, vectors = ensure_index()
            self._send_json(200, {
                "ok": True,
                "chunks": len(chunks),
                "dim": int(vectors.shape[1]),
                "model": os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
                "groq_key_set": bool(os.environ.get("GROQ_API_KEY")),
            })
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 10_000:
                self._send_json(400, {"error": "expected JSON body {\"question\": ...}"})
                return
            raw = self.rfile.read(length)
            try:
                question = str(json.loads(raw).get("question", ""))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "invalid JSON body"})
                return
            self._send_json(200, answer_question(question))
        except Exception as exc:
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args: Any) -> None:  # silence the default
        pass


# ---------------------------------------------------------------------------
# Local harness — python api/ask.py runs demo turns against the local index.
# (On Vercel __main__ never runs; the handler class above is the entrypoint.)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env_path = _REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    print("model:", os.environ.get("GROQ_MODEL", DEFAULT_MODEL))
    print("key set:", bool(os.environ.get("GROQ_API_KEY")))
    demos = [
        "What is the lock-in period of the HDFC ELSS Tax Saver Fund?",
        "What is the exit load of HDFC Mid Cap?",
        "Should I invest in HDFC Mid Cap?",
        "My PAN is ABCDE1234F",
        "What is the expense ratio of HDFC Flexi Cap?",
        "How do I download my mutual fund statement?",
    ]
    for q in demos:
        result = answer_question(q)
        print("-" * 72)
        print("Q:", q)
        print("path:", result["path"], "| intent:", result.get("intent", ""),
              "| hits:", len(result.get("hits", [])), "| llm:", result.get("llm", {}))
        print("A:", result["body"])
        if result.get("citation_url"):
            print("cite:", result.get("citation_label"), "->", result["citation_url"])
