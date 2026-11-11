"""Phase 8 — Streamlit UI (one screen).

- Title and a one-line welcome.
- Three chips: Flexi Cap expense ratio, ELSS lock-in, Large Cap exit load.
- One text box. PII is rejected in the form before it is saved in session
  state.
- Answer, one link (new tab), last-updated line.
- Expander "Show retrieval" that reads the trace.
- Footer, exact text: ``Facts-only. No investment advice.``

Startup checks (implementation.md §7): the embedding manifest exists and
matches the configured model, the collection is non-empty, and the
configured LLM model id exists at the provider — a bad model id (for
example ``qwen/qwen3.8-27b`` on Groq) stops the app with that id in the
error. A missing or rejected API key only shows a warning banner: answers
then fall back to the cited template until the key is fixed in ``.env``.

The pipeline and the LLM check are cached, so a Streamlit rerun (every
button click) does not rebuild the Chroma client or re-probe the provider.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from groww_faq.config import AppConfig, get_config, get_secrets  # noqa: E402
from groww_faq.models import Answer  # noqa: E402
from groww_faq.query.pii import PII_MESSAGE, detect_pii  # noqa: E402
from groww_faq.query.pipeline import QueryPipeline  # noqa: E402
from groww_faq.utils import read_json  # noqa: E402

st.set_page_config(page_title="Groww MF Facts-Only FAQ", layout="centered")

CHIPS = [
    ("Flexi Cap expense ratio", "What is the expense ratio of HDFC Flexi Cap?"),
    ("ELSS lock-in", "What is the lock-in period of the HDFC ELSS Tax Saver Fund?"),
    ("Large Cap exit load", "What is the exit load of HDFC Large Cap?"),
]


@st.cache_resource(show_spinner="Loading the index and encoder ...")
def load_pipeline() -> QueryPipeline:
    """Build once per server process (Chroma client + manifest check)."""
    return QueryPipeline(get_config())


def startup_checks(cfg: AppConfig, pipeline: QueryPipeline) -> None:
    """Manifest + non-empty collection + provider model id (implementation.md §7)."""
    if pipeline.store.count() == 0:
        st.error("Index is not built. Run python scripts/ingest.py")
        st.stop()

    llm_cfg = cfg.resolved_llm(get_secrets())
    try:
        client = pipeline.ensure_llm()
    except RuntimeError as exc:
        st.warning(
            f"{exc} Answers will use the cited template until the key is "
            "added to .env and the app is restarted."
        )
        return
    if "llm_check" not in st.session_state:
        with st.spinner(f"Checking LLM model id at {llm_cfg.provider} ..."):
            st.session_state["llm_check"] = client.check_model()
    status, message = st.session_state["llm_check"]
    if status == "bad_model":  # wrong id — stop with the id in the error
        st.error(f"LLM startup check failed: {message}")
        st.stop()
    if status == "unverified":  # auth / network — degrade, don't block
        st.warning(message)


def render_answer(answer: Answer, cfg: AppConfig) -> None:
    st.markdown(answer.body)
    if answer.citation_url:
        label = answer.citation_label or "Source"
        st.markdown(
            f'<a href="{answer.citation_url}" target="_blank" '
            f'rel="noopener noreferrer">{label}</a>',
            unsafe_allow_html=True,
        )
    if answer.fetched_at:
        st.caption(f"Last updated from sources: {answer.fetched_at}")

    with st.expander("Show retrieval"):
        if answer.query_id:
            trace_path = cfg.paths.retrieval / f"{answer.query_id}.json"
            if trace_path.exists():
                trace = read_json(trace_path)
                st.caption(
                    f"query_id {trace['query_id']} · intent {trace['intent']} · "
                    f"path {trace['path']} · filter {trace.get('filter')}"
                )
                for hit in trace.get("hits", []):
                    st.markdown(
                        f"- `{hit['chunk_id']}` · score {hit['score']:.3f} · "
                        f"section *{hit.get('section', '')}* · "
                        f"fact *{hit.get('fact_key', '')}*"
                    )
                    st.text(hit.get("text", "")[:220])
            else:
                st.caption("No trace stored for this turn.")
        else:
            # PII turns return before a query_id exists — nothing is stored.
            st.caption("No trace stored for this turn.")


def main() -> None:
    st.title("Groww MF Facts-Only FAQ Assistant")
    st.markdown(
        "Ask a factual question about the four HDFC schemes — every "
        "answer cites one official page."
    )

    cfg = get_config()
    try:
        pipeline = load_pipeline()  # manifest check happens here
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    startup_checks(cfg, pipeline)

    question: str | None = None
    chip_cols = st.columns(3)
    for col, (label, chip_question) in zip(chip_cols, CHIPS):
        if col.button(label, help=chip_question):
            question = chip_question

    with st.form("ask", clear_on_submit=False):
        typed = st.text_input(
            "Your question",
            placeholder="e.g. What is the expense ratio of HDFC Large Cap?",
        )
        submitted = st.form_submit_button("Ask")
    if submitted and typed.strip():
        question = typed.strip()

    if question:
        # PII is rejected in the form before it is saved in session state.
        if detect_pii(question):
            st.session_state["answer"] = Answer(body=PII_MESSAGE, path="pii_block")
        else:
            with st.spinner("Thinking ..."):
                st.session_state["answer"] = pipeline.answer_question(question)

    if "answer" in st.session_state:
        st.divider()
        render_answer(st.session_state["answer"], cfg)

    st.divider()
    st.markdown("Facts-only. No investment advice.")


main()
