#!/usr/bin/env python
"""Offline ingestion orchestrator.

    python scripts/ingest.py                 # fetch -> load -> chunk -> embed -> index
    python scripts/ingest.py --from load     # skip the network, restart at load
    python scripts/ingest.py --from chunk
    python scripts/ingest.py --from embed
    python scripts/ingest.py --from index    # rebuild the vector DB from artifacts
    python scripts/ingest.py --only index    # demo a single stage

Exit code is 1 when any of the five pages fails at any stage, or when the
index verification (all chunks present, self-similarity, sample query)
fails.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from groww_faq.config import get_config  # noqa: E402
from groww_faq.ingestion.chunk import chunk_all  # noqa: E402
from groww_faq.ingestion.embed import embed_all  # noqa: E402
from groww_faq.ingestion.fetch import Fetcher  # noqa: E402
from groww_faq.ingestion.index import index_all  # noqa: E402
from groww_faq.ingestion.load import load_all  # noqa: E402

STAGES = ["fetch", "load", "chunk", "embed", "index"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the ingestion pipeline (phases 2-5)."
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        choices=STAGES,
        default="fetch",
        help="restart at this stage (skips the network when HTML is already saved)",
    )
    parser.add_argument(
        "--only", choices=STAGES, help="run a single stage (for phase demos)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s"
    )

    cfg = get_config()
    cfg.paths.ensure_all()

    stages = [args.only] if args.only else STAGES[STAGES.index(args.from_stage):]
    failures: list[str] = []

    for stage in stages:
        print(f"\n=== stage: {stage} " + "=" * (40 - len(stage)))
        if stage == "fetch":
            outcomes = Fetcher(cfg).fetch_all()
            for outcome in outcomes:
                status = "ok" if outcome.ok else f"FAILED ({outcome.reason})"
                print(f"  {outcome.slug:48s} {status}")
            failures += [
                f"fetch:{o.slug}:{o.reason}" for o in outcomes if not o.ok
            ]

        elif stage == "load":
            report = load_all(cfg)
            for page in report["pages"]:
                if page["ok"]:
                    facts_found = sum(
                        1
                        for v in page["facts"].values()
                        if v.get("status") == "found"
                    )
                    print(
                        f"  {page['slug']:48s} ok  "
                        f"sections={page['section_count']} facts={facts_found}"
                    )
                else:
                    print(f"  {page['slug']:48s} FAILED ({page['reason']})")
            print(
                f"  pages ok={report['pages_ok']} failed={report['pages_failed']} "
                f"facts found={report['total_facts_found']} gaps={report['total_gaps']}"
            )
            failures += [
                f"load:{r['slug']}:{r['reason']}"
                for r in report["pages"]
                if not r["ok"]
            ]

        elif stage == "chunk":
            stats = chunk_all(cfg)
            print(
                f"  total chunks={stats['total_chunks']} by_type={stats['by_type']}"
            )
            for slug, gate in stats.get("fact_gate", {}).items():
                if gate["missing_fact_chunks"]:
                    failures.append(
                        f"chunk:{slug}:missing fact chunks {gate['missing_fact_chunks']}"
                    )
                    print(
                        f"  {slug:48s} MISSING fact chunks: "
                        f"{gate['missing_fact_chunks']}"
                    )

        elif stage == "embed":
            manifest = embed_all(cfg)
            print(
                f"  model={manifest['model_name']} dim={manifest['dim']} "
                f"count={manifest['count']} "
                f"(encoded={manifest['encoded']} cache_hits={manifest['cache_hits']})"
            )

        elif stage == "index":
            try:
                report = index_all(cfg)
            except RuntimeError as exc:
                print(f"  index FAILED: {exc}")
                failures.append(f"index:{exc}")
                continue
            coll = report["collection"]
            self_sim = report["self_similarity_check"]
            print(
                f"  collection={coll['name']} count={coll['count']} "
                f"(cosine, artifacts/04_chroma/)"
            )
            print(
                f"  all {coll['chunk_count']} chunks present in the vector DB: "
                f"{coll['all_chunk_ids_present']}"
                + (
                    f" (missing={coll['missing_count']})"
                    if coll["missing_count"]
                    else ""
                )
            )
            print(
                f"  self-similarity check: {self_sim['similarity']:.4f} "
                f"(passed={self_sim['passed']})"
            )
            top = report["sample_query"].get("top_hit", {})
            print(
                f"  sample query 'expense ratio' -> "
                f"{top.get('fact_key', '?')} of {top.get('scheme_slug', '?')} "
                f"(ok={report['sample_query']['ok']})"
            )
            print(f"  run manifest: artifacts/run_manifest.json")

    print()
    if failures:
        print("Ingestion finished with failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Ingestion complete. Inspect stages with scripts/inspect.py.")
    print("Next: streamlit run ui/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
