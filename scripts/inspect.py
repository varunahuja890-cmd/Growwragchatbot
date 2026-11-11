#!/usr/bin/env python
"""FR-9 stage inspector — print any pipeline artifact for a demo.

    python scripts/inspect.py list
    python scripts/inspect.py stage 01_raw --id hdfc-mid-cap-fund-direct-growth
    python scripts/inspect.py stage 01_clean --id hdfc-mid-cap-fund-direct-growth
    python scripts/inspect.py stage 02_chunks --id <chunk_id>
    python scripts/inspect.py stage 02_chunks --id hdfc-mid-cap-fund-direct-growth
    python scripts/inspect.py stage 03_embeddings --id hdfc-mid-cap-fund-direct-growth
    python scripts/inspect.py stage 04_chroma                 # presence check + counts
    python scripts/inspect.py stage 05_retrieval [--id <query_id>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from groww_faq.config import get_config  # noqa: E402
from groww_faq.utils import read_json, read_jsonl  # noqa: E402

STAGES = ["01_raw", "01_clean", "02_chunks", "03_embeddings", "04_chroma", "05_retrieval"]


def cmd_list(cfg) -> None:
    artifacts = cfg.paths.artifacts
    if not artifacts.exists():
        print(f"no artifacts yet at {artifacts} — run scripts/ingest.py first")
        return
    for stage in STAGES:
        stage_dir = artifacts / stage
        print(f"\n{stage}:")
        if not stage_dir.exists():
            print("  (missing)")
            continue
        files = sorted(p.name for p in stage_dir.iterdir())
        if not files:
            print("  (empty)")
        for name in files:
            print(f"  {name}")


def cmd_stage(cfg, stage: str, chunk_id: str | None, limit: int) -> int:
    paths = cfg.paths

    if stage == "01_raw":
        if not chunk_id:
            _list_slugs(paths.raw, ".html")
            return 0
        meta_path = paths.raw / f"{chunk_id}.meta.json"
        html_path = paths.raw / f"{chunk_id}.html"
        if not html_path.exists():
            print(f"no raw HTML for '{chunk_id}'")
            return 1
        if meta_path.exists():
            print(read_json(meta_path))
        html = html_path.read_text(encoding="utf-8", errors="replace")
        print(f"--- first {limit} chars of {html_path.name} ---")
        print(html[:limit])

    elif stage == "01_clean":
        if not chunk_id:
            _list_slugs(paths.clean, ".json")
            return 0
        doc_path = paths.clean / f"{chunk_id}.json"
        if not doc_path.exists():
            print(f"no clean document for '{chunk_id}'")
            return 1
        doc = read_json(doc_path)
        print(f"title: {doc['page_title']}")
        print(f"url:   {doc['source_url']}")
        print(f"slug:  {doc['slug']}  type={doc['page_type']}  "
              f"fetched_at={doc['fetched_at']}")
        print(f"content_hash: {doc['content_hash']}")
        print(f"sections: {len(doc['sections'])}")
        for section in doc["sections"][:limit]:
            print(f"\n  [{section['level']}] {section['heading']}")
            for pair in section["key_value_pairs"]:
                print(f"    fact  {pair['key']} = {pair['value']!r} (via {pair['via']})")
            for table in section["tables"]:
                print(f"    table headers={table['headers']} rows={len(table['rows'])}")
            for block in section["text_blocks"][:5]:
                print(f"    text  {block[:110]!r}")
            if len(section["text_blocks"]) > 5:
                print(f"    ... {len(section['text_blocks']) - 5} more text blocks")

    elif stage == "02_chunks":
        rows = []
        for jsonl in sorted(paths.chunks.glob("*.jsonl")):
            rows.extend(read_jsonl(jsonl))
        if not rows:
            print(f"no chunks under {paths.chunks} — run the chunk stage first")
            return 1
        if chunk_id:
            matches = [r for r in rows if r["chunk_id"] == chunk_id]
            if matches:
                print(json.dumps(matches[0], ensure_ascii=False, indent=2))
                return 0
            by_slug = [r for r in rows if r["scheme_slug"] == chunk_id]
            if not by_slug:
                print(f"no chunk with id or slug '{chunk_id}'")
                return 1
            rows = by_slug
        print(f"{len(rows)} chunks (showing up to {limit}):")
        for row in rows[:limit]:
            print(
                f"  {row['chunk_id']}  {row['chunk_type']:9s} "
                f"{row['tokens']:4d}t  {row['section'][:30]:30s}  "
                f"{row['text'][:90]!r}"
            )

    elif stage == "03_embeddings":
        manifest_path = paths.embeddings / "manifest.json"
        if not manifest_path.exists():
            print(f"no manifest at {manifest_path} — run the embed stage first")
            return 1
        manifest = read_json(manifest_path)
        print(
            f"model={manifest['model_name']} dim={manifest['dim']} "
            f"count={manifest['count']} checksum={manifest['checksum'][:16]}..."
        )
        print(f"per_slug={manifest['per_slug']}")
        if chunk_id:
            npy_path = paths.embeddings / f"{chunk_id}.npy"
            if not npy_path.exists():
                print(f"no vectors for '{chunk_id}'")
                return 1
            import numpy as np

            arr = np.load(npy_path)
            print(f"{npy_path.name}: shape={arr.shape} dtype={arr.dtype}")
            norms = np.linalg.norm(arr, axis=1)
            print(f"row norms: min={norms.min():.4f} max={norms.max():.4f} "
                  f"(normalized => ~1.0)")

    elif stage == "04_chroma":
        manifest_path = paths.embeddings / "manifest.json"
        if not manifest_path.exists():
            print(f"no manifest at {manifest_path} — run the embed stage first")
            return 1
        manifest = read_json(manifest_path)
        from groww_faq.vectorstore import ChromaStore

        store = ChromaStore(paths.chroma, cfg.vectorstore.collection)
        count = store.count()
        stored = set(store.all_ids())
        expected = set(manifest["chunk_ids"])
        missing = sorted(expected - stored)
        print(f"collection: {cfg.vectorstore.collection} (cosine, {paths.chroma})")
        print(f"chunks stored: {count}  expected: {len(expected)}")
        print(f"all chunks present: {not missing and count == len(expected)}")
        if missing:
            print(f"missing ids: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
            return 1
        # per-scheme counts
        data = store.get_all()
        per_slug: dict[str, int] = {}
        for meta in data["metadatas"]:
            slug = (meta or {}).get("scheme_slug", "?")
            per_slug[slug] = per_slug.get(slug, 0) + 1
        print("per scheme_slug:")
        for slug, n in sorted(per_slug.items()):
            print(f"  {slug:48s} {n}")
        if chunk_id:
            rows = [
                (i, m) for i, m in zip(data["ids"], data["metadatas"])
                if (m or {}).get("scheme_slug") == chunk_id
                or i == chunk_id
            ]
            if not rows:
                print(f"no chunks for '{chunk_id}'")
                return 1
            print(f"\n{len(rows)} chunks for '{chunk_id}' (showing up to {limit}):")
            for cid, meta in rows[:limit]:
                print(
                    f"  {cid}  {meta.get('chunk_type', ''):9s} "
                    f"{meta.get('fact_key', ''):18s} {meta.get('section', '')[:34]}"
                )

    elif stage == "05_retrieval":
        trace_dir = paths.retrieval
        if not trace_dir.exists() or not list(trace_dir.glob("*.json")):
            print(f"no traces under {trace_dir} — ask a question in the app first")
            return 1
        traces = sorted(trace_dir.glob("*.json"))
        if not chunk_id:
            print("available query ids: " + ", ".join(t.stem for t in traces))
            return 0
        trace_path = trace_dir / f"{chunk_id}.json"
        if not trace_path.exists():
            print(f"no trace for '{chunk_id}'")
            return 1
        trace = read_json(trace_path)
        print(f"query_id:  {trace['query_id']}")
        print(f"question:  {trace['question']}")
        print(f"intent:    {trace['intent']}   path: {trace['path']}")
        print(f"filter:    {trace.get('filter')}")
        print(f"citation:  {trace.get('citation_url')}")
        print(f"answer:    {trace.get('answer_body', '')[:200]}")
        print(f"hits ({len(trace.get('hits', []))}):")
        for hit in trace.get("hits", [])[:limit]:
            print(
                f"  {hit['chunk_id']}  score={hit['score']:.3f}  "
                f"{hit.get('section', '')[:28]:28s} {hit.get('fact_key', ''):16s} "
                f"{hit.get('text', '')[:70]!r}"
            )

    return 0


def _list_slugs(directory: Path, suffix: str) -> None:
    if not directory.exists():
        print(f"{directory} does not exist — run the pipeline first")
        return
    skip = {"parse_report"}
    slugs = sorted(
        p.stem
        for p in directory.glob(f"*{suffix}")
        if p.stem not in skip and not p.name.endswith(".meta.json")
    )
    if slugs:
        print("available ids: " + ", ".join(slugs))
    else:
        print(f"nothing found under {directory}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect pipeline artifacts.")
    parser.add_argument("command", choices=["list", "stage"])
    parser.add_argument("stage", nargs="?", choices=STAGES)
    parser.add_argument("--id", help="slug, chunk id, or query id")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    cfg = get_config()

    if args.command == "list":
        cmd_list(cfg)
        return 0
    if not args.stage:
        parser.error("the 'stage' command needs a stage name")
    return cmd_stage(cfg, args.stage, args.id, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
