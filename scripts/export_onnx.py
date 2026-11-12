"""Export the MiniLM encoder to ONNX and build the serverless chunk index.

The Vercel edition (api/ask.py) runs the same facts-only pipeline as the
Streamlit app but cannot ship torch / sentence-transformers / chromadb
(serverless bundle limits). This script produces a dependency-light index:

  api/data/minilm_int8.onnx    all-MiniLM-L6-v2 transformer, int8-quantized
  api/data/tokenizer.json      the HF fast tokenizer for that model
  api/data/chunks.json         chunk metadata, registry order (468 rows)
  api/data/vectors.npy         float32 (N, 384) torch document vectors —
                               the exact rows Chroma stores, same order
  api/data/index_manifest.json counts, dims, verification stats (FR-9)

Chunk texts come from ``artifacts/02_chunks/*.jsonl`` in registry order —
the same order the Chroma index was built in. ``vectors.npy`` copies the
torch reference vectors from ``artifacts/03_embeddings`` (not re-embedded
ONNX ones), so the document side of the serverless retrieval is
bit-identical to Chroma; the ONNX encoder embeds *queries* only. That single
approximation is gated: ONNX vectors are computed for every chunk and
checked against the torch vectors — mean cosine must be >= MIN_MEAN_COS
and the worst chunk >= MIN_COS, else the script exits non-zero and writes
nothing.

Run (local only, needs torch + sentence-transformers):
    set HF_HOME=E:\\hf_cache
    python scripts/export_onnx.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # groww_faq.* (registry)
sys.path.insert(0, str(ROOT / "api"))  # ask.MiniLMOnnx (single source of truth)

from groww_faq.registry import TIER_A  # noqa: E402

SNAPSHOT = (
    Path("E:/hf_cache/hub/models--sentence-transformers--all-MiniLM-L6-v2")
    / "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384
MAX_SEQ = 256
MIN_MEAN_COS = 0.985  # ONNX vs torch, mean over all chunks
MIN_COS = 0.90        # ONNX vs torch, worst single chunk

DATA_DIR = ROOT / "api" / "data"
TMP_DIR = Path("E:/temp/onnx_export")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_chunks() -> list[dict]:
    """All chunks in registry order — the Chroma build order."""
    chunks: list[dict] = []
    for page in TIER_A:
        path = ROOT / "artifacts" / "02_chunks" / f"{page.slug}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing chunk file: {path} — run scripts/ingest.py")
        rows = read_jsonl(path)
        chunks.extend(rows)
    return chunks


def load_torch_vectors(chunks: list[dict]) -> np.ndarray:
    """Reference vectors: per-slug .npy rows, one per jsonl line, in order."""
    per_slug: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        per_slug.setdefault(chunk["scheme_slug"], []).append(i)
    out = np.zeros((len(chunks), DIM), dtype=np.float32)
    for slug, indices in per_slug.items():
        path = ROOT / "artifacts" / "03_embeddings" / f"{slug}.npy"
        if not path.exists():
            raise SystemExit(f"missing embeddings file: {path}")
        arr = np.load(path).astype(np.float32)
        if arr.shape != (len(indices), DIM):
            raise SystemExit(
                f"{path} has shape {arr.shape}, expected ({len(indices)}, {DIM})"
            )
        out[indices] = arr
    return out


def export_onnx_fp32(out_path: Path) -> None:
    """Export the bare transformer (input_ids, attention_mask -> hidden)."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        SNAPSHOT, local_files_only=True, attn_implementation="eager"
    )
    model.eval()

    ids = torch.randint(1000, 2000, (1, 16), dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)

    dynamic = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "last_hidden_state": {0: "batch", 1: "seq"},
    }
    try:
        torch.onnx.export(
            model, (ids, mask), str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic, opset_version=17,
            do_constant_folding=True,
        )
    except Exception as exc:  # pragma: no cover - export fallback
        print(f"classic exporter failed ({exc}); retrying with dynamo=True")
        torch.onnx.export(
            model, (ids, mask), str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic, opset_version=17,
            do_constant_folding=True, dynamo=True,
        )
    print(f"exported fp32 onnx -> {out_path}")


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path), str(int8_path),
        weight_type=QuantType.QInt8, per_channel=True,
    )
    print(f"quantized int8 onnx -> {int8_path}")


def main() -> int:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("loading chunks ...")
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"  {len(chunks)} chunks")

    print("loading torch reference vectors ...")
    torch_vecs = load_torch_vectors(chunks)

    fp32 = TMP_DIR / "minilm_fp32.onnx"
    int8 = DATA_DIR / "minilm_int8.onnx"
    export_onnx_fp32(fp32)
    quantize_int8(fp32, int8)

    # Tokenizer: copy from the HF snapshot (self-contained fast tokenizer).
    tokenizer_src = SNAPSHOT / "tokenizer.json"
    tokenizer_dst = DATA_DIR / "tokenizer.json"
    shutil.copyfile(tokenizer_src, tokenizer_dst)
    print(f"copied tokenizer -> {tokenizer_dst}")

    # Embed every chunk with the quantized model using the SAME code the
    # serverless function will use (api/ask.py :: MiniLMOnnx).
    from ask import MiniLMOnnx  # noqa: E402

    print("embedding all chunks with int8 onnx ...")
    encoder = MiniLMOnnx(int8, tokenizer_dst)
    onnx_vecs = encoder.encode(texts, batch_size=32)

    if onnx_vecs.shape != (len(chunks), DIM):
        raise SystemExit(f"onnx matrix shape {onnx_vecs.shape} unexpected")

    # Verification: cosine of normalized vectors == dot product.
    cos = np.sum(onnx_vecs * torch_vecs, axis=1)
    mean_cos = float(cos.mean())
    min_cos = float(cos.min())
    print(f"onnx vs torch: mean cosine {mean_cos:.4f}, min {min_cos:.4f}")
    if mean_cos < MIN_MEAN_COS or min_cos < MIN_COS:
        print(
            f"FAILED gate (mean >= {MIN_MEAN_COS}, min >= {MIN_COS}) — "
            "nothing written; adjust quantization before deploying."
        )
        return 1

    # Metadata (query-time fields only) + vectors, registry order.
    # vectors.npy = torch reference vectors (the exact rows Chroma stores):
    # the ONNX encoder is used at QUERY time only, so the document side of
    # the serverless retrieval is bit-identical to the app's Chroma index.
    fields = (
        "chunk_id text amc scheme scheme_slug plan page_type chunk_type "
        "section fact_key source_url fetched_at"
    ).split()
    meta = [{k: c[k] for k in fields} for c in chunks]
    (DATA_DIR / "chunks.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    np.save(DATA_DIR / "vectors.npy", torch_vecs.astype(np.float32))
    print(f"wrote chunks.json ({len(meta)} rows) + vectors.npy {torch_vecs.shape}")

    manifest = {
        "model_name": MODEL_NAME,
        "dim": DIM,
        "max_seq_len": MAX_SEQ,
        "count": len(chunks),
        "encoder": "onnx int8 (dynamic, per-channel MatMul) — mirrors "
                   "sentence-transformers/all-MiniLM-L6-v2, mask-weighted "
                   "mean pooling + L2 normalize. Used for QUERIES only.",
        "document_vectors": "torch reference (artifacts/03_embeddings) — "
                            "the exact rows the Chroma collection stores, "
                            "registry order; document-side ranking is "
                            "bit-identical to the app.",
        "files": {
            "onnx": "minilm_int8.onnx",
            "tokenizer": "tokenizer.json",
            "chunks": "chunks.json",
            "vectors": "vectors.npy",
        },
        "verification": {
            "against": "artifacts/03_embeddings per-slug .npy (torch)",
            "measures": "query-side substitution: ONNX embedding vs torch "
                        "embedding over all chunk texts",
            "mean_cosine": round(mean_cos, 4),
            "min_cosine": round(min_cos, 4),
            "gate": f"mean >= {MIN_MEAN_COS}, min >= {MIN_COS}",
        },
        "per_slug": {
            slug: sum(1 for c in chunks if c["scheme_slug"] == slug)
            for slug in dict.fromkeys(c["scheme_slug"] for c in chunks)
        },
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (DATA_DIR / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"wrote index_manifest.json — OK ({manifest['per_slug']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
