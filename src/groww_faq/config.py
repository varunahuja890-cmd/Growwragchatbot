"""Configuration loading.

Every non-secret knob comes from ``config/config.yaml``. Secrets come only
from ``.env`` / the environment (pydantic-settings) — never from yaml.
Paths are anchored to the project root so scripts run from any CWD.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from groww_faq.registry import TIER_A

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


class AppSettings(BaseModel):
    name: str = "Groww MF Facts-Only FAQ"
    disclaimer: str = "Facts-only. No investment advice."
    max_answer_sentences: int = 3


class IngestionSettings(BaseModel):
    urls: list[str]
    user_agent: str = "Mozilla/5.0 (compatible; GrowwFaqBot/1.0)"
    timeout_s: float = 20.0
    retries: int = 2
    delay_s: float = 1.0
    respect_robots: bool = True


class ChunkingSettings(BaseModel):
    strategy: str = "layout_hybrid"
    max_tokens: int = 400
    overlap_tokens: int = 60
    min_tokens: int = 10
    hard_max_tokens: int = 800
    fact_chunks: bool = True
    table_rows: bool = True


class EmbeddingSettings(BaseModel):
    provider: str = "sentence-transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


class VectorStoreSettings(BaseModel):
    provider: str = "chromadb"
    persist_dir: str = "artifacts/04_chroma"
    collection: str = "groww_mf_facts"
    distance: str = "cosine"


class RetrievalSettings(BaseModel):
    top_k: int = 6
    score_threshold: float = 0.25
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    scheme_filter: bool = True


class LlmSettings(BaseModel):
    provider: str = "groq"
    model: str = "qwen/qwen3-32b"
    temperature: float = 0.0
    max_tokens: int = 220
    api_key_env: str = "GROQ_API_KEY"
    timeout_s: float = 60.0


class Paths(BaseModel):
    """Resolved artifact directories (anchored to the project root)."""

    artifacts: Path = PROJECT_ROOT / "artifacts"

    @field_validator("artifacts")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def raw(self) -> Path:
        return self.artifacts / "01_raw"

    @property
    def clean(self) -> Path:
        return self.artifacts / "01_clean"

    @property
    def chunks(self) -> Path:
        return self.artifacts / "02_chunks"

    @property
    def embeddings(self) -> Path:
        return self.artifacts / "03_embeddings"

    @property
    def embedding_cache(self) -> Path:
        return self.artifacts / "03_embeddings" / "cache"

    @property
    def chroma(self) -> Path:
        return self.artifacts / "04_chroma"

    @property
    def retrieval(self) -> Path:
        return self.artifacts / "05_retrieval"

    def ensure_all(self) -> None:
        for d in (
            self.raw,
            self.clean,
            self.chunks,
            self.embeddings,
            self.embedding_cache,
            self.chroma,
            self.retrieval,
        ):
            d.mkdir(parents=True, exist_ok=True)


class Secrets(BaseSettings):
    """Secrets live in .env / the environment only, never in yaml."""

    llm_provider: str = "groq"
    groq_api_key: str | None = None
    groq_model: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    mistral_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH), env_file_encoding="utf-8", extra="ignore"
    )

    def redacted(self) -> dict[str, str]:
        """For run_manifest.json: every key appears as ``set`` or ``missing``."""

        def flag(value: str | None) -> str:
            return "set" if value else "missing"

        return {
            "GROQ_API_KEY": flag(self.groq_api_key),
            "OPENAI_API_KEY": flag(self.openai_api_key),
            "ANTHROPIC_API_KEY": flag(self.anthropic_api_key),
            "GEMINI_API_KEY": flag(self.gemini_api_key),
            "MISTRAL_API_KEY": flag(self.mistral_api_key),
        }


class AppConfig(BaseModel):
    app: AppSettings = AppSettings()
    ingestion: IngestionSettings
    chunking: ChunkingSettings = ChunkingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vectorstore: VectorStoreSettings = VectorStoreSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    llm: LlmSettings = LlmSettings()
    paths: Paths = Paths()

    def validate_registry(self) -> None:
        """Startup check: the yaml URL list is exactly the five registry URLs."""
        registry_urls = sorted(e.url for e in TIER_A)
        config_urls = sorted(self.ingestion.urls)
        if config_urls != registry_urls:
            missing = [u for u in registry_urls if u not in config_urls]
            extra = [u for u in config_urls if u not in registry_urls]
            raise ValueError(
                "config ingestion.urls must be exactly the five Tier A registry "
                f"URLs. missing={missing} extra={extra}"
            )
        if not self.embedding.model_name:
            raise ValueError("embedding.model_name must not be empty")

    def resolved_llm(self, secrets: Secrets) -> LlmSettings:
        """LLM settings with .env overrides applied (provider and Groq model)."""
        data = self.llm.model_dump()
        if secrets.llm_provider:
            data["provider"] = secrets.llm_provider
        if secrets.groq_model and data["provider"] == "groq":
            data["model"] = secrets.groq_model
        return LlmSettings(**data)


@lru_cache(maxsize=1)
def get_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    cfg = AppConfig(
        app=AppSettings(**raw.get("app", {})),
        ingestion=IngestionSettings(**raw.get("ingestion", {})),
        chunking=ChunkingSettings(**raw.get("chunking", {})),
        embedding=EmbeddingSettings(**raw.get("embedding", {})),
        vectorstore=VectorStoreSettings(**raw.get("vectorstore", {})),
        retrieval=RetrievalSettings(**raw.get("retrieval", {})),
        llm=LlmSettings(**raw.get("llm", {})),
        paths=Paths(**raw.get("paths", {})),
    )
    cfg.validate_registry()
    return cfg


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()
