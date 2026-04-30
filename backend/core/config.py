"""Single source of truth for environment-driven configuration.

Every env var is typed and validated at startup. `extra="forbid"` means a typo
in `.env` raises `ValidationError` immediately rather than silently leaving a
field as None for hours. No `os.getenv` calls live outside this module.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(..., min_length=1)

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 1440

    # ── LLM (Gemini) ──────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., min_length=1)
    cheap_model: str = "gemini-2.5-flash-lite"
    strong_model: str = "gemini-2.5-flash"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 3072

    # ── Live Conditions APIs ──────────────────────────────────────────────────
    openmeteo_base_url: str = "https://api.open-meteo.com/v1"
    amadeus_api_key: str = ""
    amadeus_api_secret: str = ""

    # ── Webhook ───────────────────────────────────────────────────────────────
    webhook_url: str = ""
    webhook_timeout_seconds: int = 10
    webhook_max_retries: int = 3

    # ── RAG ───────────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    rag_top_k: int = 5

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ── ML Classifier ─────────────────────────────────────────────────────────
    model_path: str = "ml/model.joblib"
    model_meta_path: str = "ml/model_meta.json"
    classifier_confidence_threshold: float = 0.60

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "smart-travel-planner"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the Settings singleton.

    Returning a cached instance means `Settings()` is constructed exactly once
    per process — env vars are read once, validated once, and every consumer
    sees the same object.
    """
    return Settings()
