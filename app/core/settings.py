"""Typed application settings.

Design notes:
- Loaded from environment variables (prefix ``RELAY__``) with ``__`` as the
  nested delimiter, and optionally from ``.env`` (see ``check_env.py`` usage).
- Contains NO secrets. ``GROQ_API_KEY`` is read from the environment at the
  moment a provider needs it, never stored in settings, so this object can be
  logged or serialized without leaking credentials.
- All numeric defaults are the documented starting values from Phase 1
  (docs/decision_log.md); several are ASSUMPTIONS to be replaced by measured
  values during calibration (WP9/WP10).

Env-var examples:
    RELAY__PIPELINE__MAX_MODEL_CALLS=8
    RELAY__CAPACITY__QUEUE_LIMIT=15
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineBudgets(BaseModel):
    """Hard guards against runaway orchestration."""

    max_model_calls: int = 6
    """Upper bound on provider model calls per request (incl. retries counted
    separately) — loop safety mandated by the brief."""

    max_retrieval_rounds: int = 3
    """1 initial retrieval + up to 2 query-refinement loops."""

    request_deadline_ms: float = 45_000.0
    """Total wall-clock budget per request; overrun cancels the pipeline."""

    provider_call_timeout_s: float = 20.0
    """Per-call HTTP timeout for non-streaming provider calls."""

    stream_first_token_watchdog_s: float = 25.0
    """Cap on time-to-first-token for streaming calls."""


class RetryPolicy(BaseModel):
    """Exponential backoff with jitter; see app/reliability (WP4)."""

    max_attempts: int = 3
    backoff_base_s: float = 0.6
    backoff_factor: float = 2.0
    jitter_fraction: float = 0.30
    retries_per_request_budget: int = 4


class BreakerPolicy(BaseModel):
    failure_threshold: int = 5
    failure_window_s: float = 30.0
    open_seconds: float = 20.0
    half_open_max_calls: int = 1


class AppCapacity(BaseModel):
    in_flight_limit: int = 8
    """Concurrency cap for the app as a whole (stand-in provider runs)."""

    real_in_flight_limit: int = 2
    """Much tighter cap when the real Groq free tier is the target."""

    queue_limit: int = 200
    """Bounded queue; beyond this, requests are rejected as backpressure."""


class GroqConfig(BaseModel):
    base_url: str = "https://api.groq.com/openai/v1"
    """Documented OpenAI-compatible endpoint (console.groq.com/docs/openai)."""

    default_model: str = ""
    """Left EMPTY on purpose: no model is assumed before measurement (WP9)."""

    fast_model: str = ""
    """Optional cheaper model for classification stages; also WP9-decided."""


class StandInConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8100"


class RetrievalDefaults(BaseModel):
    """Retrieval-setting B from Phase 1 is the app default; A/B/C are eval-time
    overrides (WP3/WP11)."""

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 80
    top_k: int = 5
    similarity_threshold: float = 0.25
    rerank: bool = False
    max_context_tokens: int = 3000


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RELAY__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "the-relay"
    log_level: str = "INFO"
    results_dir: Path = Path("results")
    papers_dir: Path = Path("papers")

    pipeline: PipelineBudgets = PipelineBudgets()
    retry: RetryPolicy = RetryPolicy()
    breaker: BreakerPolicy = BreakerPolicy()
    capacity: AppCapacity = AppCapacity()
    groq: GroqConfig = GroqConfig()
    standin: StandInConfig = StandInConfig()
    retrieval: RetrievalDefaults = RetrievalDefaults()


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings; call ``get_settings.cache_clear()`` after
    mutating env vars in tests."""
    return Settings()
