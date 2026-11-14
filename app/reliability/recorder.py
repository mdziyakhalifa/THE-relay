"""Request recorder — the ONLY writer of results.csv.

The 12-column contract below was locked by the user on 2026-11-11 (WP4 gate;
recorded as decision D13). Exact order, no additions, no renames:

    request_id, scenario, source, start_time, first_token_ms, total_ms,
    status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade

Row semantics (locked):
- ONE row per offered request, without exception — including queue
  rejections, quota blocks, breaker refusals, timeouts and cancellations.
- `first_token_ms` is empty when no generated content token arrived; zero is
  never substituted for a missing measurement.
- `status` is exactly one of ok / rate_limited / timeout / error. The typed
  internal outcome (capacity vs breaker vs quota vs ...) is PRESERVED in the
  structured event log via the `row_recorded` event's `outcome_kind`, never
  lost in the CSV.
- `answer_grade` is one of correct / wrong / refused_correctly /
  refused_wrongly, or EMPTY for requests without an evaluatable answer.
  Infrastructure failures are never auto-graded as correct refusals.
- `start_time` is the moment the request was OFFERED (caller-provided, not
  worker/provider time), ISO 8601 UTC with milliseconds and Z.
- `tokens_in/out` aggregate across all model calls and retries of the
  request. Only provider-REPORTED usage is aggregated into actuals;
  estimates are tracked separately and never merged (no invention).
- `retries` counts additional provider attempts beyond the first, summed
  over the whole pipeline.
- Duplicate `request_id` is a hard error (one row per offered request).
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pydantic

from app.core.logging_setup import get_logger
from app.reliability.errors import (
    AuthenticationError,
    BreakerOpen,
    CapacityRejected,
    DeadlineExceeded,
    MalformedRequestError,
    ModelCallsExhausted,
    ProviderServerError,
    QuotaBlock,
    RateLimitDailyError,
    RateLimitMinuteError,
    RelayError,
    RetrievalRoundsExhausted,
    RetryBudgetExhausted,
    TransientNetworkError,
    UserCancelled,
)

log = get_logger("relay.reliability.recorder")

# ---------------------------------------------------------------- contract

COLUMNS: tuple[str, ...] = (
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
)

STATUS_OK = "ok"
STATUS_RATE_LIMITED = "rate_limited"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"

SOURCE_REAL = "real"
SOURCE_STANDIN = "stand-in"

GRADE_CORRECT = "correct"
GRADE_WRONG = "wrong"
GRADE_REFUSED_CORRECTLY = "refused_correctly"
GRADE_REFUSED_WRONGLY = "refused_wrongly"

RETRIEVAL_NOT_APPLICABLE = "not_applicable"

SourceKind = Literal["real", "stand-in"]
StatusKind = Literal["ok", "rate_limited", "timeout", "error"]
GradeKind = Literal[
    "correct", "wrong", "refused_correctly", "refused_wrongly"
]

ISO_MS_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def utc_now_iso_ms() -> str:
    """Current UTC time as `2026-11-11T12:34:56.789Z` (locked format)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


# ------------------------------------------------------------------ mapping


def map_outcome(outcome: BaseException | None) -> tuple[StatusKind, str]:
    """Locked mapping typed outcome -> (status, typed_identity).

    The identity keeps the original distinction (local quota block vs
    provider rate limit vs breaker vs cancellation...) for the event log.
    """
    if outcome is None:
        return "ok", "completed"
    if isinstance(outcome, QuotaBlock):
        return "rate_limited", "quota_block_local"
    if isinstance(outcome, RateLimitDailyError):
        return "rate_limited", "provider_rate_limit_daily"
    if isinstance(outcome, RateLimitMinuteError):
        return "rate_limited", "provider_rate_limit_minute"
    if isinstance(outcome, DeadlineExceeded):
        return "timeout", "deadline_exceeded"
    if isinstance(outcome, CapacityRejected):
        return "error", "capacity_rejected"
    if isinstance(outcome, BreakerOpen):
        return "error", "breaker_open"
    if isinstance(outcome, UserCancelled):
        return "error", "user_cancelled"
    if isinstance(outcome, TransientNetworkError):
        return "error", "transient_network"
    if isinstance(outcome, ProviderServerError):
        return "error", "provider_server_error"
    if isinstance(outcome, AuthenticationError):
        return "error", "authentication_error"
    if isinstance(outcome, MalformedRequestError):
        return "error", "malformed_request"
    if isinstance(
        outcome, (RetryBudgetExhausted, ModelCallsExhausted, RetrievalRoundsExhausted)
    ):
        return "error", type(outcome).__name__
    if isinstance(outcome, RelayError):
        return "error", type(outcome).__name__
    return "error", f"unexpected_{type(outcome).__name__}"


# ------------------------------------------------------------------- tokens


@dataclass
class TokenUsageAccumulator:
    """Aggregates token usage across one request's model calls and retries.

    Honesty rule: `add_reported` only ever receives provider-REPORTED usage
    (Groq usage blocks, stand-in's documented tokenizer). Estimates go to
    the separate estimated_* fields and NEVER merge into actuals.
    """

    actual_in: int | None = None
    actual_out: int | None = None
    estimated_in: int = 0
    estimated_out: int = 0

    def add_reported(
        self, prompt: int | None = None, completion: int | None = None
    ) -> None:
        if prompt is not None:
            self.actual_in = (self.actual_in or 0) + int(prompt)
        if completion is not None:
            self.actual_out = (self.actual_out or 0) + int(completion)

    def add_estimated(self, prompt: int = 0, completion: int = 0) -> None:
        self.estimated_in += int(prompt)
        self.estimated_out += int(completion)


# --------------------------------------------------------------------- row


class RequestRecord(pydantic.BaseModel):
    """One results.csv row — exactly the 12 contract columns, no more."""

    model_config = pydantic.ConfigDict(extra="forbid")

    request_id: str = pydantic.Field(min_length=1)
    scenario: str = pydantic.Field(min_length=1)
    source: SourceKind
    start_time: str  # validated below against the locked format
    first_token_ms: float | None = pydantic.Field(default=None, ge=0)
    total_ms: float = pydantic.Field(ge=0)
    status: StatusKind
    retries: int = pydantic.Field(default=0, ge=0)
    tokens_in: int | None = pydantic.Field(default=None, ge=0)
    tokens_out: int | None = pydantic.Field(default=None, ge=0)
    retrieval_setting: str = pydantic.Field(min_length=1)
    answer_grade: GradeKind | None = None

    @pydantic.field_validator("start_time")
    @classmethod
    def _locked_timestamp_format(cls, value: str) -> str:
        if not ISO_MS_Z.match(value):
            raise ValueError(
                "start_time must be ISO 8601 UTC with milliseconds and Z, "
                "e.g. 2026-11-11T12:34:56.789Z"
            )
        return value

    def csv_cells(self) -> list[str]:
        """The exact 12 cells in contract order; empty string for None."""
        return [
            self.request_id,
            self.scenario,
            self.source,
            self.start_time,
            "" if self.first_token_ms is None else str(round(self.first_token_ms, 1)),
            str(round(self.total_ms, 1)),
            self.status,
            str(self.retries),
            "" if self.tokens_in is None else str(self.tokens_in),
            "" if self.tokens_out is None else str(self.tokens_out),
            self.retrieval_setting,
            "" if self.answer_grade is None else self.answer_grade,
        ]


# ----------------------------------------------------------------- writer


class RequestRecorder:
    """Appends rows to one results.csv, enforcing the locked contract."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: list[RequestRecord] = []
        self._ids: set[str] = set()
        self._header_written = False
        if self.path.is_file():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            if lines:
                if lines[0] != ",".join(COLUMNS):
                    raise ValueError(
                        f"{self.path}: existing header does not match the locked "
                        "12-column contract"
                    )
                self._header_written = True
                # pre-load known request_ids so a RESTARTED recorder still
                # enforces one-row-per-offered-request globally
                for line in lines[1:]:
                    if line:
                        self._ids.add(line.split(",", 1)[0])

    def _ensure_header(self) -> None:
        """Create the header only when the file is truly absent/empty; a file
        that appeared since __init__ is validated, never truncated."""
        if self._header_written:
            return
        if self.path.is_file():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            if lines:
                if lines[0] != ",".join(COLUMNS):
                    raise ValueError(
                        f"{self.path}: existing header does not match the locked "
                        "12-column contract"
                    )
                self._header_written = True
                for line in lines[1:]:
                    if line:
                        self._ids.add(line.split(",", 1)[0])
                return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(",".join(COLUMNS) + "\n", encoding="utf-8")
        self._header_written = True

    def has_request(self, request_id: str) -> bool:
        """True when this request_id already has a row (loaded or written).

        The API layer uses this to answer duplicates with HTTP 409 BEFORE
        offering the request; `record()` remains the hard backstop either
        way — a duplicate row can never be written.
        """
        return request_id in self._ids

    def record(self, row: RequestRecord, *, outcome_kind: str | None = None) -> None:
        """Append one row; duplicate request_id is a hard contract error."""
        if row.request_id in self._ids:
            raise ValueError(f"duplicate request_id {row.request_id!r}")
        self._ids.add(row.request_id)
        self.rows.append(row)
        self._ensure_header()
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerow(row.csv_cells())
        log.info(
            "row_recorded",
            extra={
                "request_id": row.request_id,
                "scenario": row.scenario,
                "source": row.source,
                "status": row.status,
                "outcome_kind": outcome_kind or "unspecified",
                "first_token_ms": row.first_token_ms,
                "total_ms": row.total_ms,
                "retries": row.retries,
                "tokens_in": row.tokens_in,
                "tokens_out": row.tokens_out,
            },
        )

    def record_outcome(
        self,
        *,
        request_id: str,
        scenario: str,
        source: SourceKind,
        start_time: str,
        total_ms: float,
        outcome: BaseException | None,
        first_token_ms: float | None = None,
        retries: int = 0,
        tokens: TokenUsageAccumulator | None = None,
        retrieval_setting: str = RETRIEVAL_NOT_APPLICABLE,
        answer_grade: GradeKind | None = None,
    ) -> RequestRecord:
        """Convenience entry: maps the typed outcome and stores it all."""
        status, kind = map_outcome(outcome)
        row = RequestRecord(
            request_id=request_id,
            scenario=scenario,
            source=source,
            start_time=start_time,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            status=status,
            retries=retries,
            tokens_in=tokens.actual_in if tokens else None,
            tokens_out=tokens.actual_out if tokens else None,
            retrieval_setting=retrieval_setting,
            answer_grade=answer_grade,
        )
        self.record(row, outcome_kind=kind)
        return row


# ------------------------------------------------- retrieval config metadata

RETRIEVAL_SETTINGS_METADATA: dict[str, dict] = {
    "A": {
        "chunk_tokens": 300,
        "overlap_tokens": 50,
        "top_k": 3,
        "rerank": False,
        "similarity_threshold": 0.25,
        "max_context_tokens": 3000,
    },
    "B": {
        "chunk_tokens": 500,
        "overlap_tokens": 80,
        "top_k": 5,
        "rerank": False,
        "similarity_threshold": 0.25,
        "max_context_tokens": 3000,
    },
    "C": {
        "chunk_tokens": 800,
        "overlap_tokens": 120,
        "top_k": 5,
        "rerank": True,
        "similarity_threshold": 0.25,
        "max_context_tokens": 3000,
    },
}


def save_retrieval_settings(path: Path) -> Path:
    """Write the configuration metadata file referenced by the contract
    (`retrieval_setting` column values resolve here)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "Phase-1 retrieval configurations; column values A/B/C resolve here. "
        "similarity_threshold and max_context_tokens are the shared app defaults.",
        "settings": RETRIEVAL_SETTINGS_METADATA,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
