"""RelayRunner — the request state machine (WP7, decisions D5/D6/D16).

Hand-rolled asyncio pipeline (D6): admit -> retrieve -> merged
understanding call (D5) -> optional refined second retrieval round ->
streamed answer -> citation validation -> one regeneration at most ->
honest outcome + exactly ONE results.csv row (D13) + a per-request event
file (D13 supporting logs).

Reliability composition per provider attempt (single place in the codebase):
    budget claim -> [retry executor: limiter admit -> breaker -> attempt]
with these mapping rules:
- limiter projection block        -> QuotaBlock (row rate_limited,
                                     quota_block_local; never sent, never
                                     retried, never a breaker input)
- breaker refusal                 -> BreakerOpen (row error)
- daily 429                       -> RateLimitDailyError (D8: raised through
                                     the retry executor untouched)
- per-minute 429 / 5xx / network  -> retried within the per-request retry
                                     allowance and per-call attempt bounds
- per-call/first-token watchdogs  -> TransientNetworkError (same family as
                                     httpx timeouts: retryable, breaker-
                                     counted provider-health signal)
- request wall-clock deadline     -> DeadlineExceeded (row timeout)
- user cancellation               -> row user_cancelled, CancelledError
                                     re-raised
- retries (results.csv)           = provider HTTP attempts - 1, summed over
                                     the whole pipeline (D13 wording); the
                                     event file carries the per-stage
                                     breakdown (retry vs regen vs round-2).

Honesty rules baked in:
- `first_token_ms` measures from the OFFER moment (runner entry), covers
  queue wait, and is None when no generated content token was ever
  delivered (never zero-substituted).
- When a stream attempt emits deltas and then fails, the partial content
  is discarded with an explicit `partial_stream_discarded` event; TTFT is
  measured on the attempt whose content is actually delivered.
- Token CSV values aggregate provider-REPORTED usage only; limiter booking
  may estimate (booking affects admission, never the CSV), and every
  booking is event-logged with reported-vs-estimated labeled.
- Unparseable understanding output never fabricates a refusal; evidence
  present + unparseable classification -> proceed to answer (citation
  validator backstops). Recorded explicitly (D16).
- A regeneration that fails or runs out of budget never destroys the first
  answer: the request still delivers it, flagged with invalid citations
  (availability over silent perfectionism).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core.logging_setup import get_logger
from app.core.settings import PipelineBudgets, RetryPolicy
from app.orchestrator.citations import CitationReport, validate_citations
from app.orchestrator.events import EventSink, RequestEventLog, utc_now_iso_ms
from app.orchestrator.prompts import (
    ANSWER_SYSTEM,
    UNDERSTANDING_SYSTEM,
    answer_user_prompt,
    parse_understanding,
    understanding_user_prompt,
)
from app.providers import ChatMessage, ChatUsage, LLMProvider
from app.reliability.breaker import CircuitBreaker
from app.reliability.budget import RequestBudget
from app.reliability.capacity import RequestGate
from app.reliability.errors import (
    ConfigurationError,
    DeadlineExceeded,
    RelayError,
    TransientNetworkError,
    UserCancelled,
)
from app.reliability.errors import CapacityRejected
from app.reliability.limiter import HeaderQuotaLimiter
from app.reliability.recorder import (
    RequestRecord,
    RequestRecorder,
    TokenUsageAccumulator,
    map_outcome,
)
from app.reliability.retry import RetryExecutor

log = get_logger("relay.orchestrator.runner")

PERF_CLOCK: Callable[[], float] = time.perf_counter

_REFUSAL_TEMPLATE = (
    "I cannot answer this question from the three pinned papers. ({reason})"
)

# booking allowance for the limiter projection: context + expected output
_OUTPUT_TOKEN_ALLOWANCE = 256


class _Refusal(Exception):
    """Internal control flow: the pipeline decided an honest refusal."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def source_for_provider(provider_name: str) -> str:
    """results.csv `source` mapping — refusing to mislabel is a hard rule."""
    if provider_name == "groq":
        return "real"
    if provider_name in ("stand-in", "standin"):
        return "stand-in"
    raise ConfigurationError(
        f"unknown provider name {provider_name!r}; refusing to mislabel "
        "results.csv source (allowed: groq, stand-in)"
    )


@dataclass
class RelayOutcome:
    """The full, honest result of one offered request."""

    request_id: str
    scenario: str
    retrieval_setting: str
    source: str
    status: str
    outcome_kind: str
    answer: str
    refused: bool
    refusal_reason: str
    citations: CitationReport | None
    used_items: list[dict]
    first_token_ms: float | None
    total_ms: float
    retries: int
    tokens_in: int | None
    tokens_out: int | None
    outcome: BaseException | None
    events_path: Path | None

    @property
    def citations_valid(self) -> bool | None:
        return None if self.citations is None else self.citations.valid


@dataclass
class _RunState:
    """Per-run mutable state (run() may be called concurrently)."""

    request_id: str
    scenario: str
    retrieval_setting: str
    start_time: str
    offered_perf: float
    budget: RequestBudget
    events: RequestEventLog
    tokens: TokenUsageAccumulator = field(default_factory=TokenUsageAccumulator)
    used_items: list = field(default_factory=list)
    provider_attempts: int = 0
    first_token_perf: float | None = None
    refused: bool = False
    refusal_reason: str = ""
    retrieval_ran: bool = False  # D13: not_applicable until retrieval ran
    token_sink: Callable[[dict], object] | None = None  # WP8: /ask/stream live pump


async def _notify(state: _RunState, kind: str, **fields: object) -> None:
    """Forward one live-stream event to this run's token_sink (WP8).

    The sink exists only for the streaming API endpoint; it must NEVER be
    able to break the pipeline: a throwing consumer is event-logged and
    skipped, the honest row is unaffected. Queue overflow is handled inside
    the API's sink wrapper (drop + count), never here. Cancellation always
    propagates (a cancelled run must stay cancelled).
    """
    sink = state.token_sink
    if sink is None:
        return
    try:
        outcome = sink({"kind": kind, **fields})
        if hasattr(outcome, "__await__"):
            await outcome
    except Exception as error:  # noqa: BLE001 — a dead consumer must not break the run
        state.events.emit("token_sink_error", error=str(error))


class RelayRunner:
    def __init__(
        self,
        *,
        retrievers: dict[str, object],
        provider: LLMProvider,
        model: str,
        gate: RequestGate,
        breaker: CircuitBreaker,
        limiter: HeaderQuotaLimiter,
        pipeline: PipelineBudgets,
        retry_policy: RetryPolicy,
        recorder: RequestRecorder | None = None,
        sink: EventSink | None = None,
        clock: Callable[[], float] = PERF_CLOCK,
        sleeper: Callable[[float], object] | None = None,
    ) -> None:
        self._retrievers = retrievers
        self._provider = provider
        self._model = model
        self._gate = gate
        self._breaker = breaker
        self._limiter = limiter
        self._pipeline = pipeline
        self._retry_policy = retry_policy
        self._recorder = recorder
        self._sink = sink
        self._clock = clock
        self._sleeper = sleeper  # tests inject an instant sleeper; None = real
        # construction-time contract: an unlabele-able provider must fail
        # the build, never a request row
        self._source = source_for_provider(provider.name)

    # ------------------------------------------------------------- public

    @property
    def retrieval_settings(self) -> frozenset[str]:
        """The bound retrieval settings — the API layer validates against
        this BEFORE offering a request (a 400 must never produce a row)."""
        return frozenset(self._retrievers)

    async def run(
        self,
        *,
        request_id: str,
        scenario: str,
        question: str,
        retrieval_setting: str,
        start_time: str,
        token_sink: Callable[[dict], object] | None = None,
    ) -> RelayOutcome:
        """Run one offered request; ALWAYS one row + one event file.

        `token_sink` (WP8): optional per-run async callback receiving
        {kind: status|token|reset, ...} dicts for live SSE forwarding.
        None (the default) changes nothing — /ask stays fully buffered.
        """
        events = self._sink.open(request_id) if self._sink else _NullLog()
        state = _RunState(
            request_id=request_id,
            scenario=scenario,
            retrieval_setting=retrieval_setting,
            start_time=start_time,
            offered_perf=self._clock(),
            budget=RequestBudget(
                self._pipeline,
                self._retry_policy,
                clock=self._clock,
                start_at=self._clock(),
            ),
            events=events,
            token_sink=token_sink,
        )
        events.emit(
            "offered",
            scenario=scenario,
            question=question,
            retrieval_setting=retrieval_setting,
            start_time=start_time,
            source=self._source,
        )
        try:
            if retrieval_setting not in self._retrievers:
                raise ConfigurationError(
                    f"no retriever bound for retrieval_setting {retrieval_setting!r}"
                )
            async with self._gate.slot() as waited:
                queue_wait_ms = (self._clock() - state.offered_perf) * 1000.0
                events.emit(
                    "queue_admitted",
                    waited=waited,
                    queue_wait_ms=round(queue_wait_ms, 1),
                )
                await _notify(
                    state,
                    "status",
                    stage="admitted",
                    queue_wait_ms=round(queue_wait_ms, 1),
                )
                return await self._execute_pipeline(state, question)
        except _Refusal as refusal:
            return self._finish_refusal(state, refusal.reason)
        except CapacityRejected as error:
            return self._finish_typed(state, error)
        except RelayError as error:
            return self._finish_typed(state, error)
        except asyncio.CancelledError:
            self._finish_typed(state, UserCancelled("request cancelled"))
            raise
        except Exception as error:  # bug safety net: still one honest row
            self._finish_typed(state, error, unexpected=True)
            raise
        finally:
            events.close()

    # ----------------------------------------------------------- pipeline

    async def _execute_pipeline(self, state: _RunState, question: str) -> RelayOutcome:
        budget, events = state.budget, state.events
        retriever = self._retrievers[state.retrieval_setting]

        # ---- retrieval round 1
        budget.check_deadline()
        budget.claim_retrieval_round()
        result = await self._retrieve(state, retriever, query=question, round_no=1)
        if not result.items:
            raise _Refusal(
                "no relevant passage was retrieved from the pinned papers"
            )
        state.used_items = list(result.items)

        # ---- merged understanding call (D5)
        understanding = await self._understanding_call(
            state, question=question, result=result, round_no=1
        )
        if understanding is not None and not understanding["sufficient"]:
            refined = understanding["refined_search_terms"]
            events.emit(
                "understanding_insufficient",
                round=1,
                reason=understanding["reason"],
                refined=refined,
            )
            if not refined or not refined.strip():
                raise _Refusal(
                    understanding["reason"] or "understanding judged the evidence insufficient"
                )
            # ---- one refinement round: retrieve again, understand again
            budget.check_deadline()
            budget.claim_retrieval_round()
            result2 = await self._retrieve(
                state, retriever, query=refined.strip(), round_no=2, refined=True
            )
            if not result2.items:
                raise _Refusal(
                    "refined retrieval found no relevant passage in the pinned papers"
                )
            state.used_items = list(result2.items)
            understanding2 = await self._understanding_call(
                state, question=question, result=result2, round_no=2
            )
            if understanding2 is not None and not understanding2["sufficient"]:
                raise _Refusal(
                    understanding2["reason"]
                    or "evidence still insufficient after one refinement round"
                )

        # ---- streamed answer
        answer_text, meta = await self._answer_call(
            state, question=question, label="answer"
        )
        report = validate_citations(answer_text, state.used_items)
        events.emit(
            "citation_checked",
            valid=report.valid,
            uncited=report.uncited,
            cited=len(report.cited),
            unknown=len(report.unknown),
        )

        # ---- one regeneration at most, only for citation-invalid answers
        if not report.valid:
            events.emit("regen_started", reason="invalid citations")
            await _notify(state, "reset", reason="regen")
            try:
                answer2, meta2 = await self._answer_call(
                    state, question=question, label="regen"
                )
            except RelayError as error:
                events.emit(
                    "regen_aborted",
                    error_kind=type(error).__name__,
                    delivered_first_answer=True,
                )
                if isinstance(error, DeadlineExceeded):
                    # the first answer EXISTS but the deadline expired: the
                    # request returned no user-visible answer (D16 rule) ->
                    # honest timeout row; first_token_ms stays empty
                    return self._finish_typed(state, error)
                raise
            report2 = validate_citations(answer2, state.used_items)
            events.emit(
                "citation_rechecked",
                valid=report2.valid,
                uncited=report2.uncited,
                cited=len(report2.cited),
                unknown=len(report2.unknown),
            )
            if report2.valid:
                answer_text, report = answer2, report2
                meta = meta2  # the delivered attempt defines TTFT (D16)
            else:
                events.emit("regen_still_invalid", delivered_first_answer=True)
        state.first_token_perf = meta.get("first_token_perf")

        total_ms = (self._clock() - state.offered_perf) * 1000.0
        return self._finish_ok(
            state,
            answer=answer_text,
            refused=False,
            refusal_reason="",
            report=report,
            total_ms=total_ms,
        )

    # ------------------------------------------------------------ stages

    async def _retrieve(
        self,
        state: _RunState,
        retriever,
        *,
        query: str,
        round_no: int,
        refined: bool = False,
    ):
        events = state.events
        started = self._clock()
        result = await retriever.retrieve(query)
        state.retrieval_ran = True
        duration_ms = (self._clock() - started) * 1000.0
        events.emit(
            "retrieval_finished",
            round=round_no,
            refined=refined,
            query=query,
            duration_ms=round(duration_ms, 1),
            n_items=len(result.items),
            n_candidates_after_threshold=result.n_candidates_after_threshold,
            reranked=result.reranked,
            total_est_tokens=result.total_est_tokens,
            first_chunk_over_budget=result.first_chunk_over_budget,
        )
        await _notify(
            state,
            "status",
            stage="retrieved",
            round=round_no,
            n_items=len(result.items),
        )
        return result

    async def _understanding_call(
        self, state: _RunState, *, question: str, result, round_no: int
    ) -> dict | None:
        """The D5 merged call; returns the parsed verdict or None (unparseable)."""
        state.budget.claim_model_call()
        messages = [
            ChatMessage(role="system", content=UNDERSTANDING_SYSTEM),
            ChatMessage(
                role="user",
                content=understanding_user_prompt(question, result.context_text),
            ),
        ]
        started = self._clock()
        response = await self._reliable_call(
            state, label=f"understanding_round{round_no}", messages=messages, stream=False
        )
        parsed = parse_understanding(response.content)
        state.events.emit(
            "understanding_finished",
            round=round_no,
            duration_ms=round((self._clock() - started) * 1000.0, 1),
            parse_ok=parsed is not None,
            sufficient=None if parsed is None else parsed["sufficient"],
        )
        await _notify(
            state,
            "status",
            stage="understanding",
            round=round_no,
            parse_ok=parsed is not None,
            sufficient=None if parsed is None else parsed["sufficient"],
        )
        return parsed

    async def _answer_call(self, state: _RunState, *, question: str, label: str):
        """One streamed answer stage (used for the answer AND the regen).

        Returns (content, meta) where meta carries finish_reason, reported
        usage, and THIS attempt's first-token perf — the DELIVERED attempt's
        value becomes the row's first_token_ms (D16 rule), decided upstream.
        """
        state.budget.claim_model_call()
        messages = [
            ChatMessage(role="system", content=ANSWER_SYSTEM),
            ChatMessage(
                role="user",
                content=answer_user_prompt(question, _context_text(state.used_items)),
            ),
        ]
        started = self._clock()
        await _notify(state, "status", stage=label, phase="answer_started")
        content, meta = await self._reliable_call(
            state, label=label, messages=messages, stream=True
        )
        state.events.emit(
            "answer_finished",
            stage=label,
            duration_ms=round((self._clock() - started) * 1000.0, 1),
            chars=len(content),
            finish_reason=meta.get("finish_reason"),
        )
        return content, meta

    # ------------------------------------------- reliability composition

    async def _reliable_call(
        self, state: _RunState, *, label: str, messages: list[ChatMessage], stream: bool
    ):
        """limiter -> breaker -> attempt, retried by the per-request budget.

        Admission happens PER ATTEMPT: every attempt that will actually
        reach the provider re-projects quota (D12) and books usage after
        its call; attempts refused by the limiter or breaker never count
        as provider attempts and never book usage.
        """
        budget, events = state.budget, state.events
        est_tokens = (
            sum(item.est_tokens for item in state.used_items) + _OUTPUT_TOKEN_ALLOWANCE
        )

        async def attempt():
            async def inner():
                decision = self._limiter.admit(est_tokens=est_tokens)
                events.emit(
                    "limiter_admitted",
                    stage=label,
                    observe_only=decision.observe_only,
                    est_booking_tokens=est_tokens,
                )
                state.provider_attempts += 1  # reached the provider
                events.emit(
                    "provider_attempt",
                    stage=label,
                    provider_attempt=state.provider_attempts,
                    stream=stream,
                )
                if stream:
                    return await self._stream_attempt(state, label, messages)
                return await self._complete_attempt(state, label, messages)

            return await self._breaker.call(inner)

        executor = RetryExecutor(
            self._retry_policy,
            sleeper=self._sleeper,
            on_retry=_make_on_retry(state, label),
        )
        return await executor.run(attempt, label=label)

    async def _complete_attempt(self, state: _RunState, label: str, messages):
        budget = state.budget
        timeout_s = min(
            self._pipeline.provider_call_timeout_s, max(0.05, budget.remaining_ms() / 1000.0)
        )
        try:
            async with asyncio.timeout(timeout_s):
                response = await self._provider.complete(
                    messages,
                    model=self._model,
                    temperature=0.0,
                    timeout_s=timeout_s,
                )
        except TimeoutError as error:
            raise TransientNetworkError(
                f"{label}: provider call exceeded {timeout_s:.1f}s watchdog"
            ) from error
        usage = response.usage
        # D21 ORDER: book BEFORE observing. The header snapshot of THIS
        # response already reflects this call's consumption, so observing
        # after booking would double-count it (measured WP10: projected
        # 2924 - 2737 = 187 -> 8 spurious local blocks). Booking first
        # covers the header-less case; the observation then resets it.
        self._book_usage(
            state,
            label=label,
            usage=usage,
            est_fallback=_estimate_tokens("".join(m.content for m in messages))
            + _OUTPUT_TOKEN_ALLOWANCE,
        )
        self._observe_limits(state, response.limits)
        state.events.emit(
            "provider_call",
            stage=label,
            transport_ms=round(response.transport_ms, 1),
            finish_reason=response.finish_reason,
            usage_reported=usage is not None,
            prompt_tokens=None if usage is None else usage.prompt_tokens,
            completion_tokens=None if usage is None else usage.completion_tokens,
        )
        if usage is not None:
            state.tokens.add_reported(
                prompt=usage.prompt_tokens, completion=usage.completion_tokens
            )
        return response

    async def _stream_attempt(self, state: _RunState, label: str, messages):
        budget, events = state.budget, state.events
        watchdog_s = min(
            self._pipeline.stream_first_token_watchdog_s, max(0.05, budget.remaining_ms() / 1000.0)
        )
        agen = self._provider.stream(messages, model=self._model, temperature=0.0)
        parts: list[str] = []
        first_token_perf: float | None = None
        finish_reason: str | None = None
        usage: ChatUsage | None = None
        attempt_marker = state.provider_attempts
        try:
            try:
                first = await asyncio.wait_for(
                    agen.__anext__(), timeout=watchdog_s
                )
            except asyncio.TimeoutError as error:
                raise TransientNetworkError(
                    f"{label}: first-token watchdog {watchdog_s:.1f}s expired"
                ) from error
            if first.content_delta:
                parts.append(first.content_delta)
                first_token_perf = self._clock()
                await _notify(state, "token", delta=first.content_delta)
                events.emit(
                    "answer_first_token",
                    stage=label,
                    attempt=attempt_marker,
                    ttft_ms=round(
                        (first_token_perf - state.offered_perf) * 1000.0, 1
                    ),
                )
            try:
                async with asyncio.timeout(max(0.05, budget.remaining_ms() / 1000.0)):
                    async for chunk in agen:
                        if chunk.content_delta:
                            if first_token_perf is None:
                                first_token_perf = self._clock()
                                events.emit(
                                    "answer_first_token",
                                    stage=label,
                                    attempt=attempt_marker,
                                    ttft_ms=round(
                                        (first_token_perf - state.offered_perf)
                                        * 1000.0,
                                        1,
                                    ),
                                )
                            parts.append(chunk.content_delta)
                            await _notify(state, "token", delta=chunk.content_delta)
                        if chunk.finish_reason:
                            finish_reason = chunk.finish_reason
                        if chunk.usage is not None:
                            usage = chunk.usage
            except asyncio.TimeoutError as error:
                raise DeadlineExceeded(
                    "request deadline reached during answer stream",
                    elapsed_ms=round(
                        (self._clock() - state.offered_perf) * 1000.0, 1
                    ),
                ) from error
        except BaseException as error:
            if parts:
                # partial content is discarded and never user-visible: the
                # row's first_token_ms stays empty, the event file explains
                events.emit(
                    "partial_stream_discarded",
                    stage=label,
                    attempt=attempt_marker,
                    discarded_chars=len("".join(parts)),
                )
                if not isinstance(error, asyncio.CancelledError):
                    await _notify(
                        state,
                        "reset",
                        reason="partial_discarded",
                        stage=label,
                        attempt=attempt_marker,
                    )
            raise
        finally:
            await _aclose_quietly(agen)

        # D21 ORDER: book BEFORE observing (same rationale as _complete_attempt).
        self._book_usage(
            state,
            label=label,
            usage=usage,
            est_fallback=_estimate_tokens(
                "".join(m.content for m in messages) + "".join(parts)
            ),
        )
        self._observe_limits(state, getattr(self._provider, "last_limits", None))
        events.emit(
            "provider_call",
            stage=label,
            stream=True,
            finish_reason=finish_reason,
            usage_reported=usage is not None,
            prompt_tokens=None if usage is None else usage.prompt_tokens,
            completion_tokens=None if usage is None else usage.completion_tokens,
        )
        if usage is not None:
            state.tokens.add_reported(
                prompt=usage.prompt_tokens, completion=usage.completion_tokens
            )
        events.emit(
            "answer_completed",
            stage=label,
            ts=utc_now_iso_ms(),
            chars=len("".join(parts)),
        )
        return "".join(parts), {
            "finish_reason": finish_reason,
            "usage": usage,
            "first_token_perf": first_token_perf,
        }

    def _observe_limits(self, state: _RunState, snapshot) -> None:
        """Observe headers ONLY when the provider actually sent some — an
        empty stand-in snapshot must never wipe a real observation."""
        if snapshot is None:
            return
        if any(
            value is not None
            for value in (
                snapshot.rpd_limit,
                snapshot.rpd_remaining,
                snapshot.tpm_limit,
                snapshot.tpm_remaining,
                snapshot.rpm_limit,
                snapshot.rpm_remaining,
            )
        ):
            self._limiter.observe(snapshot)
            state.events.emit(
                "limits_observed",
                rpd_remaining=snapshot.rpd_remaining,
                tpm_remaining=snapshot.tpm_remaining,
                rpm_remaining=snapshot.rpm_remaining,
            )

    def _book_usage(
        self, state: _RunState, *, label: str, usage: ChatUsage | None, est_fallback: int
    ) -> None:
        """Limiter booking: reported tokens when the provider sent them,
        else an explicit estimate — booking only ever affects admission."""
        if usage is not None and (
            usage.prompt_tokens is not None or usage.completion_tokens is not None
        ):
            tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
            basis = "reported"
        else:
            tokens = est_fallback
            basis = "estimated"
        self._limiter.report_usage(requests=1, tokens=tokens)
        state.events.emit("token_booking", stage=label, basis=basis, tokens=tokens)

    # ----------------------------------------------------------- finishing

    def _finish_ok(
        self,
        state: _RunState,
        *,
        answer: str,
        refused: bool,
        refusal_reason: str,
        report: CitationReport | None,
        total_ms: float,
    ) -> RelayOutcome:
        state.events.emit(
            "outcome_typed",
            status="ok",
            outcome_kind="refusal" if refused else "completed",
            citations_valid=None if report is None else report.valid,
        )
        row = RequestRecord(
            request_id=state.request_id,
            scenario=state.scenario,
            source=self._source,
            start_time=state.start_time,
            first_token_ms=_first_token_ms(state),
            total_ms=total_ms,
            status="ok",
            retries=max(0, state.provider_attempts - 1),
            tokens_in=state.tokens.actual_in,
            tokens_out=state.tokens.actual_out,
            retrieval_setting=state.retrieval_setting,
            answer_grade=None,  # graded at WP11, never auto-assigned here
        )
        if self._recorder is not None:
            self._recorder.record(
                row, outcome_kind="refusal" if refused else "completed"
            )
        return RelayOutcome(
            request_id=state.request_id,
            scenario=state.scenario,
            retrieval_setting=state.retrieval_setting,
            source=row.source,
            status="ok",            outcome_kind="refusal" if refused else "completed",
            answer=answer,
            refused=refused,
            refusal_reason=refusal_reason,
            citations=report,
            used_items=[_item_view(item) for item in state.used_items],
            first_token_ms=_first_token_ms(state),
            total_ms=total_ms,
            retries=row.retries,
            tokens_in=state.tokens.actual_in,
            tokens_out=state.tokens.actual_out,
            outcome=None,
            events_path=state.events.path,
        )

    def _finish_refusal(self, state: _RunState, reason: str) -> RelayOutcome:
        state.refused = True
        state.refusal_reason = reason
        answer = _REFUSAL_TEMPLATE.format(reason=reason)
        state.events.emit("refusal", reason=reason)
        return self._finish_ok(
            state,
            answer=answer,
            refused=True,
            refusal_reason=reason,
            report=None,
            total_ms=(self._clock() - state.offered_perf) * 1000.0,
        )

    def _finish_typed(
        self, state: _RunState, error: BaseException, *, unexpected: bool = False
    ) -> RelayOutcome:
        status, kind = map_outcome(error)
        if unexpected:
            kind = f"unexpected_{type(error).__name__}"
        state.events.emit(
            "outcome_typed", status=status, outcome_kind=kind, error=str(error)
        )
        total_ms = (self._clock() - state.offered_perf) * 1000.0
        if self._recorder is not None:
            self._recorder.record_outcome(
                request_id=state.request_id,
                scenario=state.scenario,
                source=self._source,
                start_time=state.start_time,
                total_ms=total_ms,
                outcome=error,
                first_token_ms=_first_token_ms(state),
                retries=max(0, state.provider_attempts - 1),
                tokens=state.tokens,
                retrieval_setting=(
                    state.retrieval_setting if state.retrieval_ran else "not_applicable"
                ),
            )
        return RelayOutcome(
            request_id=state.request_id,
            scenario=state.scenario,
            retrieval_setting=state.retrieval_setting,
            source=self._source,
            status=status,
            outcome_kind=kind,
            answer="",
            refused=False,
            refusal_reason="",
            citations=None,
            used_items=[_item_view(item) for item in state.used_items],
            first_token_ms=_first_token_ms(state),
            total_ms=total_ms,
            retries=max(0, state.provider_attempts - 1),
            tokens_in=state.tokens.actual_in,
            tokens_out=state.tokens.actual_out,
            outcome=error,
            events_path=state.events.path,
        )


# ------------------------------------------------------------------ helpers


def _make_on_retry(state: _RunState, label: str):
    def on_retry(attempt_no: int, delay_s: float, error: BaseException) -> None:
        state.events.emit(
            "retry_scheduled",
            stage=label,
            attempt=attempt_no,
            delay_s=round(delay_s, 3),
            error_kind=type(error).__name__,
        )
        # sync callback inside the retry executor: schedule the (async)
        # live-stream marker instead of blocking the retry path (WP8)
        asyncio.get_running_loop().create_task(
            _notify(
                state,
                "status",
                stage=label,
                phase="retry",
                attempt=attempt_no,
                delay_s=round(delay_s, 3),
                error_kind=type(error).__name__,
            )
        )
        state.budget.claim_retry()  # per-request allowance; may raise

    return on_retry


def _first_token_ms(state: _RunState) -> float | None:
    if state.first_token_perf is None:
        return None  # empty cell in the CSV — never zero-substituted
    return round((state.first_token_perf - state.offered_perf) * 1000.0, 1)


def _estimate_tokens(text: str) -> int:
    return max(1, round(len(text or "") / 4))


def _context_text(items: list) -> str:
    from app.retrieval.retriever import _block_text

    return "\n\n".join(_block_text(item) for item in items)


def _item_view(item) -> dict:
    return {
        "chunk_id": item.chunk_id,
        "paper_key": item.paper_key,
        "versioned_id": item.versioned_id,
        "page": item.page,
        "section": item.section,
        "score": round(float(item.score), 4),
    }


async def _aclose_quietly(agen) -> None:
    import contextlib

    with contextlib.suppress(Exception):
        await agen.aclose()


class _NullLog:
    """Sink-less mode: events go nowhere (unit tests that don't need files)."""

    path = None

    def emit(self, kind: str, **fields: object) -> None: ...

    def close(self) -> None: ...
