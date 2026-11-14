# THE-relay
An evidence-grounded AI research assistant and LLM reliability testing platform built with FastAPI, React, RAG, Groq, and a simulated LLM provider. Features streaming responses, citation validation, retrieval evaluation, load testing, and performance monitoring.

# The Relay

A production-oriented AI research assistant that answers engineering questions using **only three pinned arXiv papers**, with page-grounded citations, explicit refusal behavior, fault-tolerant provider access, and fully observable, recomputable request outcomes.

> **Status:** complete (WP0–WP12, 2026-11-12).  
> **Tests:** 209/209.  
> **Report:** `report/report.md` · **Decision log:** `docs/decision_log.md` (D1–D22) · **Progress:** `docs/progress_checklist.md`

---

## Table of contents

- [Overview](#overview)
- [Core principles & honesty contract](#core-principles--honesty-contract)
- [System architecture](#system-architecture)
- [Repository layout](#repository-layout)
- [Component deep dive](#component-deep-dive)
  - [1. Core configuration & cross-cutting concerns](#1-core-configuration--cross-cutting-concerns)
  - [2. Retrieval subsystem](#2-retrieval-subsystem)
  - [3. Provider transport layer](#3-provider-transport-layer)
  - [4. Reliability primitives](#4-reliability-primitives)
  - [5. Orchestrator (RelayRunner)](#5-orchestrator-relayrunner)
  - [6. API layer](#6-api-layer)
  - [7. Stand-in service](#7-stand-in-service)
  - [8. Observability, records & artifacts](#8-observability-records--artifacts)
- [End-to-end request lifecycle](#end-to-end-request-lifecycle)
- [Data contracts](#data-contracts)
- [Retrieval configurations](#retrieval-configurations)
- [Configuration & environment](#configuration--environment)
- [Running the system](#running-the-system)
- [Testing, evaluation & load experiments](#testing-evaluation--load-experiments)
- [Measured results](#measured-results)
- [Design decisions & tradeoffs](#design-decisions--tradeoffs)
- [Known limitations](#known-limitations)
- [Security](#security)

---

## Overview

**The Relay** is an engineering-focused RAG assistant constrained to a fixed corpus of three papers:

1. **Attention Is All You Need** — `arXiv:1706.03762v7`
2. **Moshi: a speech-text foundation model for real-time dialogue** — `arXiv:2410.00037v1`
3. **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling** — `arXiv:2507.07955v2`

The system is designed around *production behavior*, not demo behavior: every offered request is recorded, provider/reliability failures are typed and observable, citations are validated against delivered evidence, and experiments distinguish real-provider measurements from stand-in (simulator) measurements.

### Key capabilities
- **Grounded QA over a fixed corpus** with page-pinned evidence chunks and bracketed citations (`[arXiv:<id> p.N]`).
- **Explicit refusal** when evidence is missing/insufficient or citations cannot be made valid.
- **Dual provider support**: real Groq (OpenAI-compatible HTTP) and a local scripted stand-in service, sharing one transport implementation.
- **Reliability controls**: admission/capacity gate, quota projection limiter, retries, circuit breaker, and per-request budgets/deadlines.
- **Observability by construction**: `results.csv` (one row per offered request), per-request event logs, and structured logging/stage metrics.
- **HTTP API** (`/ask`, `/ask/stream`, `/health`) with SSE streaming semantics and honest disconnect handling.

---

## Core principles & honesty contract

The architecture is built to make behavior auditable and hard to misrepresent:

- **Corpus-bound answers:** claims must be supported by retrieved evidence from the three pinned papers only.
- **Citation validity:** a non-refusal answer must cite delivered evidence; cited `(arXiv id, page)` pairs are validated against evidence actually provided to the model. Invalid citations trigger at most one regeneration; if still invalid/failing, the system follows the recorded honesty/availability rules (see orchestrator behavior).
- **One row per offered request (`D13`):** every offered request yields exactly one row in `results/results.csv` with the locked schema  
  `request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`.
  - `status ∈ {ok, rate_limited, timeout, error}`
  - `first_token_ms` is the time from **original offer** to first delivered content token, or empty if none was delivered (never `0`)
  - `source ∈ {real, stand-in}`
  - `retrieval_setting` is a stable config id (`A|B|C`) or `not_applicable` when retrieval never ran
- **No invented usage:** token counts in results come from provider-reported usage aggregation only (estimates may be used for admission/booking decisions but are tracked separately and never substituted into the CSV contract).
- **Simulator results are labeled:** load/eval runs against the stand-in are recorded as `source=stand-in` and must not be presented as Groq behavior.
- **User-gated real traffic:** real Groq calibration/probes are explicit, budgeted operations (not routine paths).

---

## System architecture

### High-level component diagram

```text
                              +---------------------------+
                              |        API Clients        |
                              |  (HTTP / SSE consumers)   |
                              +-------------+-------------+
                                            |
                                            v
+-----------------------------------------------------------------------------------+
|                                   app/api/service.py                               |
|  FastAPI: POST /ask, POST /ask/stream, GET /health                                |
|  - request validation (before offer)  - outcome->HTTP mapping  - SSE pump        |
|  - start_time captured at HTTP entry  - disconnect -> honest cancellation        |
+-------------------------------------------+---------------------------------------+
                                            | (offered request + token_sink?)
                                            v
+-----------------------------------------------------------------------------------+
|                              app/orchestrator/runner.py                           |
|                                   RelayRunner (D6/D16)                            |
|  admit -> retrieve -> understand(D5) -> [refine round] -> answer stream           |
|           -> citation validate -> [1 regen max] -> finalize + record              |
|  owns: budgets/deadline, TTFT semantics, refusal logic, event emission            |
+----+-------------------+--------------------+-------------------+----------------+
     |                   |                    |                   |
     | capacity/budget   | retrieve            | LLM calls         | record/events
     v                   v                    v                   v
+-----------+   +------------------+   +----------------+   +----------------------+
|reliability|   |retrieval/        |   |providers/      |   |reliability/recorder  |
|gate/budget|   |retriever,index,  |   |groq,standin,   |   |+ orchestrator/events |
|limiter,   |   |embedder,reranker |   |compat transport|   |(results.csv, JSONL)  |
|retry,     |   +------------------+   +--------+-------+   +----------------------+
|breaker    |                                  |
+-----------+                                  | OpenAI-compatible HTTP
                                               v
                         +-------------------------------------------+
                         |  Real Groq API  OR  Local Stand-in API    |
                         |         (app/standin FastAPI service)     |
                         +-------------------------------------------+
```

### Architectural style
- **Hand-rolled async state machine** (`RelayRunner`) instead of an orchestration framework for explicit control flow, observability, and interviewability.
- **Single shared provider transport** for real and simulated providers to prevent behavioral drift.
- **Composable reliability**: transport is pure I/O; retry/limiter/breaker/capacity/budgets are composed around attempts in the orchestrator.
- **Evidence-first retrieval design**: page-pinned chunks, citation-ready provenance, and deterministic index configs (`A/B/C`).

---

## Repository layout

```text
the-relay/
├─ app/
│  ├─ api/                 # FastAPI endpoints, SSE, HTTP outcome mapping, observability hooks
│  ├─ core/                # Typed settings, logging setup, request context, stage metrics
│  ├─ orchestrator/        # RelayRunner state machine, prompts, citation validation, per-request events
│  ├─ providers/           # Shared OpenAI-compatible transport + Groq/stand-in provider adapters
│  ├─ reliability/         # Capacity gate, budgets, limiter, retry, breaker, recorder, error taxonomy
│  ├─ retrieval/           # Download/extract/chunk/embed/index/retrieve/rerank + paper registry
│  └─ standin/             # Scripted stand-in LLM service (tokenizer + FastAPI simulator)
├─ scripts/                # CLI entry points: ingest/build, API, eval, load tests, calibration, utilities
├─ papers/
│  ├─ original_pdfs/       # Pinned source PDFs + manifest
│  ├─ extracted_text/      # Page-extracted text/chunk artifacts
│  └─ indexes/             # Built retrieval indexes: config_a, config_b, config_c
├─ results/
│  ├─ results.csv          # Locked request-level record (one row per offered request)
│  ├─ retrieval_settings.json
│  ├─ logs/                # Load/calibration logs and summaries
│  └─ plots/               # Experiment plots
├─ evaluations/            # Evaluation outputs
├─ report/                 # Final engineering report artifacts
├─ docs/                   # Decision log, progress checklist, interview/prep notes
├─ tests/
│  ├─ unit/                # Component-level tests
│  └─ integration/         # Stand-in wire/e2e, runner e2e, stream disconnect tests
├─ frontend/               # Frontend workspace (Vite/TypeScript scaffolding)
├─ .env.example            # Environment variable template (no secrets)
├─ pytest.ini
├─ requirements.txt
└─ requirements.lock.txt
```

> Note: this project lives under `the-relay/` in the workspace. Run commands from that directory unless stated otherwise.

---

## Component deep dive

### 1. Core configuration & cross-cutting concerns

**Key modules:** `app/core/settings.py`, `app/core/logging_setup.py`, `app/core/request_context.py`, `app/core/stage_metrics.py`

- **Typed settings (`pydantic-settings`)** loaded from env vars with prefix `RELAY__` and nested delimiter `__`, plus optional `.env`.
- **No secrets in settings objects:** `GROQ_API_KEY` is read from the environment at provider-use time, not stored in settings.
- **Centralized defaults for:**
  - pipeline budgets (`max_model_calls`, `max_retrieval_rounds`, request deadline, provider timeouts/watchdogs)
  - retry policy (attempts/backoff/jitter/request retry allowance)
  - circuit breaker policy (threshold/window/open duration/half-open probes)
  - app capacity (in-flight + queue limits; tighter real-provider cap)
  - provider base URLs/models (real model left unset until measured/calibrated)
  - retrieval defaults (config B-style defaults; A/B/C are evaluated as explicit configs)
- **Structured logging & request context** support consistent, machine-readable diagnostics and correlation.
- **Stage metrics** provide timing/measurement hooks for pipeline observability.

### 2. Retrieval subsystem

**Key modules:** `app/retrieval/download.py`, `extract.py`, `chunker.py`, `papers_registry.py`, `embedder.py`, `indexer.py`, `retriever.py`, `reranker.py`

**Goal:** produce citation-ready, page-accurate evidence for a fixed paper corpus.

#### Pipeline stages (offline/ingest/build)
1. **Paper registry & download**: pinned paper identities/versions are tracked explicitly; PDFs and manifests live under `papers/`.
2. **Extraction**: PDF text extracted with page awareness (page index integrity verified during development).
3. **Chunking (page-pinned, D4)**: chunks never cross PDF page boundaries, so page citations are exact by construction. Section labels are recorded when confidently detected; otherwise marked conservatively (never guessed).
4. **Embedding**: local `sentence-transformers` embeddings (`all-MiniLM-L6-v2`, 384-dim) for retrieval vectors.
5. **Indexing (D2)**: normalized vectors stored in NumPy-backed indexes (flat cosine search), suitable for a small corpus (hundreds of chunks).
6. **Config variants (A/B/C)**: prebuilt indexes with different chunking/rerank behavior for comparison.

#### Online retrieval (`Retriever`)
- **Async facade over CPU-bound work:** query embedding, search, and reranking run in worker threads (`asyncio.to_thread`) to avoid blocking the event loop.
- **Flow per query:** embed query → bi-encoder cosine search (top-k, threshold) → optional cross-encoder rerank (config-dependent) → token-budget trim → assemble citation-ready context blocks.
- **Reranking semantics (D9):** cross-encoder reorders survivors only; its raw score scale is not mixed with cosine thresholds/scores.
- **Context budget handling (D10):** greedy best-first trim by estimated tokens; if the first chunk alone exceeds budget it is still included (flagged) rather than dropping all evidence.
- **Provenance-rich evidence items:** each item carries chunk/paper/version/page/section/text/score/token estimate, enabling deterministic citation validation.

### 3. Provider transport layer

**Key modules:** `app/providers/compat.py`, `app/providers/groq.py`, `app/providers/standin.py`, `app/providers/__init__.py`

- **One shared OpenAI-compatible transport (`OpenAICompatTransport`)** implements:
  - request construction (`/chat/completions`, stream options)
  - non-stream and SSE stream parsing
  - transport timing (`transport_ms`)
  - HTTP status → typed error mapping
  - rate-limit header capture into limiter snapshots
- **Provider adapters are thin wrappers:**
  - **Groq provider:** adds base URL + API key (from environment) / auth headers
  - **Stand-in provider:** points to local base URL, no auth; mirrors the same wire contract
- **Pure transport boundary:** no retries, breaker, limiter, or orchestration logic inside providers. This is composed by the runner/reliability layer to keep behavior explicit and testable.
- **Honesty in parsing:** missing/unknown usage is preserved as unavailable (not invented); malformed responses become typed errors.

### 4. Reliability primitives

**Key modules:** `app/reliability/errors.py`, `retry.py`, `breaker.py`, `capacity.py`, `budget.py`, `limiter.py`, `recorder.py`

This layer encodes failure semantics and enforces the project’s observability/honesty guarantees.

- **Typed error taxonomy (`errors.py`)**
  - Distinguishes retryable vs non-retryable/provider-health vs quota/backpressure vs configuration/deadline/cancellation cases.
  - Supports outcome mapping into the locked CSV `status` + typed `outcome_kind`.

- **Capacity gate (`capacity.py`)**
  - Bounded in-flight + bounded queue admission.
  - Returns honest backpressure rejections when saturated (mapped to HTTP 503 at API boundary; CSV remains `error/capacity_rejected`).

- **Request budgets (`budget.py`)**
  - Per-request guards: max model calls, max retrieval rounds, retry allowance, wall-clock deadline.
  - Prevents runaway orchestration and enforces explicit timeout behavior.

- **Retry executor (`retry.py`)**
  - Exponential backoff + jitter with attempt caps and per-request retry budget.
  - Honors retry-after semantics where applicable.
  - **D8:** daily quota exhaustion is never retried.

- **Circuit breaker (`breaker.py`, D11/D20)**
  - Tracks **provider-health faults only** (e.g. transient/network/5xx-class health signals as implemented).
  - Quota pressure is excluded from breaker health decisions (owned by limiter/retry) per measured/load-driven revisions.
  - Supports open/cooldown/half-open probe behavior to fail fast during provider sickness while allowing recovery.

- **Header-observation quota limiter (`limiter.py`, D12/D21)**
  - Observe-only until real rate-limit headers arrive.
  - Projects remaining quota from last observations minus locally booked usage.
  - Prioritizes daily exhaustion handling; replaces projections on fresh observations.
  - Includes timestamped/aging-aware booking behavior to avoid stale minute-window ratchets (post-measurement fix).

- **Recorder (`recorder.py`, D13)**
  - Enforces the **locked 12-column `results.csv` schema** structurally.
  - Validates enums/formats, prevents duplicate request IDs (including across restarts/preloaded state), and preserves stable row ordering/formatting behavior.
  - Aggregates provider-reported token usage only; separates concerns from limiter booking estimates.
  - Provides outcome mapping (`map_outcome`) and row recording APIs used by the runner.

### 5. Orchestrator (RelayRunner)

**Key modules:** `app/orchestrator/runner.py`, `prompts.py`, `citations.py`, `events.py`

`RelayRunner` is the core state machine (D6) and the source of truth for request semantics.

#### Responsibilities
- Admit requests through capacity control and execute the full pipeline within budgets/deadlines.
- Perform retrieval (round 1, optional refined round 2) and merged understanding/classification (D5).
- Stream/generate an answer, validate citations, and optionally regenerate once.
- Compose reliability per provider attempt (budget claim → retry/limiter/breaker/admission path as implemented).
- Emit per-request events and guarantee exactly one final recorded outcome/row.
- Define TTFT and outcome semantics consistently for HTTP and load tooling.

#### Pipeline behavior (summary)
1. **Offer & admit**: request is offered (start time captured), capacity slot acquired (queue wait included in TTFT boundary).
2. **Retrieve (round 1)**: query corpus using selected retrieval setting; empty retrieval can short-circuit to honest refusal (no unnecessary provider call).
3. **Understanding call (D5)**: merged structured understanding/classification/refined-query decision.
   - unparseable output does not fabricate refusal when evidence exists (citation validator remains backstop)
   - insufficient evidence may trigger one refinement round (re-retrieve + re-understand)
4. **Answer generation (streamed)**: streamed attempt captures first delivered token timing and usage metadata.
5. **Citation validation**: checks bracketed `arXiv:<id> p.N` citations against delivered evidence.
6. **Regeneration (max 1)**: only for citation-invalid answers; delivery/TTFT/failure semantics follow D16 (delivered attempt defines TTFT; failed regen does not erase the fact pattern and final outcome is recorded honestly).
7. **Finalize**: emits typed outcome/events and records exactly one CSV row (including for expected failures/cancellation/unexpected exceptions).

#### Notable semantics
- **TTFT**: first *generated content token of the answer actually delivered*, measured from original offer time; empty if none delivered.
- **Partial stream discard**: tokens from a failed/aborted stream attempt are not surfaced as delivered output and TTFT is not fabricated.
- **Retries accounting**: `results.csv` `retries` counts provider attempts beyond the first across the pipeline; detailed per-stage breakdown is kept in event logs.
- **Per-run isolation**: token sinks and mutable run state are per request, enabling safe concurrent execution.
- **Failure isolation**: sink/notification failures must not corrupt the recorded request outcome.

### 6. API layer

**Key modules:** `app/api/service.py`, `app/api/observability.py`, `scripts/run_api.py`

FastAPI surface over a single `RelayRunner` instance.

#### Endpoints
- **`POST /ask`** (buffered)
  - Validates request before offering (no row on validation failure)
  - Runs request and maps typed outcome to HTTP status
  - Returns structured outcome payload (answer on `ok`, error metadata otherwise)

- **`POST /ask/stream`** (SSE)
  - Same pre-offer validation and start-time semantics as `/ask`
  - Streams live events from runner `token_sink` via bounded queue
  - Emits named SSE events: `status`, `token`, `reset`, plus terminal `done`
  - `done` always carries final reconciled outcome (including full answer text) so clients can recover from dropped/reset live tokens
  - Uses keepalive comments and non-blocking disconnect polling (Windows/proactor-safe behavior documented in code/decision log)

- **`GET /health`**
  - Exposes introspection for breaker state, limiter projection summary, and gate counters (useful for ops/load harness visibility)

#### HTTP mapping (D17)
- `ok` (including honest refusals) → `200`
- `rate_limited` → `429` (+ retry-after when available)
- `timeout` → `504`
- `error` with `capacity_rejected` → `503` (local backpressure)
- other `error` outcomes → `502`
- unexpected runner exception path → `500` (row already recorded)

> CSV row status semantics remain locked to `ok|rate_limited|timeout|error`; finer-grained `outcome_kind` is preserved in event logs/payloads.

#### Validation-before-offer
- Invalid JSON/body, empty question, unknown retrieval setting, bad request ID → `400`
- Duplicate request ID (already recorded) → `409`
- These do **not** create `results.csv` rows because the request was never offered.

#### Streaming robustness
- Bounded SSE queue protects the runner from slow consumers; overflow drops live token events (counted) but `done` reconciles.
- Client disconnect cancels the run and triggers honest cancellation recording.

### 7. Stand-in service

**Key modules:** `app/standin/service.py`, `app/standin/tokenizer.py`, `scripts/run_standin.py`

A local, OpenAI-compatible **scripted LLM simulator** used for deterministic development, reliability/load testing, and evaluation without consuming real Groq quota.

#### Design goals
- Same wire-level interface shape as the real provider (so the shared transport and orchestration code paths are exercised).
- Deterministic, documented token accounting.
- Scriptable failure modes and latency shaping for load/reliability scenarios.

#### Tokenizer (documented)
- Tokenization/counting rules are explicit and stable (word/punctuation-oriented scheme with documented framing assumptions).
- Usage is reported consistently (including stream usage when requested by contract in this service), enabling honest CSV aggregation in stand-in runs.

#### Service capabilities
- Non-stream and SSE streaming responses (`/chat/completions`-compatible behavior).
- `max_tokens` / finish reason handling and deterministic generated text behavior.
- Admin/shaping controls (mode switching, delay/TTFT/chunk pacing/length shaping, counters/reset/state inspection) for scenarios such as:
  - healthy/OK responses
  - minute-rate-limit and daily-rate-limit simulations
  - server errors
  - malformed responses
  - hangs/timeouts
  - abrupt stream/body truncation

**Important:** stand-in measurements describe simulator behavior only and are always labeled `source=stand-in`.

### 8. Observability, records & artifacts

**Key modules/artifacts:** `app/reliability/recorder.py`, `app/orchestrator/events.py`, `app/core/logging_setup.py`, `app/core/stage_metrics.py`, `results/*`, `papers/*`

- **`results/results.csv`**: canonical request ledger (locked schema, one row per offered request).
- **Per-request event files (JSONL)**: stage timings, queue wait/admission, retrieval details, provider attempts/calls, retry scheduling, limiter/booking observations, first-token/completion markers, citation checks, typed outcomes, and debug traces for discarded partial streams/regens.
- **Logs & experiment artifacts**: load-test logs/summaries, calibration logs, plots, and evaluation outputs under `results/`, `evaluations/`.
- **Retrieval metadata**: `results/retrieval_settings.json` documents stable retrieval config IDs and their meaning (A/B/C).
- **Paper artifacts**: pinned PDFs, extracted page text/chunks, and prebuilt indexes under `papers/`.

---

## End-to-end request lifecycle

```text
Client POST /ask or /ask/stream
  └─ API validates input (before offer)
       ├─ invalid -> 400/409, no row recorded
       └─ valid -> capture start_time (offer), assign/accept request_id
            └─ RelayRunner.run(...)
                 ├─ emit "offered"
                 ├─ capacity gate slot acquire (queue wait measured)
                 ├─ retrieval round 1 (embed/search/rerank/trim)
                 │    ├─ no evidence -> refusal outcome
                 │    └─ evidence -> understanding call (D5)
                 ├─ if insufficient and refinable -> retrieval round 2 + understanding
                 │    └─ still insufficient -> refusal outcome
                 ├─ streamed answer attempt(s) with reliability composition
                 │    ├─ per attempt: budget claim, retry/limiter/breaker, call/stream
                 │    ├─ track delivered first token (TTFT source)
                 │    └─ on stream failure: discard partial, record event
                 ├─ citation validation
                 │    └─ invalid -> one regen attempt (if allowed)
                 ├─ finalize typed outcome (ok/refusal/rate_limited/timeout/error)
                 ├─ append per-request events and exactly one results.csv row
                 └─ return outcome
                      ├─ /ask: JSON body + mapped HTTP status
                      └─ /ask/stream: status/token/reset events + terminal done
```

---

## Data contracts

### `results/results.csv` (locked schema, D13)

| Column | Meaning |
|---|---|
| `request_id` | Unique request identifier (one row per offered request) |
| `scenario` | Scenario/run label (e.g. API, load test, calibration) |
| `source` | `real` (Groq) or `stand-in` |
| `start_time` | Offer timestamp (ISO 8601 UTC with ms, `...Z`) |
| `first_token_ms` | Offer → first delivered content token; empty if none |
| `total_ms` | Total request duration from offer to terminal outcome |
| `status` | `ok`, `rate_limited`, `timeout`, or `error` |
| `retries` | Provider attempts beyond the first, summed across pipeline |
| `tokens_in` | Aggregated provider-reported prompt/input tokens (nullable) |
| `tokens_out` | Aggregated provider-reported completion/output tokens (nullable) |
| `retrieval_setting` | `A`, `B`, `C`, or `not_applicable` |
| `answer_grade` | Eval-time grade (`correct`, `wrong`, `refused_correctly`, `refused_wrongly`) or empty |

### Request APIs (simplified)

#### `POST /ask` request body
```json
{
  "question": "string (required, non-empty)",
  "retrieval_setting": "A|B|C (optional, default B)",
  "scenario": "string (optional, default api)",
  "request_id": "string (optional, unique; server generates if omitted)"
}
```

#### `POST /ask` response (shape)
Includes `request_id`, `status`, `outcome_kind`, `answer` (on `ok`), `refused`, citation summary, timing/retry/token fields, and `row_recorded: true` for offered requests.

#### `POST /ask/stream` SSE events
- `event: status` — lifecycle/status updates (admitted, retrieved, understanding, answer_started/retry, etc.)
- `event: token` — live content delta
- `event: reset` — client should clear buffered partial output (e.g. regen/discard cases)
- `event: done` — terminal reconciled outcome (always includes final answer payload/state)

### Event logs
Per-request JSONL event files capture detailed execution traces (queue wait, stage durations, attempts/retries, limiter observations/bookings, token accounting basis, citation checks, typed outcomes). These complement `results.csv` without changing its locked schema.

---

## Retrieval configurations

The project evaluates/uses stable retrieval setting IDs (documented in `results/retrieval_settings.json` and project docs):

| Setting | Chunks (A/M/H = 387/250/173 total corpus splits vary by config) | Chunking intent | Rerank | Notes |
|---|---:|---|---:|---|
| **A** | 387 total (40 / 200 / 147) | Smaller chunks, finer granularity | No | Better recall for narrowly stated facts in some cases |
| **B** | 250 total (27 / 129 / 94) | Balanced default | No | App default baseline; warm practical latency |
| **C** | 173 total (18 / 86 / 69) | Larger chunks | Yes (cross-encoder reorder of survivors) | Higher eval accuracy in measured stand-in eval; added rerank latency |

**Retrieval facts (measured in project artifacts/logs):**
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- Cross-encoder reranker (config C): `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranker outputs are treated as ordering logits (not thresholded on the cosine scale)

---

## Configuration & environment

### Environment loading
- Settings are read from environment variables with prefix `RELAY__` and nested delimiter `__`.
- `.env` is supported for local development (see `.env.example`).
- **Secrets:** keep `GROQ_API_KEY` in `.env` only (git-ignored). Never hardcode or log it.

### Important config groups (`app/core/settings.py`)
- `RELAY__PIPELINE__*` — model-call/retrieval/deadline/timeouts/watchdogs
- `RELAY__RETRY__*` — retry attempts/backoff/jitter/request retry budget
- `RELAY__BREAKER__*` — breaker thresholds/window/cooldown/probe behavior
- `RELAY__CAPACITY__*` — in-flight and queue limits (general + real-provider caps)
- `RELAY__GROQ__*` — Groq base URL/model settings (real model intentionally unset before calibration)
- `RELAY__STANDIN__*` — stand-in base URL
- `RELAY__RETRIEVAL__*` — default retrieval parameters

### Paths
- `results_dir` defaults to `results/`
- `papers_dir` defaults to `papers/`

---

## Running the system

> Commands below assume Windows PowerShell and execution from `the-relay/`.

### 1) Setup

```powershell
cd "C:\ai app\the-relay"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\check_env.py
copy .env.example .env   # add GROQ_API_KEY only if/when running user-approved real-Groq flows
```

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 2) Build/verify corpus artifacts (if needed)

The repository already includes extracted text and built indexes in `papers/`. For rebuild/ingest workflows, use the scripts under `scripts/` (examples):

```powershell
# Download pinned PDFs / verify integrity (as applicable)
.\.venv\Scripts\python.exe -m scripts.download_papers
.\.venv\Scripts\python.exe -m scripts.verify_pagenums

# Extract/chunk and build indexes (project-specific scripts)
.\.venv\Scripts\python.exe -m scripts.ingest_papers
.\.venv\Scripts\python.exe -m scripts.build_index
```

(Use the script help/options in each file as the source of truth for arguments.)

### 3) Run the API (stand-in default path)

```powershell
.\.venv\Scripts\python.exe -m scripts.run_api
```

`run_api.py` builds the app components, binds retrievers, boots the stand-in-backed setup by default, performs warmup, and serves:
- `POST /ask`
- `POST /ask/stream`
- `GET /health`

### 4) Example request (buffered)

```powershell
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What problem does multi-head attention solve?\",\"retrieval_setting\":\"B\"}"
```

### 5) Example request (SSE stream)

```powershell
curl -N -X POST http://127.0.0.1:8000/ask/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"How does H-Net perform dynamic chunking?\",\"retrieval_setting\":\"C\"}"
```

### 6) Real Groq runs (explicit, user-gated)

Scripts such as `scripts/calibrate_groq.py` and real-provider probe modes in load tooling perform **real provider traffic** and require explicit sign-off/budget awareness. They are documented for controlled calibration, not routine operation.

---

## Testing, evaluation & load experiments

### Automated tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The suite covers core units (retrieval, reliability, providers, orchestrator, API, stand-in tokenizer/service) and integrations (stand-in wire/e2e, runner e2e, stream disconnect behavior). Project status at completion: **209/209 tests passing**.

### Retrieval evaluation (stand-in, deterministic harness)

```powershell
.\.venv\Scripts\python.exe -m scripts.run_eval --setting A
.\.venv\Scripts\python.exe -m scripts.run_eval --setting B
.\.venv\Scripts\python.exe -m scripts.run_eval --setting C
```

Produces evaluation outputs and records honest request rows according to the same system contracts.

### Load experiments (stand-in scenarios)

```powershell
.\.venv\Scripts\python.exe -m scripts.run_load_tests --scenario baseline
.\.venv\Scripts\python.exe -m scripts.run_load_tests --scenario burst
.\.venv\Scripts\python.exe -m scripts.run_load_tests --scenario minute429
.\.venv\Scripts\python.exe -m scripts.run_load_tests --scenario daily429
.\.venv\Scripts\python.exe -m scripts.run_load_tests --scenario breaker
.\.venv\Scripts\python.exe -m scripts.run_load_tests --scenario all
```

The harness boots the same app path, applies scripted stand-in modes/shaping, verifies `rows == offers` invariants per run, and emits logs/plots/summaries under `results/`.

---

## Measured results

> Values below are project-recorded measurements (machine/run specific) and should be interpreted in context of the documented experiments and sources (`results/*`, `docs/decision_log.md`, `report/report.md`).

| Area | Measurement |
|---|---|
| Test suite | **209 / 209 passing** |
| API warmup effect (boot probe) | Cold TTFT **27,626.9 ms** → warm **113.2 ms** (dominated by initial model/index load) |
| Real Groq wire/stream latency (small probes) | Stream TTFT ~**446 ms** (measured), non-stream higher |
| Real Groq full RAG round-trip (calibration success) | TTFT **1489.4 ms**, total **1759.5 ms**, citations valid (real retrieval + real provider) |
| Real Groq measured limits (this account/model) | **RPD 1000**, **TPM 8000**, **no RPM header** reported for selected model |
| Stand-in baseline load (60 offers) | TTFT median **855.5 ms** (p95 **905.0**), total median **1698.6 ms** |
| Stand-in burst load (60 offers, high concurrency) | TTFT median **3003.6 ms** (queue-wait visible in TTFT boundary), all completed |
| Stand-in retrieval eval (16 questions) | **A:** 50.0% · **B:** 62.5% · **C:** 75.0% (rerank improves accuracy, +~400 ms) |
| Recorded outcomes | **419** honest rows in `results/results.csv` (real + stand-in sources) |

---

## Design decisions & tradeoffs

The project maintains a detailed, evidence-labeled decision log in `docs/decision_log.md` (D1–D22). Highlights:

- **D1/D14 – Shared raw HTTP transport:** `httpx` OpenAI-compatible transport for full header/body/stream control and identical behavior across Groq and stand-in.
- **D2 – NumPy flat cosine index:** simple, transparent, sufficient for a tiny corpus; not optimized for very large-scale ANN workloads.
- **D4 – Page-pinned chunking:** preserves exact page provenance for citations at the cost of occasional page-boundary splits.
- **D5 – Merged understanding/classification call:** quota-aware single structured call with defensive parsing and refinement path.
- **D6 – Custom async state machine:** explicit control flow, budgets, and observability over framework abstraction.
- **D8/D11/D12/D20/D21 – Reliability semantics refined by measurement:** daily quota never retried; breaker treated as provider-health signal (not quota pressure); limiter is header-observation based and corrected for projection/aging issues discovered under real/load evidence.
- **D13 – Locked results contract:** structural enforcement of one-row-per-offered-request and no-fabrication rules for metrics/usage.
- **D15 – Scripted stand-in:** deterministic, failure-injectable simulator to exercise the same code paths safely and reproducibly.
- **D16/D17 – TTFT, citations, SSE, and disconnect honesty:** explicit semantics for delivered content, partial-stream discard/regeneration, and client disconnect cancellation.

---

## Known limitations

- **Corpus scope is fixed** to three pinned papers; out-of-scope questions should be refused (and refusal correctness is part of the honesty model).
- **Retrieval/indexing choices favor clarity over scale** (NumPy flat search, local embedding/rerank latency tradeoffs, CPU-bound model warmup costs).
- **Stand-in != real provider behavior**: latency/failure distributions and model quality differ; only `source=real` rows represent Groq measurements.
- **Real-provider evaluation coverage is limited by budget/sign-off constraints**; some behaviors (e.g. naturally observed 429 signatures) may remain partially measured/documented depending on executed probes.
- **Platform/runtime observations** (e.g. Windows/uvicorn disconnect detection behavior, teardown noise) are documented where encountered and handled defensively, but may vary by environment.

---

## Security

- **Secrets handling:** store `GROQ_API_KEY` only in `.env` (git-ignored). Never commit, paste into chat, or log secrets.
- **Settings safety:** settings objects do not retain API keys; provider auth headers are constructed at request time.
- **Operational hygiene:** the project workflow includes leak scans before commits (see decision/progress/report artifacts for process evidence).
- **Network boundaries:** real-provider traffic is explicit and budgeted; routine development/testing can run entirely against the local stand-in.

