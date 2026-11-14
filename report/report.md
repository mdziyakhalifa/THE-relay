# The Relay — Two-Page Engineering Report

**Date:** 2026-11-12 · **Suite:** 209/209 tests passing · **Real-Groq spend:** 16 offers / 13,528 tokens (every batch user-approved) · **Evidence:** `results/results.csv`, `results/logs/wp10_summary.json`, `evaluations/eval_results.json`, `docs/decision_log.md` (D1–D22)

The Relay answers engineering questions using **only** three pinned arXiv papers — *Attention Is All You Need* (1706.03762**v7**), *Moshi* (2410.00037**v1**), *Dynamic Chunking / H-Net* (2507.07955**v2**) — through Groq (`openai/gpt-oss-20b`) or a self-hosted stand-in provider. Its defining contract is honesty: every answer must cite `[arXiv:<id> p.N]` markers that match the evidence actually delivered, or the system refuses; and every *offered* request produces exactly one row in a locked 12-column `results.csv` (D13) — successes, refusals, quota blocks, and failures all look alike in the record. Heavy load runs against the stand-in (D15, scripted provider); real-Groq traffic is user-gated and budgeted. All numbers below are **MEASURED** unless explicitly labeled otherwise.

## 1. Design decisions that shaped the measurements

| Decision | What it bought |
|---|---|
| D1/D14 — one shared OpenAI-compatible HTTPX transport for Groq *and* the stand-in | one code path; load results and real behavior differ only by provider |
| D4 — page-pinned chunks (never cross pages) | every citation resolves to a real page; no synthetic page blobs |
| D5 — understanding + classification merged into one call (≤1 refinement round) | fewer provider calls per question |
| D6 — hand-rolled asyncio state machine | observable staging; no framework opacity |
| D8 — daily-quota 429s are never retried | instant, honest rows instead of doomed retries |
| D12 — limiter = header-observation projector | never assumes free-tier limits; observe-only until headers arrive |
| D16 — citation contract; TTFT = first visible token | "first token" cannot be gilded by hidden pre-work |
| D17 — validation (400) and duplicate (409) checks **before** offering | rejected requests produce no row, by design |
| D20/D21 — breaker counts provider health only; limiter books before observing + minute-window aging | quota storms cost retry time, never a false health-trip; no stale-shut projection |

## 2. Measured results

### 2.1 API layer (WP8)

Boot warmup measured: **cold TTFT 27,626.9 ms → warm 113.2 ms** — a warmup pass at boot removes the first-query cost. HTTP outcomes map deterministically: ok/refusal→200, rate_limited→429, timeout→504, capacity_rejected→503, provider fault→502, unexpected→500. Streaming SSE uses the grammar `status|token|reset|done` with a bounded queue (1024) and counted drops; a mid-stream client disconnect records an honest `user_cancelled` row.

### 2.2 Real-Groq calibration (WP9, D18)

The account exposes 11 models and **no** `llama-3.1-8b-instant`; `openai/gpt-oss-20b` was selected from measurement. Measured limits: **RPD 1000 / TPM 8000 / no RPM header** ( Groq emits no RPM header for this account ). A full real RAG round-trip (retrieve → understand → answer with valid citations) measured **wire TTFT ≈ 446 ms**. gpt-oss-20b's reasoning tokens are excluded from `content` but counted in usage — the recorder books them honestly. TPM 8000 works out to ~2–3 RAG calls per minute. **0 real 429s** occurred during calibration.

### 2.3 Load behavior (WP10, stand-in — never presented as real-Groq)

| Scenario | Offers→rows | Measured outcome |
|---|---|---|
| baseline | 60→60 ok | TTFT median 855.5 ms, p95 905; total median 1698.6 ms |
| burst (25 concurrent vs in-flight limit 8) | 60→60 ok | TTFT median 3003.6 ms — the excess is **measured queue wait**, not provider latency |
| minute429 | 20→20 | pre-D20: 1 rate_limited + 19 breaker_open; **re-measured under D20: 20× rate_limited, 0 breaker_open, retries_sum 40** |
| daily429 | 12→12 | 12× rate_limited, retries_sum 0 (D8 never-retry), rows recorded in 38–134 ms |
| breaker | 35→35 | 25 breaker_open refusals + full recovery (10/10 ok) after the cooldown |

Load testing earned its keep twice: it exposed a **LogRecord reserved-field bug** in the breaker (crashed logging under concurrency; fixed with a regression test) and, on the real probes, a **limiter projection double-count** — the runner observed a response's headers *then* booked that same call's tokens, so the projection read `2924 − 2737 = 187` and blocked 8 consecutive real offers locally (measured ratchet: no traffic → no new headers → no recovery). D21 fixed both halves: book *before* observe, and age minute-window bookings out after 60 s (daily/RPD bookings never age). Re-verified at the exact measured numbers (unit), stand-in baseline re-run clean (60/60, TTFT median 864.4 ms), suite 209/209.

### 2.4 Real probes (WP10, user-approved 10 sequential offers, 30 s pacing)

**10/10 honest rows:** 1 real answer (**TTFT 1682.7 ms**, citations valid), **1 honest refusal** — the requested fact (Glorot/Xavier weight initialization) is *absent from all chunk stores*, so refusing was the correct behavior, verified post-hoc — and 8 `quota_block_local` rows recorded in 10–26 ms with zero provider traffic (the pre-D21 projection; the blocks were correct rows *for the buggy projection*, which the fix removed; not re-run per user decision).

### 2.5 Retrieval evaluation (WP11, D22 — stand-in, zero real traffic)

16 fact-checked questions (5 per paper + 1 cross-paper); ground truth = every store chunk containing the expected facts (any-of delivery counts); deterministic string grading, no LLM judge.

| Setting | Chunks | Rerank | Correct | GT delivered | TTFT median |
|---|---|---|---|---|---|
| A (fine) | 387 | no | 8/16 (50.0%) | 7/15 | 32.8 ms |
| B (balanced) | 250 | no | 10/16 (62.5%) | 9/15 | 33.0 ms |
| C (coarse) | 173 | **yes** | **12/16 (75.0%)** | 11/15 | 444.6 ms |

**All 18 misses are retrieval misses** — zero invalid citations, zero refusals, zero missing facts across 48 rows. Four questions fail in *every* config (attention: label smoothing, positional encoding; Moshi: Depth Transformer, Inner Monologue) — MiniLM retrieves topically-adjacent chunks instead. C's cross-encoder reranker recovers 2 more B-misses and buys the best accuracy at **+~412 ms** retrieval latency. A grader-wording artifact ("real time" vs "real-time") was caught by a per-miss audit, fixed, and all settings re-measured; both measurements are kept in `results.csv`.

## 3. Honest limitations

- **Answer content on the stand-in is canned by construction** (D15): the evaluation measures retrieval quality, the citation pipeline, refusal behavior, and row honesty — *not* real generation. A real-Groq evaluation (~50–60K tokens) was estimated and **declined by user decision**; real-content quality is unmeasured, not claimed.
- **Real-Groq evidence is 16 offers total** (6 calibration + 10 probes). Sustained real-load behavior is not measured.
- **TPM 8000** caps real throughput at ~2–3 RAG calls/min (measured); quota storms cost retry time by design (D20), not breaker trips.
- **D13 hard-death edge:** a killed process can leave in-flight offers without rows (documented; clean re-runs are the evidence).
- The stand-in tokenizer is a documented approximation (D15), not Groq's tokenizer.
- Retrieval has no query rewriting beyond one refinement round; the 4 stable misses are characterized, not fixed.

## 4. Conclusion

The reliability layers did their measured jobs: every failure mode — quota, saturation, breaker trips, disconnects, validation — produced an honest, typed row; the breaker stayed a provider-health signal (D20) and the limiter learned to age with the provider's own window (D21). The evaluation isolates the one real weakness: retrieval recall on 4 of 16 question types, with a measured accuracy/latency dial (A/B/C) to turn. What remains unmeasured is stated as unmeasured — which is the point of the instrument.

**Reproduce:** `python -m scripts.run_load_tests --scenario <name>` · `python -m scripts.run_eval --setting B` · `python -m scripts.run_api` (from `the-relay/`, venv active; real traffic requires `.env` and user sign-off).
