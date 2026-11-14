# The Relay — Decision Log

Status legend used in every entry:

- **DESIGN** — engineering choice made on reasoning; not empirically proven.
- **ASSUMPTION** — a starting value that has NOT been measured; must be replaced
  by a measured value in later work packages or by explicitly reported as
  `not measured` in the report.
- **DOCUMENTED** — taken from a provider's official documentation, with URL and
  fetch date; still to be verified against live behavior where possible.
- **MEASURED** — backed by a captured experiment artifact under `results/`.

Every entry follows: Problem → Options → Chosen → Evidence → Disadvantages →
Revisit trigger.

---

## D1 — Groq client: official SDK vs direct HTTP

- **Problem:** We must capture exact rate-limit response headers and raw 429
  bodies for calibration, and must measure timing boundaries precisely.
- **Options:** (a) `groq` Python SDK; (b) `httpx` against Groq's official
  OpenAI-compatible endpoint.
- **Chosen:** (b) `httpx`.
- **Evidence:** Groq documents an OpenAI-compatible REST API
  (console.groq.com/docs/openai) and documented rate-limit headers
  (`x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`,
  `x-ratelimit-remaining-*`, `x-ratelimit-reset-*`, `retry-after`) —
  DOCUMENTED, fetched 2026-11-10. Raw HTTP gives us headers, raw error bodies,
  and full control of streaming and timeouts. Status: DESIGN.
- **Disadvantages:** We own protocol details (usage accounting, SSE parsing)
  that the SDK would handle.
- **Revisit trigger:** If live endpoint behavior diverges from the documented
  OpenAI-compatible contract, switch to the official SDK.

## D2 — Vector index: FAISS vs ChromaDB vs NumPy

- **Problem:** Need nearest-neighbor retrieval over chunks of exactly 3 papers.
- **Options:** FAISS, ChromaDB, plain NumPy cosine.
- **Chosen:** NumPy flat cosine over normalized embeddings.
- **Evidence:** Corpus scale is tiny (≈ hundreds of chunks from three papers);
  exhaustive search cost is trivial. Status: DESIGN + to be MEASURED in WP3.
- **Disadvantages:** No ANN sublinear search; would not scale to ≥100k chunks.
- **Revisit trigger:** Corpus grows by order(s) of magnitude.

## D3 — Embeddings: sentence-transformers vs ONNX fastembed

- **Problem:** Free, local, quality embeddings for retrieval.
- **Options:** sentence-transformers (MiniLM family) vs fastembed (ONNX, no torch).
- **Chosen:** sentence-transformers `all-MiniLM-L6-v2` for retrieval;
  `cross-encoder/ms-marco-MiniLM-L-6-v2` for optional reranking (config C).
- **Evidence:** Assignment-preferred stack; widely documented; runnable offline.
  Status: DESIGN; heavy torch install is a known setup cost.
- **Disadvantages:** First install downloads torch (~2 GB disk).
- **Revisit trigger:** Install problems on Windows → switch to fastembed.

## D4 — Chunking: page-pinned, never crossing pages

- **Problem:** Citations must name an exact paper version + page; free-flowing
  chunks would make page numbers ambiguous or invented.
- **Options:** semantic chunking across pages / page-pinned chunks.
- **Chosen:** Page-pinned. A chunk never spans two PDF pages, so
  `page_number` is exact by construction. Section heading recorded only when
  confidently detected from PDF font information; otherwise `unknown`,
  never guessed. Status: DESIGN.
- **Disadvantages:** Slightly more chunks; text flow across page boundaries is
  split (rare in these papers).
- **Revisit trigger:** If retrieval quality measurably suffers at page joins.

## D5 — Question understanding + classification merged into one LLM call

- **Problem:** Free-tier Groq quota is tight; two small calls (understand,
  then classify) burn RPM/TPM without adding observable quality.
- **Options:** two separate calls / one structured-JSON call.
- **Chosen:** one call returning `{intent, question_type, refined_search_terms}`
  used to route (chatty vs research vs lookup) before retrieval.
- **Evidence:** DESIGN; verified against measured quota at WP9.
- **Disadvantages:** Slightly more brittle prompt; structured outputs must be
  validated defensively.
- **Revisit trigger:** If classification accuracy measurably degrades answers.

## D6 — Orchestration: hand-rolled asyncio state machine, not LangGraph

- **Problem:** Need staged pipeline with branch + loop + budgets, observable
  and explainable in an interview.
- **Options:** LangGraph / custom asyncio runner.
- **Chosen:** custom asyncio state machine with per-stage timing, hard budgets
  (≤6 model calls, ≤3 retrieval rounds, 45 s deadline).
- **Evidence:** Assignment permits LangGraph "only if it provides a clear
  benefit"; at this size it adds a dependency and hides control flow. Status:
  DESIGN.
- **Disadvantages:** We re-implement small orchestration guarantees ourselves.
- **Revisit trigger:** Multi-graph spending grows beyond this scope.

## D7 — Load tester: custom asyncio harness vs Locust

- **Problem:** Need TTFT measured from request receipt to first *content*
  token, identical measurement boundaries to the app, one row per offered
  request in `results.csv`.
- **Options:** Locust / custom asyncio harness.
- **Chosen:** custom asyncio harness (scripts/run_load_tests.py).
- **Evidence:** Locust's generic stats don't natively record the required
  12-column schema or TTFT-to-first-content-token semantics. Status: DESIGN.
- **Disadvantages:** We build scenario scheduling ourselves.
- **Revisit trigger:** Requirements stop fitting the harness.

## D8 — Daily-quota exhaustion is never retried

- **Problem:** Retrying a permanently exhausted daily quota wastes budget and
  time, and wastefully hammers the provider.
- **Options:** on-429 retry always / distinguish transient vs daily exhaustion.
- **Chosen:** A 429 whose documented/measured signature indicates daily
  exhaustion (RPD/TPD) is non-retryable; the budget tracker short-circuits
  locally once daily counters say we're exhausted. Per-minute 429s are
  retried with exponential backoff + jitter honoring `retry-after`.
- **Evidence:** DOCUMENTED today (console.groq.com/docs/rate-limits,
  2026-11-10): org-level RPM/RPD/TPM/TPD; `retry-after` present only with 429.
  Live error signatures to be MEASURED at WP9.
- **Disadvantages:** A misclassified daily-exhaust 429 would stop requests
  until the next window — mitigated by re-reading `x-ratelimit-reset-*`
  headers each response.
- **Revisit trigger:** Live measured signature contradicts the classifier.

## D9 — Cross-encoder scores re-order survivors only; no threshold on them

- **Problem:** Config C reranks the bi-encoder's top-k with a cross-encoder.
  What do we do with its score, and does `similarity_threshold` apply?
- **Options:** combine scores / re-apply threshold on cross-encoder score /
  re-sort survivors by cross-encoder score only.
- **Chosen:** re-sort the bi-encoder survivors (still ≥ `threshold` on
  cosine), discard bi-encoder scores from output, take top_k. The
  cross-encoder score is never compared to a cutoff.
- **Evidence:** MEASURED 2026-11-10 with the installed library: checkpoint
  `cross-encoder/ms-marco-MiniLM-L-6-v2` overrides the library's sigmoid
  default with `Identity()`, so `predict` returns raw logits (unbounded —
  e.g. 4.696 / −1.530 in the spot-check). There is no meaningful numeric
  scale to threshold; only ranking is meaningful. Scale ranges of the two
  scorers must never be mixed in one column without labeling.
- **Disadvantages:** A weak-but-above-threshold bi-encoder hit can outrank a
  marginally-thresholded strong hit — accepted; the threshold pre-filter's
  false-negatives remain possible (documented, an eval concern for WP11).
- **Revisit trigger:** If WP11 shows rerank losing important thresholded-out
  pages, consider reranking beyond top-k (e.g. top-8) or raising top_k.

## D10 — Context budget trim: greedy prefix, best-chunk-over-budget exception

- **Problem:** `max_context_tokens` can be smaller than one retrieved block.
  Silently dropping ALL evidence vs silently exceeding the budget?
- **Chosen:** Greedy best-first trim. If even the FIRST (highest-scorer)
  block alone exceeds the budget, include it anyway, log a warning, and
  surface `first_chunk_over_budget=True` in the result. Subsequent blocks
  are simply not added.
- **Evidence:** Design reasoning: an answer with one oversized grounded
  chunk is better than an empty refusal caused purely by a config
  mismatch; the flag keeps the behavior honest and observable (tests
  lock it in). Phase-1 default (3000 tokens budget vs ≤~1250-token
  blocks at 800-token config) makes the exception rare.
- **Disadvantages:** A pathological tiny budget can exceed the cap by up to
  one block (~+250 tokens worst case) — logged, measured in tests, and
  visible to the orchestrator.
- **Revisit trigger:** If the orchestrator needs a hard cap (e.g. for strict
  prompt sizing), it may truncate WITHIN the finest evidence text — out of
  scope for now because blocks are citation units (D4).

## D11 — Circuit breaker counts provider-health faults only

- **Problem:** Which failures open the breaker? Counting everything
  (incl. daily-quota 429s, auth, our own backpressure) would trip the
  breaker on business conditions instead of sickness.
- **Chosen:** OPENING counts ONLY `is_provider_fault` errors: transient
  network faults, 5xx, per-minute 429s (sustained saturation is a health
  signal). Daily quota / auth / malformed requests / capacity / budget
  guards are never breaker inputs. Success semantics: ONE clean success in
  CLOSED clears the window (consecutive-failure reading of the threshold).
- **Evidence:** DESIGN 2026-11-10, locked by unit tests
  (tests/unit/test_breaker.py: daily-quota and auth never trip it;
  per-minute saturation does). Half-open allows exactly one concurrent
  probe; a clean probe closes, a failed probe reopens with fresh cooldown.
- **Disadvantages:** Non-consecutive storm patterns (2 fails, 1 success,
  2 fails…) need 2×threshold events to trip — conservative; accepted.
- **Revisit trigger:** If live load tests show breaker flapping or
  non-tripping on real failure storms (WP10 evidence).

## D12 — Limiter is a header-observation projector, not an assumed-limits table

- **Problem:** Local rate-limit gating must not hardcode free-tier numbers
  that were never measured (violates honesty rules).
- **Chosen:** The limiter (a) stays in observe-only mode until the first
  x-ratelimit-* header set arrives, (b) projects remaining quota as
  last-observed remaining MINUS locally booked usage, (c) replaces the
  projection on every new observation (headers documented as always
  present), (d) blocks DAILY exhaustion (RPD) with priority — never
  guessing a reset time; unblocking requires a NEW header observation
  (re-read per call, per D8).
- **Evidence:** DESIGN 2026-11-10 on top of the DOCUMENTED always-present
  headers (decision log verified-facts, fetched 2026-11-10). Unit tests
  lock: observe-only admission, per-quota projection blocks
  (rpd > rpm > tpm priority), usage booking, observation replace.
- **Disadvantages:** Between observations a burst can overshoot; the
  provider's own 429 plus the retry executor are the documented backstop.
  Time-aware unblocking waits for MEASURED reset-header semantics (WP9).
- **Revisit trigger:** WP9 measurements contradict the header semantics.

---

## WP2 evidence — extraction tuning round (2026-11-10, MEASURED)

| Observation (from pymupdf font dump) | Effect | Rule that handles it |
|---|---|---|
| Moshi's true headings are bold and only **1.09× body size** | pure size gate loses them | numbered / "Appendix X." / Abstract-family patterns are the anchor |
| H-Net figure glyphs "n e w _ p r o d u c t _" (1.21×, non-bold, spaced) | noise | spaces-ratio ≤ 0.30 + non-bold requires ≥1.45× |
| moshi emoji labels 🤖/👤 (2.47×) and math symbols ℳ ℰ | noise | candidate text needs ≥2 ASCII letters |
| H-Net author block (bold/non-bold, 1.20×, page 1) | noise | page-1 gate: only ratio ≥1.40 flags on page 1 |
| Bold table values "41.29" / "0.743" / "1920 5.3 …" | noise | bare-number candidates need components ≤2 digits AND left-margin x0 ≤130; first word ≥3 digits rejects "1920…" rows |
| LaTeX split headings "1" + "Introduction"; "3.1" + "Scaled Dot-Product Attention" | numbers lost from labels | `_merge_number_split_headings` joins number-only lines with the next flagged heading, or a title-shaped line (≤60 chars, uppercase start, no trailing punctuation) |
| **Residual noise (documented limitation):** 3 Figure-1 legend labels on H-Net page 3 ("Dechunking", "Chunking", "Dynamic Chunking", bold 1.36×) pass as headings | section labels for a few chunks on one page may carry a figure label | accepted: page anchors remain exact; headings are heuristic by design; 3/37 H-Net headings = 8% of its labels, impact on retrieval expected negligible and judged in WP11 |

Printed page numbers vs PDF page index (scripts/verify_pagenums.py, MEASURED 2026-11-10):
attention 14/15 footer==index (title page prints none, 0 mismatches), moshi 66/66, hnet 39/39 →
citing "page N" is both the PDF page index and the printed page number in these three papers.

## WP3 evidence — retrieval build + spot-check (2026-11-10, MEASURED)

All numbers below are from this machine, this library build; not portable claims.

| Observation | Value | Note |
|---|---|---|
| all-MiniLM-L6-v2 reported embedding dim | 384 | matches DESIGN expectation |
| First `embedder_loaded` ms (incl. ~90 MB HF download) | 22,032 ms | one-time; subsequent loads from cache |
| Config A chunks (attention/moshi/hnet) | 40 / 200 / 147 = 387 | 300tok/50overlap |
| Config B chunks (attention/moshi/hnet) | 27 / 129 / 94 = **250** | 500tok/80overlap; **equals the 250 preview chunks from WP2** — independent consistency check ✔ |
| Config C chunks (attention/moshi/hnet) | 18 / 86 / 69 = 173 | 800tok/120overlap |
| Embed time per config (CPU) | A 10.7 s / B 6.2 s / C 4.4 s | ~58 ms/chunk incl. tokenization |
| Spot-check per-probe latency, config B (warm) | 12–27 ms | embed+search+trim only |
| Spot-check per-probe latency, config C (warm) | 287–425 ms | cross-encoder pass over ≤5 candidates dominates |
| Cross-encoder default activation for this checkpoint | `Identity()` → raw logits | printed scores e.g. 4.696, −1.530; ordering-only use (D9) |

Qualitative spot-check findings (≤3 shown of ≤5 hits per probe, configs A/B/C):

- Direct probes hit the right paper and pages in all configs (e.g. multi-head
  attention → attention p.13–15 visualizations; Mimi semantic/acoustic tokens →
  moshi p.9 §3.3 + p.24 §5.2; dynamic chunking → hnet p.2 §1).
- Config-weakness observation: with B (larger chunks), the "learning rate
  warmup" probe did NOT surface attention's own §5.2/5.3 optimizer schedule in
  its top-3; with A it moved to rank 1–3 (hnet p.12 §3 rank 2 in both), and with
  C rerank attention p.7 §5.2 reached rank 2. Judged qualitatively — the formal
  retrieval-quality comparison is WP11's job.
- Config C visibly re-orders: for the multi-head attention probe it lifts the
  §3.2/§3.2.2 definition pages (p.4, p.5) above the visualization figures.

Library-version facts discovered while writing WP3 code (DOCUMENTED from the
installed package, sentence-transformers v5-era layout):

- `SentenceTransformer.encode` has NO `is_query` parameter — unknown kwargs are
  forwarded unpredictably; MiniLM is symmetric, so we pass none (embedder
  docstring records this; asymmetric models would need `prompt_name`).
- `get_sentence_embedding_dimension` is deprecated in favor of
  `get_embedding_dimension`; embedder uses the new getter with fallback.
- `CrossEncoder.predict` accepts `(query, text)` pairs and our `numpy` +
  `show_progress_bar` usage matches this build's signature.

## WP4 evidence — reliability primitives (2026-11-10, MEASURED tests / DESIGN)

| Component | File | Behavior locked by tests |
|---|---|---|
| Error taxonomy (retryable vs not; D8 inside) | `app/reliability/errors.py` | classify-only unit tests implicit via consumers |
| Retry executor | `app/reliability/retry.py` | backoff bounds, retry-after honored (never shorter), per-call attempt cap, per-request allowance guard, daily/auth never retried, unexpected errors pass through |
| Circuit breaker | `app/reliability/breaker.py` | threshold/window with scripted clock, cooldown refusal, single-concurrent half-open probe, reopen on failed probe, consecutive-success clearing, business conditions never counted |
| Capacity gate | `app/reliability/capacity.py` | immediate admission vs queued vs honest `CapacityRejected`; counts exact; context-manager releases on error |
| Request budget | `app/reliability/budget.py` | model-call/round/retry claims + guard raises; deadline via scripted clock; introspection summary |
| Quota limiter | `app/reliability/limiter.py` | observe-only until headers; rpd>rpm>tpm priority projection; usage booking; new observation replaces/unblocks (D8, D12) |

Test counts: 46 new unit tests in WP4 (71 total suite, all passing, scripted
clocks everywhere — zero real sleeps in reliability tests).

Deliberately NOT measured yet (honesty): breaker window behavior under real
provider failure storms (WP10), header parsing (WP5 adapter), live quota
numbers (WP9 — none hardcoded anywhere).

The WP4 request recorder is now implemented against the EXACT 12-column
contract supplied by the user at the WP4 gate (2026-11-11):

## D13 — results.csv contract locked; recorder enforces it structurally

- **Contract (user-supplied, exact):** columns
  `request_id, scenario, source, start_time, first_token_ms, total_ms,
  status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`
  in this order; no additions/renames ever. One row per offered request
  without exception (completed, queue-rejected, quota-blocked,
  breaker-refused, rate-limited, timed-out, failed, cancelled-when-
  detectable). Statuses exactly `ok|rate_limited|timeout|error`; grades
  exactly `correct|wrong|refused_correctly|refused_wrongly` or EMPTY when
  no evaluatable answer (infrastructure failures are never auto-graded as
  correct refusals). `start_time` = the moment the request was OFFERED,
  ISO 8601 UTC with milliseconds and Z (`2026-11-11T12:34:56.789Z`).
  `first_token_ms` measured from the ORIGINAL offer time, empty when no
  token (zero never substitutes). `tokens_in/out` aggregate across ALL
  model calls and retries; only provider-REPORTED usage is aggregated,
  estimates tracked separately, nothing invented. `retries` = additional
  provider attempts beyond the first, summed over the pipeline. `source`
  exactly `real|stand-in`. `retrieval_setting` = stable config id
  (`A|B|C`, `not_applicable` when retrieval never ran) with a separate
  metadata file documenting chunk/overlap/top-k/rerank per id.
- **Chosen implementation:** `app/reliability/recorder.py` — pydantic
  `RequestRecord` with Literal-locked status/source/grade, extra-columns
  forbidden, timestamp-format validator; `RequestRecorder` with
  duplicate-id hard error (including pre-loaded ids from a restarted
  file, so one-row-per-offered-request holds across restarts),
  write-time header validation (a file that appeared since construction
  is validated, never truncated), `\n` line endings for stable diffs.
  `map_outcome()` is the SINGLE mapping table typed outcome ->
  (status, outcome_kind); the original typed identity is preserved in
  the structured `row_recorded` log event, never lost to the 4-value
  status. `TokenUsageAccumulator` separates reported vs estimated usage.
- **Evidence:** 42 recorder unit tests lock every validation point from
  the user's checklist (schema, one-row-per-outcome, empty TTFT, grade
  rules, timestamp format, aggregation, duplicate ids, statuses,
  metadata file). Full suite 113/113 passing. A demonstration file was
  generated in the OS temp dir to show the format for all terminal
  outcomes — it is a FORMAT DEMO, not benchmark data, and is not part of
  results/.
- **Remaining limitations (honest):** per-request event files (queue
  wait, stage durations, provider headers, token accounting detail) are
  emitted by the orchestrator/load harness wiring at WP7/WP10 — the
  recorder logs `row_recorded` events now but cannot force other
  components to log; measured free-tier limits remain unknown until
  WP9; token reporting by the stand-in will follow its documented
  tokenizer (implemented at WP6).

## D14 — One shared OpenAI-compatible transport for Groq AND the stand-in

- **Problem:** Two providers (real Groq; local stand-in) must behave
  identically on the wire so load experiments exercise the same code
  path; duplicated clients would drift silently.
- **Chosen:** `app/providers/compat.py` implements the transport ONCE
  (request building, SSE parsing, status→typed-error mapping, rate-limit
  header capture, transport timing); `GroqProvider` adds only
  credentials, `StandInProvider` only base-URL/absent-auth. Providers are
  PURE TRANSPORT: no retry/breaker/limiter inside — WP7 composes
  reliability around them, so a transport can never retry on its own.
  The API key exists only inside the Authorization header — never logged,
  never returned, never stored in settings.
- **Evidence:** DESIGN 2026-11-10/11; unit tests use `httpx.MockTransport`
  — ZERO live network traffic in WP5 (the Groq budget rule holds).
  429 classification: remaining-REQUESTS counter == 0 → daily (D8), else
  per-minute with retry-after — DOCUMENTED-header based, to be confirmed
  against MEASURED signatures at WP9. `stream_options.include_usage` sent
  on streams; absent usage degrades to None, never invented.
- **Disadvantages:** Raw HTTP means we own SSE parsing quirks (kept in
  one file; the vendor SDK would hide headers we need).
- **Revisit trigger:** Live behavior at WP9/WP10 diverging from the
  documented contract.

## WP5 evidence — provider transport (2026-11-11, tests MEASURED offline)

| Component | File | Locked behavior |
|---|---|---|
| Shared transport | `app/providers/compat.py` | non-stream + SSE parse; per-call `transport_ms`; headers→`ProviderLimitsSnapshot`; typed errors incl. 429 daily/minute split |
| Groq | `app/providers/groq.py` | key from env/.env only (ConfigurationError when absent, raised BEFORE any traffic); Bearer header per request |
| Stand-in | `app/providers/standin.py` | settings default URL `http://127.0.0.1:8100`; no auth; usage=None when the service reports none; docstring holds the WP6 service contract (OpenAI-compatible + documented tokenizer on usage) |
| Error taxonomy | `app/reliability/errors.py` | `NonRetryableProviderError` now carries `status_code`/`retry_after_s` as real attributes; `UserCancelled`/`ConfigurationError` added |

Tests: 13 new provider tests (suite 126/126, all passing; zero network).
Test gotcha recorded: an injected `httpx.AsyncClient` must include
`base_url`, or relative URLs break cookie extraction
("unknown url type: '/chat/completions'").

Not measured (honesty): live Groq behavior — no request sent without the
WP9 sign-off; the stand-in service itself is WP6's deliverable.

## D15 — Stand-in service: documented tokenizer + scripted failure modes

- **Problem:** Load experiments need a controllable, deterministic LLM
  endpoint whose token accounting is trustworthy per D13 (only
  provider-REPORTED usage aggregated), and whose failure modes can be
  scripted without touching real Groq.
- **Chosen:** `app/standin/` — (1) `tokenizer.py` is the DOCUMENTED
  tokenizer: a token is one match of `\\w+|[^\\w\\s]`; whitespace never
  contributes; per-message framing overhead = 4 tokens (an approximation,
  documented as such); prompt_tokens = Σ(role+content+4 per message);
  completion_tokens = count of the generated text exactly; same input →
  same numbers, no hidden counting state. (2) `service.py` is a FastAPI
  OpenAI-compatible simulator: non-stream + SSE, `max_tokens` honored as
  an exact target (finish_reason `length`), usage ALWAYS reported on
  streams when include_usage is requested (a service guarantee — stronger
  than Groq's, whose honoring is unconfirmed until WP9), no auth, NO
  rate-limit headers in ok mode (the limiter's honest observe-only path).
- **Scripted failure modes** (admin API: `/_admin/mode` with optional
  `times=N` auto-revert; `/_admin/shaping` sticky delay/TTFT/length;
  `/_admin/state` counters; `/_admin/reset`): ok → 200; rate_limited_daily
  → 429 remaining-requests=0 (RateLimitDailyError, D8 never-retry);
  rate_limited_minute → 429 remaining>0 (RateLimitMinuteError);
  server_error → configurable 5xx (ProviderServerError); malformed →
  200 non-JSON (MalformedRequestError); hang → real client timeout
  (TransientNetworkError); abrupt_close → truncated body / SSE without
  [DONE] (silent truncation the orchestrator must detect at WP7).
- **Honesty rule:** everything measured against this service describes
  the SIMULATOR and is never presented as Groq behavior; results.csv
  rows carry `source=stand-in` for exactly this reason.
- **Evidence (2026-11-11, MEASURED):** 38 new tests — 9 tokenizer, 23
  in-process ASGI service tests, 2 raw-socket wire tests (REAL hang →
  ReadTimeout → TransientNetworkError; REAL incomplete-chunked mid-stream
  close → RemoteProtocolError → TransientNetworkError), 4 end-to-end
  tests booting the real uvicorn server on an ephemeral port and driving
  it with the actual StandInProvider (complete, stream, usage match,
  scripted daily 429 round trip). Suite 164/164.
- **Disadvantages:** latency shaping is fixed sleeps, not distributions;
  admin-state concurrency is approximate under interleaved requests
  (single event loop, documented); the simulator's generation is
  word-cycle text, not a language model.
- **Revisit trigger:** WP10 scenarios needing failure shapes the modes
  cannot express.

## WP6 evidence — stand-in service (2026-11-11, MEASURED tests)

| Scripted mode | HTTP behavior | Client-side typed outcome (tested) |
|---|---|---|
| ok | 200 + usage (tokenizer) | ChatResponse / stream chunks |
| rate_limited_daily | 429, remaining-requests=0, retry-after | RateLimitDailyError (never retried) |
| rate_limited_minute | 429, remaining-requests>0, retry-after | RateLimitMinuteError |
| server_error | 503 (default) / configurable 5xx | ProviderServerError |
| malformed | 200 non-JSON body | MalformedRequestError |
| hang | accepts, stalls beyond client timeout | TransientNetworkError (real socket test) |
| abrupt_close | truncated JSON / SSE ends without [DONE] | MalformedRequestError (non-stream); silent truncation detected by orchestrator at WP7 |

## D16 — RelayRunner composition: citations, TTFT semantics, retries, honesty rules

- **Citation contract:** a valid citation tag is a bracketed span
  containing `arXiv:<versioned_id>` and `p.<page>` (the exact provenance
  the WP3 evidence blocks carry, so the model copies real tags). A
  non-refusal answer with ZERO citations is invalid (uncited); any cited
  (id, page) pair absent from the DELIVERED evidence is invalid
  (page-pinned chunks, D4, make the check exact). Refusals are not
  validated (no citations expected); WP11 grades refused_* answers.
- **Understanding robustness (D5):** unparseable understanding output
  NEVER fabricates a refusal — with evidence present the pipeline
  proceeds to the answer and the citation validator backstops. An
  explicitly insufficient verdict gets ONE refinement round: re-retrieve
  with `refined_search_terms`, understand again; still insufficient or no
  refined terms -> honest refusal. Empty retrieval -> refusal WITHOUT any
  provider call (quota discipline).
- **Regeneration:** exactly one re-answer when the first answer's
  citations are invalid. Regen valid -> deliver it (TTFT from the
  delivered attempt). Regen still invalid -> deliver the FIRST answer,
  flagged `citations_valid=False` (availability over silent perfection).
  Regen fails or hits the deadline -> the request ends as that typed
  failure; nothing user-visible was delivered, so the row's
  `first_token_ms` stays empty (the event file carries both attempts).
- **first_token_ms rule (D13 interpretation):** the first generated
  content token of the answer ACTUALLY DELIVERED, measured from the offer
  moment (queue wait included). Refusal rows: empty (the template text is
  not generated content). Failed/timeout rows: empty even when a partial
  stream emitted tokens — partial content is discarded with an explicit
  `partial_stream_discarded` event, never shown to the user, never
  zero-substituted into the CSV.
- **retries (results.csv)** = provider HTTP attempts − 1, summed over the
  whole pipeline (retry-executor attempts + regen + refinement-round
  understanding); the event file carries the per-stage breakdown. Attempts
  refused by the limiter or breaker never count (no traffic was sent) and
  never book usage.
- **Composition per attempt:** budget claim (per LOGICAL call) ->
  [retry executor: limiter re-project -> breaker -> attempt]. Watchdog
  expiries (per-call timeout, first-token watchdog) map to
  `TransientNetworkError` — the same family as httpx timeouts: retryable,
  breaker-counted provider-health signals. The REQUEST deadline expiry is
  `DeadlineExceeded` (status timeout). User cancellation -> `user_cancelled`
  row, then CancelledError re-raised. Empty observed header snapshots
  NEVER wipe a real observation (mixed-provider safety). Limiter booking
  uses reported tokens when present, else a char/4 estimate + 256-token
  output allowance; booking affects admission only, every booking
  event-logged with its basis (D13 honesty: CSV uses reported only).
- **Runner contract:** run() NEVER raises for expected outcomes — it
  always returns a RelayOutcome, records exactly ONE results.csv row and
  one per-request event file (even for programmer bugs: `unexpected_*`
  kind). Provider-source labels are validated at construction (an
  unlabele-able provider fails the build, never a request).
- **Found-and-fixed during composition:** (1) WP4 capacity gate leaked the
  `_waiting` counter when a queued request was cancelled — fixed
  (cancellation now decrements; queue-wait evidence stays exact). (2) WP6
  stand-in stream chunking corrupted punctuation adjacency ("p.4" streamed
  as "p . 4"), breaking citation fidelity — stream pieces now concatenate
  back to the EXACT original text (fidelity test added).

## WP7 evidence — RelayRunner (2026-11-11, MEASURED tests)

| Component | File | Locked behavior |
|---|---|---|
| State machine | `app/orchestrator/runner.py` | admit -> retrieve -> D5 understanding -> refined round-2 -> streamed answer -> citation check -> one regen -> honest row + event file; outcome-not-raise |
| Event logs (D13) | `app/orchestrator/events.py` | per-request JSONL: queue wait/rejection, stage durations, provider attempts/calls, retry events, first-token/completion timestamps, token bookings (reported vs estimated), typed outcome |
| Prompts + parser | `app/orchestrator/prompts.py` | D5 merged understanding JSON; defensive parse (None, never invented) |
| Citations | `app/orchestrator/citations.py` | bracket arXiv:+p. contract; membership vs delivered evidence; uncited = invalid |
| Capacity fix | `app/reliability/capacity.py` | queue-wait cancellation decrements `_waiting` (leak fixed) |
| Stream fidelity | `app/standin/service.py` | streamed deltas reassemble to the EXACT original text |

Tests: 23 new (20 runner unit + 2 runner/stand-in e2e + 1 stream-fidelity);
suite 187/187, zero Groq traffic. The e2e tests prove the full honesty
chain: service tokenizer -> provider-reported usage -> runner aggregation
-> results.csv exact values; and the scripted daily-429 -> typed row with
retries=0.

## D17 - API layer: HTTP outcome mapping, validation-before-offer, SSE semantics

The FastAPI layer (`app/api/service.py`, booted by `scripts/run_api.py`)
wraps ONE RelayRunner. It NEVER changes the runner's honesty semantics
(D13/D16): one offered request = one row + one event file, whatever the
HTTP surface does.

- **Outcome -> HTTP mapping** (`/ask`): ok (incl. honest refusal) -> 200;
  rate_limited -> 429 (+ `retry_after_s` when the provider sent one);
  timeout -> 504; error with `outcome_kind=capacity_rejected` -> 503 (OUR
  backpressure, not provider sickness); any other error -> 502; unexpected
  runner re-raise -> 500 (row already recorded as `unexpected_*`). The CSV
  row is UNTOUCHED by this mapping: it keeps status
  ok/rate_limited/timeout/error and the typed `outcome_kind` lives in the
  event file, exactly as D13 requires — only the HTTP surface
  distinguishes 503-vs-502.
- **Validation happens BEFORE offering** (no row, no provider traffic):
  invalid JSON body / non-object body / missing-empty-whitespace question /
  unknown `retrieval_setting` / bad `request_id` pattern -> 400 with
  `{error:{kind,message}}`; a `request_id` that already has a row -> 409
  (`recorder.has_request()` pre-check, added in WP8; `record()` remains
  the hard backstop either way). The runner keeps its own honest-row
  safety net for direct callers (the load harness never validates first,
  by design).
- **`start_time` is captured at HTTP entry**, not at runner entry, so the
  measured TTFT (offered_perf -> first token) covers queue wait
  end-to-end and matches the load harness's client-side boundary (D7).
- **SSE `/ask/stream` grammar**: named events `status` (admitted /
  retrieved / understanding / answer_started / retry), `token` (one
  content delta), `reset` (regen or partial-discard — clear your buffer),
  and a terminal `done` that ALWAYS carries the full final outcome incl.
  the complete answer text, so a client that dropped or reset tokens can
  reconcile. `: keepalive` comments on idle. Live transport stays 200
  even when the answer fails — the terminal status rides in `done` (SSE
  convention, matches how real OpenAI-style streams behave).
- **Backpressure**: the sink pushes into a bounded asyncio.Queue
  (maxsize 1024, `put_nowait`); on overflow, LIVE token events are
  dropped and counted (`tokens_dropped` in `done`); status/reset events
  queue normally. A slow consumer can never couple back into the runner.
- **Runner `token_sink` extension (WP8)**: per-run kwarg
  `run(token_sink=...)`, stored on `_RunState` — never per-runner, so
  concurrent runs stay independent. `_notify()` is failure-isolated (a
  throwing consumer is event-logged and skipped; the row is unaffected);
  cancellation always propagates. Status events: admitted, retrieved,
  understanding, answer_started, retry; reset on regen and on
  partial-stream-discard (skipped when the run is already being
  cancelled — a cancelled task must not await).
- **Client disconnect mid-stream**: the SSE generator's `finally` cancels
  the runner task -> the runner records its honest `user_cancelled` row
  (D13: one row per offered request, even when the consumer vanished)
  and re-raises. Cleanup failures are event-logged
  (`stream_cleanup_error`), never silently swallowed.
- **MEASURED Windows gotcha (this machine, verified by isolated repro
  `repro_sse_disconnect.py`):** uvicorn 0.54 h11 on the proactor loop
  sends ASGI `spec_version` "2.3", so Starlette uses the collapsing task
  group + `listen_for_disconnect` path — but `connection_lost` NEVER
  fires mid-stream when the client closes (reads are paused under flow
  control; the proactor EOF is never noticed). Detection therefore runs
  through a per-iteration NON-BLOCKING `request.is_disconnected()` poll
  in the pump loop (pre-cancelled-scope receive; it also RESUMES the
  paused reads, which is what actually drives EOF detection), bounded by
  the 1 s keepalive cadence. httpx `ASGITransport.receive()` blocks until
  response completion, so this poll is inert in the in-process unit
  tests.
- **Retry marker**: `on_retry` is a SYNC callback inside the retry
  executor (WP4), so the live-stream retry status is delivered via
  `loop.create_task(_notify(...))` — scheduled, never blocking the
  backoff path.
- **Stand-in shaping extension (D15 +)**: `stream_chunk_delay_ms` spaces
  consecutive stream chunks apart (delivery pacing only; text and
  reported usage unchanged). Added for deterministic disconnect-race
  testing; also useful for WP10 stream-duration shaping.
- **MEASURED cold-start finding (2026-11-11, this machine)**: the first
  `/ask` after boot measured first_token_ms=27626.9 / total 27630.3 —
  dominated by the ~25 s lazy MiniLM/index load, NOT the pipeline.
  `scripts/run_api.py` now runs a boot `_warmup()` probe (one retrieve
  per bound setting) BEFORE uvicorn starts listening; re-measured warm:
  first_token_ms=113.2, total_ms=117.1 (tokens_in=2047, tokens_out=62,
  retries=1 = understanding+answer attempts-1, citations valid, source
  stand-in, scenario wp8-smoke). Smoke artifacts deleted after
  verification; `results/retrieval_settings.json` retained (D13
  config-metadata file).

## WP8 evidence — API layer (2026-11-11, MEASURED tests)

| Component | File | Locked behavior |
|---|---|---|
| HTTP mapping | `app/api/service.py` | D17 outcome->status map incl. 503-vs-502 capacity split; row semantics unchanged |
| Validation | `app/api/service.py` | 400/409 BEFORE offering; zero rows on validation failure (asserted) |
| SSE pump | `app/api/service.py` | bounded queue 1024 + drop counting; keepalive; `done` reconciles; disconnect -> cancel -> honest row |
| Runner sink | `app/orchestrator/runner.py` | per-run token_sink; failure-isolated `_notify`; retry marker via create_task |
| Recorder pre-check | `app/reliability/recorder.py` | `has_request()` for the 409 path; `record()` still the hard backstop |
| Stand-in pacing | `app/standin/service.py` | `stream_chunk_delay_ms` (D15 +): delivery pacing only, text/usage unchanged |
| Boot warmup | `scripts/run_api.py` | real A/B/C retrievers, stand-in default, groq gated on explicit `--model`, warmup probe |

Tests: 17 new (15 API unit incl. full HTTP-mapping matrix + SSE grammar +
regen reset + validation-no-row + 409 + 503 + 429s; 1 stand-in pacing; 1
real-uvicorn disconnect integration proving user_cancelled row). Suite
204/204, zero Groq traffic. Real smoke run measured: /health 200 with
breaker/limiter/gate introspection; full /ask chain over real HTTP with
real A/B/C indexes (warm numbers above).

## D18 - WP9 calibration: MEASURED real-Groq behavior for this account

Calibration ran 2026-11-12 with user sign-off (Steps A+B+C approved;
optional Step D NOT run). Instrument: `scripts/calibrate_groq.py`; every
measured fact appended to `results/logs/groq_calibration.jsonl`; the 3
runner offers produced real rows in `results/results.csv` (source=real).
Key placement needed a masked-input helper (`scripts/set_groq_key.py`)
after manual file creation failed repeatedly; the key value never
entered chat or logs (only lengths were ever printed).

- **Model choice (measured, Step A)**: `GET /models` listed 11 models for
  this account — `llama-3.1-8b-instant` (the original free-tier default
  suggestion) is NOT available to it. Chat-capable general models:
  `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`,
  `allam-2-7b`; prompt-guard (moderation) and whisper (ASR) models are
  not chat models. Chosen: **openai/gpt-oss-20b** (smallest general chat
  model; English-capable; fastest). 1 request, 0 tokens.
- **Wire probe (measured, Step B, 2 requests / 220 tokens)**:
  non-stream transport 953.6 ms (wall 1084.8 ms); stream TTFT 445.6 ms,
  total 448.8 ms; both finish_reason=stop; tiny-prompt usage 78 in /
  32 out. gpt-oss reasoning is PARSED: hidden reasoning is EXCLUDED from
  `content` but INCLUDED in reported completion tokens (32 completion
  tokens for a 1-word answer). Wire + usage + TTFT all consistent with
  the D14 transport's expectations.
- **Live per-model limits for gpt-oss-20b (from x-ratelimit-* headers,
  measured)**: RPD limit 1000 / remaining 999; **TPM limit 8000** /
  remaining 7858; **no RPM limit header (null)** for this model — Groq
  enforces only RPD+TPM here. (Previously DOCUMENTED-only; now measured.)
- **Full-runner calibration (measured, Step C, 3 offers / ~5K tokens)**:
  - wp9-calib-1: FULL SUCCESS end-to-end against real Groq (real
    retrieval B -> understanding -> streamed answer): TTFT 1489.4 ms,
    total 1759.5 ms, tokens 2352 in / 633 out, citations_valid=true,
    retries=1 (= attempts-1: understanding+answer, no error retries).
  - wp9-calib-2: `rate_limited` / `quota_block_local` — the limiter's
    LOCAL projection blocked the ANSWER call after the understanding
    call ran (projected 1317 remaining < needed 1919). Row honestly
    carries the partial work (tokens 1787/197, empty first_token_ms).
  - wp9-calib-3: `quota_block_local` before ANY provider call (21.2 ms,
    empty tokens) — once blocked locally, no new provider response
    arrives, so the projection cannot refresh until traffic resumes
    (designed conservatism, D12).
- **The systemic, measured finding**: TPM 8000 with ~2-3K-token RAG
  calls sustains only ~2-3 calls/minute; back-to-back offers hit local
  quota blocks BY DESIGN (the limiter refuses calls it projects would
  be 429'd — protecting the account instead of burning retries).
  Consequence for WP10/WP11 real probes: pace requests (~30 s apart) or
  accept honest `quota_block_local` rows.
- **D8 revisit status**: ZERO real 429s occurred (none forced, none
  natural) — the minute-vs-daily 429 classification remains
  DOCUMENTED-header-based; the revisit rule stays open until a natural
  429 is measured.
- **Budget accounting (measured)**: 6 offers total (1 models + 2 wire +
  3 runner); provider-reported tokens consumed: 5,189 of the approved
  ~50K; 6 of the approved <=100 requests. Step D (5 more offers) remains
  unspent and unapproved.
- **Cosmetic observation (honest, undiagnosed)**: the calibration script
  exited 1 AFTER completing all work (all output printed, artifacts
  verified complete); import-only runs exit 0, so this is teardown-time
  noise under torch/httpx on Windows, not a logic fault. Its exit code
  must not be used as a success signal — artifacts are the evidence.

## WP9 evidence — real calibration (2026-11-12, MEASURED)

| Fact | Value | Basis |
|---|---|---|
| Available chat models (this account) | 11 total; chat-capable: gpt-oss-120b/20b, qwen3.8-27b, allam-2-7b | MEASURED, GET /models |
| Calibration model | openai/gpt-oss-20b | measured choice (smallest general chat model) |
| gpt-oss-20b limits | RPD 1000, TPM 8000, RPM none | MEASURED, x-ratelimit-* headers |
| Tiny-call latency | non-stream ~954 ms; stream TTFT ~446 ms | MEASURED, 2 wire probes |
| Reasoning-token behavior | content excludes reasoning; usage includes it | MEASURED |
| Full RAG round-trip (real) | TTFT 1489.4 / total 1759.5 ms, 2352 in / 633 out, citations valid | MEASURED, wp9-calib-1 |
| Local quota projection under back-to-back load | blocks after ~5K tokens/min consumed; rows stay honest | MEASURED, wp9-calib-2/3 |
| Real 429 signatures | none observed (0 forced, 0 natural) | MEASURED absence |
| Budget spent | 6 offers / 5,189 tokens of approved 100 / ~50K | MEASURED |

## D19 - WP10 load experiments: stand-in scenario set + the load-bug it caught

Instrument: `scripts/run_load_tests.py` (the file D7 named). It boots its
own uvicorn servers through `build_relay_app` (extracted from
`scripts/run_api.py` so harness and server share ONE boot path), applies
stand-in admin modes per scenario, offers requests over real HTTP, and
verifies the D13 invariant per run (rows == offers, else hard failure).
Scenario set user-confirmed 2026-11-12. ALL results are stand-in
(source=stand-in); zero real traffic in WP10 so far.

- **baseline (MEASURED, 60/60 rows, 4 concurrent)**: TTFT median 855.5 /
  p95 905.0 / max 928.2 ms (0.8 s scripted first-token shaping +
  pipeline); total median 1698.6 ms; client wall median 1719.9 ms;
  retries_sum 120 (= 60 x [understanding+answer attempts-1], no error
  retries); wall-clock 26.5 s.
- **burst (MEASURED, 60/60 rows, 25 concurrent vs in-flight 8)**: ALL ok;
  TTFT median 3003.6 / p95 3617.0 / max 4093.6 ms — the ~+2.1 s median
  TTFT vs baseline IS queue wait, measured through the D17 boundary
  (start_time at HTTP entry). Wall-clock 10.0 s for 60 offers. Zero queue
  overflows (queue_limit 200 > 60 offers): 503 capacity rejection is
  unit-tested, not load-measured at this scale — honest boundary.
- **minute429 (MEASURED, 20/20 rows)**: 1x rate_limited after its
  retries; 19x error/breaker_open (fast local refusals); retries_sum 3.
  DESIGN CONSEQUENCE MEASURED: a sustained minute-429 storm trips the
  breaker (RateLimitMinuteError is a RetryableProviderError, so its
  failures count toward the breaker threshold) — the caller then gets
  fast breaker_open refusals instead of retry storms. Whether quota
  429s SHOULD count toward a provider-HEALTH breaker is surfaced as a
  design question for the user (kept as-is until they answer).
- **daily429 (MEASURED, 12/12 rows)**: 12x rate_limited, retries_sum 0 —
  D8's never-retry rule measured at load scale; every row landed in
  38-134 ms; wall-clock 0.8 s (no hammering of a dead quota).
- **breaker (MEASURED, 35/35 rows, AFTER the fix below)**: phase A under
  scripted 503s: 25x breaker_open fast refusals; phase B (mode ok after
  the 20 s open window): 10/10 completed, recovery TTFT median 349.6 ms
  (0.3 s shaping + pipeline); retries_sum 21; zero unexpected outcomes.
- **BUG FOUND AND FIXED under load (the point of WP10)**: the first
  successful half-open probe crashed with `unexpected_KeyError` —
  `breaker.py` logged `breaker_closed` with `extra={"name": ...}`;
  `name` is a reserved LogRecord attribute so `makeRecord` raises
  "Attempt to overwrite 'name' in LogRecord", which propagated into the
  runner and poisoned an honest row. WHY TESTS MISSED IT: pytest runs
  with WARNING-level logging by default, and a disabled `info()` call
  never reaches `makeRecord` — the crash needed INFO-level logging
  (production config). Fixed to `extra={"breaker": ...}` plus a
  regression test that forces INFO level; the re-run measured a clean
  recovery (10/10 completed, 0 unexpected).
- **Operator gotchas recorded honestly**: (1) PowerShell's
  `Select-Object -First N` early-terminates the pipeline and KILLS the
  native python process mid-run (a burst invocation died at 2 offers
  this way; re-run clean). (2) The killed burst left 2 recorded rows and
  a few in-flight offers whose rows were never written because the
  whole process died — hard process death remains the one D13 edge a
  recorder cannot close (it cannot record an outcome that never
  happened); the complete clean run is the load evidence. (3) The
  harness (and calibration script) exits 1 after completing all work —
  teardown noise under torch/uvicorn on Windows; artifacts are the
  success signal, not the exit code.

## D20 - Quota 429s are NOT breaker inputs (user-approved revision of the
## minute-429 classification)

WP10's measured minute429 storm (D19: 1 rate_limited + 19 breaker_open)
surfaced that `is_provider_fault` counted `RateLimitMinuteError` toward
the breaker threshold (WP4 rationale: "sustained saturation is a breaker
concern"). The user chose the exclusion semantics (2026-11-12): **the
breaker is a PROVIDER-HEALTH signal only** — quota pressure belongs to
the limiter (projection/admission) and the retry layer (backoff), which
already own it end-to-end.

- **Change**: `is_provider_fault` now counts only
  `TransientNetworkError | ProviderServerError`. Both quota 429s
  (`RateLimitMinuteError`, `RateLimitDailyError`) are excluded — daily
  was already excluded (non-retryable family), minute is removed by this
  decision. Retry semantics are UNCHANGED (they check the
  `RetryableProviderError` family, not the breaker input): minute-429s
  still retry with backoff honoring retry-after; daily still never
  retries (D8).
- **Re-measured minute429 under D20 (MEASURED, 2026-11-12, 20/20 rows)**:
  20x rate_limited (`provider_rate_limit_minute`, HTTP 429), ZERO
  breaker_open, retries_sum 40 (= 20 offers x 2 retries, all attempts
  exhausted honestly against retry-after 0.5 s), per-offer total median
  1882.3 ms (2 x ~0.5 s backoff + overhead), wall-clock 19.2 s. The
  measured trade: a quota storm now costs retry TIME, never a
  health-trip. daily429 was not re-run — its code path is untouched by
  D20 (daily was never a breaker input; D19's 12x rate_limited /
  retries_sum 0 remains valid).
- **Tests**: `test_minute_rate_limit_counts_as_provider_fault` replaced
  by `test_minute_rate_limit_is_excluded_from_breaker_counting` (4x
  minute-429 never trips, failure window stays 0, and a following real
  5xx still counts normally). Suite 205/205 after reconciliation.

## D21 - Limiter projection ratchet: double-count fix + minute-window aging
## (measured on the WP10 real probes; user-approved "fix only, no re-run")

The 10 real Groq probes (2026-11-12, user-approved) measured: 2 ok rows
(1 completed answer TTFT 1682.7 ms / 2352 in / 738 out, citations valid;
1 HONEST REFUSAL - the understanding stage declined with 4792 in / 457
out across two D5 rounds, no answer call) and 8 quota_block_local rows
blocked locally in 10-26 ms with ZERO provider traffic - despite 30 s
pacing. The header math in the event files proves the root causes:

- **Double-count (primary)**: the runner observed a response's headers
  (observe resets the since-accumulator) and THEN booked that same call's
  reported usage. The observation already includes the call that produced
  it, so projected = observed_remaining - just_booked_call. Measured
  exactly: observe tpm_remaining=2924 then book 2737 -> projected 187 <
  needed ~2024 -> blocked. Every later offer was blocked with the same
  stale projection (no traffic -> no new observation -> no recovery) - a
  self-sustaining ratchet over 4.5 minutes while Groq's server-side
  window demonstrably reset (offer-001 started a fresh ~8000-token
  window ~60 s after offer-000).
- **No aging (secondary)**: bookings carried no timestamps, so minute-
  window usage never expired from the projection even when the
  provider's window had rolled.

**Fix (user chose "fix only, no re-run")**:
1. Runner order: `_book_usage` BEFORE `_observe_limits` in BOTH attempt
   paths (`_complete_attempt`, `_stream_attempt`) - booking covers the
   header-less case; a following observation supersedes it (since=0).
2. Limiter (D21): bookings are timestamped (`_Booking.at`, injectable
   clock); the minute projections (RPM/TPM) count only bookings newer
   than `minute_window_s` (default 60 s); RPD bookings NEVER age (a day
   does not reset itself; D8 unchanged).

**Verification (MEASURED)**: unit test reproduces the exact measured
scenario (2924/2737 -> QuotaBlock projected=187; +61 s -> admitted,
all-time total keeps 2737 for RPD, window total 0); runner-level
regression proves the understanding booking is superseded by its own
observation (only the stream booking remains on a header-less fake);
stand-in baseline re-run 60/60 completed, TTFT median 864.4 ms (vs 855.5
pre-fix - same profile); suite 209/209.

**Honest boundaries**: the fix is NOT re-measured against real Groq (the
user declined a re-run; ~13,528 real tokens spent of the ~50K envelope
stays final). The WP10 real-probe rows stand as measured history: 2 ok +
8 honest local blocks - the blocks were CORRECT rows for the buggy
projection, and the buggy projection is what the fix removes. The
offer-001 refusal is carried to WP11 as evaluation input (the B-index
retrieval for the weight-initialization question did not yield
citable evidence).

## WP10 evidence — load experiments (2026-11-12, MEASURED, stand-in)

| Scenario | Offers | Rows | Status mix | Key numbers |
|---|---|---|---|---|
| baseline | 60 | 60 | 60 ok | TTFT med 855.5 / p95 905 ms; total med 1698.6 ms |
| burst | 60 | 60 | 60 ok | TTFT med 3003.6 ms (= queue wait); wall 10.0 s |
| minute429 | 20 | 20 | 20 rate_limited (re-measured under D20: 0 breaker_open, retries_sum 40) | D19 records the pre-D20 storm (1 + 19 breaker_open) as history |
| daily429 | 12 | 12 | 12 rate_limited | retries_sum 0 (D8); rows in 38-134 ms |
| breaker | 35 | 35 | 25 breaker_open + 10 ok | recovery 10/10 after fix; 0 unexpected |
| real_probe (GROQ, D21) | 10 | 10 | 2 ok + 8 rate_limited (quota_block_local) | answer TTFT 1682.7 ms; 1 honest refusal; 8 local blocks = measured projection ratchet (fixed, not re-run) |

Artifacts: results/logs/load_<scenario>.jsonl (per request),
results/logs/wp10_summary.json, results/plots/wp10_<scenario>.png,
results.csv rows wp10-*. Bugs fixed + regression tests; suite 209/209.

## D22 - WP11 evaluation: 16 fact-checked questions, deterministic grading,
## stand-in-only scope (real-Groq run DECLINED by user decision)

Instrument: `scripts/run_eval.py` (boots the REAL stack via
`build_relay_app`, stand-in provider). Question set: 5 per pinned paper +
1 cross-paper = 16, ALL answerable and verified against the ACTUAL
config_b chunk texts (expected facts located in chunk text by a scripted
scan; the ground-truth set per question = ALL chunks containing every
expected fact, any-of delivery counts as a hit). Grading is deterministic
string logic, NO LLM judge: correct = not refused AND citations valid AND
every expected fact in the answer AND ground-truth chunk delivered.

HONEST SCOPE: on the stand-in the answer content is CANNED by
construction - the facts check exercises stream fidelity + grading
mechanics, NOT reasoning. Genuinely measured: retrieval quality per
setting, citation pipeline, refusal behavior, row honesty (D13 invariant
held: 16 rows = 16 offers per run, 48 rows total).

**MEASURED (2026-11-12, setting x 16 questions, zero real traffic):**

Final re-measured numbers (2026-11-12, after the wording fix documented
below):

| Setting | Chunks | Rerank | Correct | GT delivered | TTFT median |
|---|---|---|---|---|---|
| A | 387 fine | no | 8/16 (50.0%) | 7/15 | 32.8 ms |
| B | 250 balanced | no | 10/16 (62.5%) | 9/15 | 33.0 ms |
| C | 173 coarse | YES | 12/16 (75.0%) | 11/15 | 444.6 ms |

- Per paper (A/B/C): attention 2/2/3 of 5; moshi 2/2/3 of 5; hnet 3/5/5
  of 5; cross 1/1/1.
- ALL 18 misses (A 8, B 6, C 4) are retrieval misses (no ground-truth
  chunk in the delivered top-5); zero invalid citations, zero refusals,
  zero missing facts across the 48 final-run rows - the citation
  pipeline and canned-answer fidelity are perfect; retrieval is the
  ONLY failure mode measured.
- Stable misses (fail in EVERY setting): 4 questions - attention "label
  smoothing", attention "positional encoding", moshi "Depth
  Transformer", moshi "Inner Monologue". MiniLM retrieves
  topically-adjacent chunks instead (e.g. hnet schedule discussions beat
  attention's training-schedule chunk for the warmup question). C's
  reranker additionally recovers 2 B-misses (learning-rate warmup,
  moshi full-duplex). A alone also misses 2 hnet questions (three-stage
  processing, tokenization drawbacks) - the fine 387-chunk split spreads
  those facts across chunk boundaries.
- REVISION (2026-11-12): the FIRST measurement (A 8/16 TTFT 36.3 ms,
  B 10/16 TTFT 32.9 ms, C 11/16 TTFT 439.4 ms) contained one grader-
  wording artifact, found by a per-miss reason audit: the full-duplex
  question's canned answer wrote "real time" (unhyphenated) while the
  verified fact is "real-time" - C had DELIVERED the ground truth
  (gt=True, cited p.1) and the substring grader failed the fact check.
  The canned sentence was fixed (the facts themselves were unchanged and
  verified in chunk text); all three settings were re-measured with
  identical harness logic. First-measurement rows remain in results.csv
  as history; the table above is the final measurement.
- Measured trade-off: C's reranker buys the best accuracy (75.0%) at
  +~400 ms retrieval latency; A is worst despite the finest chunking
  (more, smaller chunks split the facts across boundaries - e.g. hnet's
  three-stage description spans chunks in A).
- Refusal honesty was incidentally MEASURED at WP10: the weight-init
  (Glorot/Xavier) question is NOT answerable from the B store (fact
  absent from all chunks) and the real probe REFUSED it - correct
  behavior; that probe type is documented here, not part of the
  answerable 16.
- Budget note: a REAL-Groq evaluation of the same 16 questions would
  consume ~50-60K tokens (16 x ~3K, plus reasoning overhead) - EXCEEDS
  the original ~50K envelope. The user DECLINED the run (2026-11-12):
  stand-in evidence is FINAL for WP11; real-content evaluation is
  documented as future work, not an unmeasured claim about the system.

## Verified-fact log (external sources)

| Fact | Basis |
|---|---|
| Python 3.12.10 + pip 25.0.1 present; git NOT installed before `winget install Git.Git` | MEASURED locally, 2026-11-10 |
| arXiv:1706.03762v7 exists (2 Aug 2023, "15 pages, 5 figures") | DOCUMENTED, arxiv.org/abs/1706.03762, fetched 2026-11-10 |
| arXiv:2410.00037v1 exists (17 Sep 2024); latest version is v2 | DOCUMENTED, arxiv.org/abs/2410.00037, fetched 2026-11-10; brief approved to pin v1 |
| arXiv:2507.07955v2 exists (15 Jul 2025) | DOCUMENTED, arxiv.org/abs/2507.07955, fetched 2026-11-10 |
| Groq limits are org-level; RPM/RPD/TPM/TPD (+ ITPM/OTPM where configured on the account); 429 on exceed; rate-limit headers always sent, `retry-after` only on 429 | DOCUMENTED, console.groq.com/docs/rate-limits, fetched 2026-11-10 |
| Free-tier per-model limits for this account: gpt-oss-20b MEASURED 2026-11-12 from live headers (RPD 1000, TPM 8000, no RPM header); other models and TPD not measured | MEASURED for gpt-oss-20b (WP9, D18); DOCUMENTED-otherwise |
| sentence-transformers must download `all-MiniLM-L6-v2` (~90 MB) and `cross-encoder/ms-marco-MiniLM-L-6-v2` once each; offline afterwards (HF hub cache) | DOCUMENTED (HF model cards); MEASURED locally 2026-11-10 (both downloads completed, spot-checks run offline-cache-first after) |
| **Smart App Control is ON in enforcement mode** (`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy` → `VerifiedAndReputablePolicyState = 1`), and it exhibits **block-then-pass** behavior on newly downloaded compiled wheels: pandas' `pandas_parser` DLL and a scipy extension DLL were blocked on first import, then imported cleanly minutes later | MEASURED locally, 2026-11-10 (`scripts/check_env.py` fail → `scripts/diag_imports.py` pass) |
| pyarrow deliberately NOT installed (optional dep of pandas; not needed for CSV-based reporting; fewer compiled artifacts for SAC to gate) | DESIGN decision, 2026-11-10 |
