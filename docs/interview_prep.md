# Interview Prep — The Relay (2026-11-12)

Everything below is grounded in the decision log (D1–D22) and measured
artifacts. Numbers marked **M** are MEASURED; if you can't cite a number,
say "not measured" — that honesty is itself the project's thesis.

## 30-second pitch

The Relay answers engineering questions using **only** three pinned arXiv
papers. Its defining feature is honesty: an answer must cite pages it was
actually shown, or it refuses; every offered request lands as exactly one row
in a locked 12-column CSV — successes, quota blocks, and failures all look
alike in the record. Reliability layers (limiter, retry, breaker, queue,
budgets) were load-tested against a self-hosted stand-in, and the two bugs
that load testing caught are the best part of the story.

## The story arc to tell (measure → surface → decide → re-measure)

1. **D19/D20 — breaker vs quota storm.** A scripted minute-429 storm tripped
   the breaker (1 rate_limited + 19 breaker_open refusals, **M**). Surfaced the
   semantics question; decision (user-approved): the breaker counts provider
   HEALTH only — quota pressure belongs to limiter/retry. Re-measured clean:
   20× rate_limited, 0 breaker_open, retries_sum 40 **M**.
2. **Real probes → D21 — the projection ratchet.** 10 paced real probes
   produced 1 answer (TTFT 1682.7 ms **M**), 1 honest refusal, and 8 LOCAL
   quota blocks in 10–26 ms while Groq's own window demonstrably reset —
   header math exposed a double-count (observe 2924, book 2737 → projected
   187 **M**). Fixed: book before observe + minute-window aging (RPD never
   ages); unit-verified at the exact measured numbers.
3. **WP11 eval — the wording artifact.** A per-miss reason audit caught the
   grader failing "real-time" vs "real time" on an answer whose ground truth
   WAS delivered. Fixed, re-measured all settings; both measurements kept.

## Numbers to know cold

| Question | Answer |
|---|---|
| Suite | 209/209 |
| Warmup TTFT (cold → warm) | 27,626.9 → 113.2 ms **M** |
| Real wire TTFT, full RAG round-trip | ≈ 446 ms **M** (1 real answer probe: 1682.7 ms **M**) |
| Real account limits | RPD 1000 / TPM 8000 / no RPM header **M** (11 models; no llama-3.1-8b-instant) |
| Stand-in baseline | 60/60 ok, TTFT med 855.5 ms, p95 905 **M** |
| Burst (25 conc vs limit 8) | TTFT med 3003.6 ms = queue wait **M** |
| daily429 | 12× rate_limited, retries 0, rows 38–134 ms **M** (D8) |
| Breaker scenario | 25 open-refusals → 10/10 recovery **M** |
| Eval A/B/C | 8, 10, 12 of 16 = 50.0 / 62.5 / 75.0% **M**; C costs +~412 ms |
| Eval misses | ALL 18 = retrieval misses; 0 bad citations/refusals/missing facts **M** |
| Real spend | 16 offers / 13,528 tokens (user-approved batches) **M** |
| Rows recorded | 419 (real 13 / stand-in 406) **M** |

## Likely questions + grounded answers

**Why hand-rolled orchestration instead of LangGraph?** (D6) Observable
staging was the requirement: every stage transition is an event, every attempt
is attributable, and the recorder's contract (D13) needs exact semantics —
a framework hides exactly the things this project promises to show.

**Why a custom load tester instead of Locust?** (D7) The recorder and scenario
shaping needed to be part of the measurement, not around it: per-offer rows,
admin-driven fault shaping, pacing, and recovery phases are first-class in the
harness.

**How do you know answers aren't hallucinated?** (D4/D16) Chunks are
page-pinned; the prompt includes delivered evidence with arXiv IDs and pages;
the citation validator requires bracketed `[arXiv:<id> p.N]` markers matching
delivered evidence; anything else → refusal. The eval measured zero invalid
citations across 48 graded rows **M**.

**Isn't the stand-in evaluation circular?** (D22) Yes — and it says so. On the
stand-in, answer content is canned by construction; what IS measured is
retrieval quality per index config, the citation pipeline, refusal behavior,
and row honesty. A real-content eval (~50–60K tokens) was estimated and
declined by user decision — it's documented as unmeasured, not claimed.

**What did load testing actually find?** Two real bugs: a LogRecord
reserved-field crash under concurrency (fixed + regression test), and the
limiter double-count that ratcheted 8 real probes shut locally (fixed in D21,
verified at the measured numbers). Load testing paid for itself.

**Why exclude quota 429s from the breaker?** (D20) Measured trade-off: a quota
storm should cost retry TIME (honest rate_limited rows, retries_sum 40 **M**),
never fake a provider-health failure. The limiter owns admission; the breaker
owns sickness.

**Why did it refuse a question once?** The weight-initialization (Glorot/Xavier)
fact is absent from ALL chunk stores — the refusal was correct. Verified
post-hoc by scanning the store; documented in D22.

**Biggest limitation?** Three, stated upfront: (1) no real-content evaluation
(declined; unmeasured); (2) real-Groq evidence is 16 offers — sustained real
load is unmeasured; (3) 4 of 16 eval questions fail retrieval in every index
config (MiniLM topical adjacency) — characterized, not fixed.

**What next?** Real-content eval on a fresh budget; x-ratelimit-reset-* header
parsing (noted as future work in D8/D12 — the aging fix made it less urgent);
retrieval work on the 4 stable misses (query expansion or hybrid scoring).

**Why NumPy and not FAISS?** (D2) Index sizes are 173–387 chunks — exact brute
force over a 384-dim matrix is microseconds, dependency-free, and fully
recomputable. FAISS buys nothing at this scale.

## Say these before being asked (pre-empt the criticisms)

- "The stand-in's answer content is canned — here is exactly what the eval
  does and doesn't measure."
- "Real-Groq evidence is deliberately small: 16 offers, 13,528 tokens, every
  batch individually approved."
- "Two of my commits exist because measurement found bugs my tests didn't."
- "When measurement contradicted a design assumption, I re-measured and kept
  both results — the old rows stay in results.csv as history."
