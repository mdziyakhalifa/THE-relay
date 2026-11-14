# The Relay — Engineering Progress Checklist

Status values: ☐ todo · ⏳ in progress · ☑ done · ⏸ blocked / awaiting approval

## Phases

- ☑ Phase 1 — Requirements, design, architecture, plan (approved 2026-11-10)
- ☑ Phase 2 — Environment setup (WP0)
- ☑ Phase 3 — Paper ingestion & RAG (WP2, WP3)
- ☐ Phase 4 — Groq pipeline (WP5, WP7, WP8)
- ☐ Phase 5 — Stand-in service (WP6)
- ☐ Phase 6 — Reliability hardening (WP4 integration)
- ☐ Phase 7 — Experiments (WP9 calibration is user-gated, WP10)
- ☑ Phase 8 — Analysis (WP11: stand-in eval measured A/B/C; real-Groq eval declined by user decision — stand-in evidence final)
- ◐ Phase 9 — Report (two-page report written + approved by user → `report/report.md`; interview prep → `docs/interview_prep.md`; Vaani-feedback session remains a user-visit item)
- ☑ Phase 10 — Final review (integrity review PASSED: results.csv contract verified — exact 12-column header, 419 rows, 0 enum violations, 0 missing ids; full-tree leak scan clean; D1–D22 present; suite 209/209; README refreshed; all work committed)

## Work packages

| WP | Deliverable | Status | Confirmed on |
|---|---|---|---|
| WP0 | venv, dependencies, secrets, verification | ☑ (GROQ key still pending — needed only at WP5; SAC block-then-pass quirk documented) | 2026-11-10 |
| WP1 | Core: settings, logging, request context, stage metrics | ☐ | |
| WP2 | Paper download/extraction/chunking (page-pinned) | ☑ (3 papers extracted; 25/25 tests; pagenum integrity verified; 3 hnet figure labels = documented residual noise) | 2026-11-10 |
| WP3 | Embeddings, index, retriever (+ optional reranker) | ☑ (3 configs built 387/250/173 chunks, dim 384; 250==WP2 preview cross-check ✔; 37/37 tests; spot-checks A/B/C run; cross-encoder scores = raw logits recorded D9) | 2026-11-10 |
| WP4 | Reliability: limiter, retry, breaker, queue, budgets, recorder | ☑ (6 primitives + recorder per user-locked 12-column contract, D13; 113/113 tests incl. 42 recorder tests) | 2026-11-11 |
| WP5 | Providers: Groq (HTTPX) + stand-in client, typed errors | ☑ (shared compat transport D14; Groq/stand-in clients; 13 provider tests; 126/126 suite; ZERO live traffic — key still pending, needed only at WP9) | 2026-11-11 |
| WP6 | Stand-in FastAPI service + failure-mode tests | ☑ (documented tokenizer D15; 7 scripted failure modes w/ admin API; real-wire + real-uvicorn e2e tests; 164/164 suite) | 2026-11-11 |
| WP7 | Orchestration: RelayRunner, stages, refusal, regen, citation validator | ☑ (D16: citation contract, TTFT/first-token rule, per-attempt composition; per-request event files; 2 bugs found+fixed (capacity cancel leak, stream fidelity); 187/187 suite) | 2026-11-11 |
| WP8 | API layer: /ask, /ask/stream, /health, timeouts | ☑ (D17: HTTP outcome mapping, validation→400/409 with no row, SSE grammar + bounded-queue drop semantics, Windows disconnect fix → user_cancelled row; boot warmup (cold TTFT 27626.9 ms → warm 113.2 ms MEASURED); 17 new tests, 204/204 suite; zero Groq traffic) | 2026-11-11 |
| WP9 | Real calibration (NEEDS USER SIGN-OFF before sending) | ☑ (user approved A+B+C; Steps A/B/C executed with sign-off — measured: gpt-oss-20b RPD 1000/TPM 8000/no RPM, wire TTFT ~446 ms, full real RAG round-trip ok with valid citations; local TPM projection blocks under back-to-back load = designed behavior; 0 real 429s; budget 6 offers/5,189 tokens of approved 100/~50K; optional Step D not run — D18) | 2026-11-12 |
| WP10 | Load experiments: 5 scenarios × stand-in + small real probes | ☑ (all 5 stand-in scenarios + D20 re-measurement measured; breaker LogRecord bug found+fixed under load; 10 real Groq probes executed per sign-off — 2 ok incl. 1 honest refusal + 8 honest local blocks; D21 limiter double-count + aging fix unit-verified at the measured numbers, stand-in regression clean, suite 209/209; real evidence final per user "no re-run") | 2026-11-12 |
| WP11 | Evaluation (≥15 questions), grading, report tables/plots | ☑ (16 fact-checked questions + deterministic grading harness `scripts/run_eval.py`; stand-in runs MEASURED: A 8/16 50.0% TTFT 32.8 ms, B 10/16 62.5% 33.0 ms, C 12/16 75.0% 444.6 ms (reranker, +~400 ms); ALL 18 misses retrieval-only — 0 citation failures/refusals/missing facts; stable misses in every setting: label smoothing, positional encoding, Depth Transformer, Inner Monologue; hnet 5/5 in B/C; grader wording artifact found+fixed via per-miss audit, both measurements kept; Glorot weight-init fact absent from stores → WP10 real-probe refusal was correct honesty; D13 invariant 16/16 rows per run; real-Groq eval DECLINED by user decision 2026-11-12 — stand-in evidence final; suite 209/209) | 2026-11-12 |
| WP12 | Final integrity review, README, interview prep | ☑ (two-page report `report/report.md` approved as-is by user; README refreshed to completed status with architecture + measured headline numbers; `docs/interview_prep.md` written — pitch, story arc, numbers table, grounded Q&A; final integrity review PASSED: results.csv 419 rows / exact contract / 0 violations, full-tree leak scan clean, D1–D22 present, suite 209/209; only open item: Vaani-feedback user visit) | 2026-11-12 |

## Guardrails in force

- No Groq requests beyond the approved calibration budget
  (≤100 requests / ~50K tokens), and none at all without explicit sign-off at WP9.
- No fabricated numbers: unmeasured values are labeled `not measured`.
- `results.csv` schema is validated by script before submission.
- API keys only ever in `.env` (git-ignored); never in chat or logs.
