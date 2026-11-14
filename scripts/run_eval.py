"""WP11: evaluation harness - 16 fact-checked questions, automated grading.

Run (from the project root):

    .venv\\Scripts\\python.exe -m scripts.run_eval --setting B
    .venv\\Scripts\\python.exe -m scripts.run_eval --setting A
    .venv\\Scripts\\python.exe -m scripts.run_eval --setting C
    .venv\\Scripts\\python.exe -m scripts.run_eval --limit 2   (smoke)

Boots the REAL stack (identical wiring to scripts.run_api via
build_relay_app + the stand-in provider) and, for each question:

1. PRE-RETRIEVES the question through the same retriever the runner uses
   (deterministic) - recording the delivered chunk ids and whether ANY
   GROUND-TRUTH chunk was delivered (the honest retrieval-quality measure).
2. SHAPES the stand-in with a canned answer that (a) states the verified
   facts and (b) cites the delivered ground-truth chunk (or the top
   delivered chunk when the ground truth was NOT delivered - the row then
   grades incorrect with reason "retrieval miss").
3. OFFERS the question through POST /ask (sequential, no load).
4. GRADES deterministically (no LLM judge; auditable string logic):
       correct   : not refused AND citations valid AND every expected
                   fact present AND ground-truth chunk delivered
       refused   : the runner refused (honest row; recorded separately)
       incorrect : otherwise (retrieval miss / invalid citations /
                   missing facts / errors)

HONEST SCOPE (D22): on the stand-in, answer CONTENT is canned by
construction - the facts check therefore exercises stream fidelity +
grading mechanics, not reasoning. What IS genuinely measured here:
retrieval quality per setting, the citation pipeline, refusal behavior,
and end-to-end row honesty. Real-content evaluation requires the real
provider (separate user sign-off).

Every question's facts were verified against the ACTUAL config_b chunk
texts before inclusion (ground-truth chunk sets computed by a scripted
scan). The Attention paper's weight-initialization fact (Glorot/Xavier)
is NOT in the B store - the WP10 real probe's refusal of it was CORRECT
honesty; documented in the eval report, not part of this answerable set.

Outputs (merged across invocations, keyed by retrieval setting):
    evaluations/eval_results.json   per-question records + aggregates
    rows -> results/results.csv (scenario wp11-eval), events -> results/events/
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import uvicorn

from app.core.logging_setup import force_utf8_stdio
from app.standin.service import create_app as create_standin_app
from scripts.run_api import build_relay_app, _warmup

RESULTS = Path("results")
EVAL_DIR = Path("evaluations")

# ----------------------------------------------------------- question set
# (paper_key, question, expected_facts, ground_truth_chunk_ids | None,
#  canned_fact_sentence)
# expected_facts: case-insensitive substrings a correct answer must contain.
# ground_truth_chunk_ids: ALL config_b chunks whose text contains every
# expected fact (computed by a scripted scan over the chunk store before
# inclusion); delivery of ANY of them counts as an evidence hit. None for
# the cross-paper question.

QUESTIONS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...] | None, str], ...] = (
    # --- Attention Is All You Need (1706.03762v7) ---
    (
        "attention",
        "How does multi-head attention extend the attention mechanism?",
        ("parallel", "attention layers"),
        ("attention|1706.03762v7|p4|0", "attention|1706.03762v7|p5|0", "attention|1706.03762v7|p6|1"),
        "Multi-head attention extends attention by running several attention layers in parallel.",
    ),
    (
        "attention",
        "Why is the learning rate warmed up and then decayed?",
        ("linear", "square root"),
        ("attention|1706.03762v7|p7|1",),
        "The learning rate increases linearly for the first warmup_steps and then decreases proportionally to the inverse square root of the step number.",
    ),
    (
        "attention",
        "What label smoothing value did the Transformer use, and what did it trade off?",
        ("0.1", "perplexity", "BLEU"),
        ("attention|1706.03762v7|p8|0",),
        "The Transformer used label smoothing of value 0.1, which hurts perplexity but improves accuracy and BLEU score.",
    ),
    (
        "attention",
        "How does the Transformer inject word-order information without recurrence or convolution?",
        ("positional encoding", "sine", "cosine"),
        ("attention|1706.03762v7|p6|0",),
        "Word order is injected through positional encoding using sine and cosine functions of different frequencies.",
    ),
    (
        "attention",
        "What BLEU scores did the big Transformer achieve on WMT 2014 English-to-German and English-to-French?",
        ("28.4", "41.8"),
        ("attention|1706.03762v7|p1|0", "attention|1706.03762v7|p8|0"),
        "The big Transformer achieved 28.4 BLEU on WMT 2014 English-to-German and 41.8 on English-to-French.",
    ),
    # --- Moshi (2410.00037v1) ---
    (
        "moshi",
        "How does Moshi talk and listen at the same time?",
        ("full-duplex", "real-time"),
        ("moshi|2410.00037v1|p1|0", "moshi|2410.00037v1|p1|1", "moshi|2410.00037v1|p2|1",
         "moshi|2410.00037v1|p3|0", "moshi|2410.00037v1|p5|1", "moshi|2410.00037v1|p6|0",
         "moshi|2410.00037v1|p41|0"),
        "Moshi is a full-duplex spoken dialogue framework that always listens and always speaks in real-time.",
    ),
    (
        "moshi",
        "What are semantic and acoustic audio tokens in Mimi?",
        ("semantic", "acoustic", "Mimi"),
        ("moshi|2410.00037v1|p1|0", "moshi|2410.00037v1|p3|0", "moshi|2410.00037v1|p3|1",
         "moshi|2410.00037v1|p4|0", "moshi|2410.00037v1|p4|1", "moshi|2410.00037v1|p5|0",
         "moshi|2410.00037v1|p6|0", "moshi|2410.00037v1|p6|1", "moshi|2410.00037v1|p7|0",
         "moshi|2410.00037v1|p9|0", "moshi|2410.00037v1|p9|1", "moshi|2410.00037v1|p10|0",
         "moshi|2410.00037v1|p11|1", "moshi|2410.00037v1|p12|0", "moshi|2410.00037v1|p14|0",
         "moshi|2410.00037v1|p14|1", "moshi|2410.00037v1|p15|0", "moshi|2410.00037v1|p15|1",
         "moshi|2410.00037v1|p16|1", "moshi|2410.00037v1|p23|0", "moshi|2410.00037v1|p23|1",
         "moshi|2410.00037v1|p24|0", "moshi|2410.00037v1|p25|0", "moshi|2410.00037v1|p25|1",
         "moshi|2410.00037v1|p26|1", "moshi|2410.00037v1|p27|1", "moshi|2410.00037v1|p28|1",
         "moshi|2410.00037v1|p29|1", "moshi|2410.00037v1|p34|0", "moshi|2410.00037v1|p41|0",
         "moshi|2410.00037v1|p56|0", "moshi|2410.00037v1|p57|0", "moshi|2410.00037v1|p58|0"),
        "Mimi produces discrete audio tokens whose first level carries semantic information while the remaining levels carry acoustic detail.",
    ),
    (
        "moshi",
        "What is the role of the Depth Transformer in Moshi's architecture?",
        ("Depth Transformer", "streaming"),
        ("moshi|2410.00037v1|p7|0", "moshi|2410.00037v1|p14|1", "moshi|2410.00037v1|p21|0"),
        "The Depth Transformer jointly models the audio streams of Moshi and the user together with Moshi's text tokens, and is compatible with streaming inference.",
    ),
    (
        "moshi",
        "What is Inner Monologue and what does it improve?",
        ("Inner Monologue", "text tokens", "factuality"),
        ("moshi|2410.00037v1|p3|1",),
        "Inner Monologue predicts time-aligned text tokens before audio tokens, improving the factuality and linguistic quality of generated speech.",
    ),
    (
        "moshi",
        "Why does Moshi force a delay between text and audio tokens?",
        ("delay", "ASR", "TTS"),
        ("moshi|2410.00037v1|p3|1", "moshi|2410.00037v1|p5|1", "moshi|2410.00037v1|p16|0",
         "moshi|2410.00037v1|p16|1", "moshi|2410.00037v1|p31|0", "moshi|2410.00037v1|p41|1",
         "moshi|2410.00037v1|p55|0"),
        "Forcing a delay between text and audio tokens allows deriving streaming ASR and streaming TTS from the model.",
    ),
    # --- H-Net (2507.07955v2) ---
    (
        "hnet",
        "What does dynamic chunking replace in hierarchical models?",
        ("tokenization", "dynamic chunking"),
        ("hnet|2507.07955v2|p1|0", "hnet|2507.07955v2|p2|0", "hnet|2507.07955v2|p4|1",
         "hnet|2507.07955v2|p13|0", "hnet|2507.07955v2|p31|1"),
        "Dynamic chunking replaces fixed-vocabulary tokenization with content- and context-dependent segmentation learned from data.",
    ),
    (
        "hnet",
        "How does H-Net select chunk boundaries robustly?",
        ("routing", "smoothing"),
        ("hnet|2507.07955v2|p2|1", "hnet|2507.07955v2|p3|0", "hnet|2507.07955v2|p6|1",
         "hnet|2507.07955v2|p17|1", "hnet|2507.07955v2|p18|0"),
        "A routing module dynamically draws chunk boundaries, and a smoothing module makes the boundaries robust.",
    ),
    (
        "hnet",
        "Describe the three stages of H-Net's hierarchical processing.",
        ("encoder", "main network", "decoder"),
        ("hnet|2507.07955v2|p2|1", "hnet|2507.07955v2|p4|1", "hnet|2507.07955v2|p4|2",
         "hnet|2507.07955v2|p5|0", "hnet|2507.07955v2|p5|1", "hnet|2507.07955v2|p6|0",
         "hnet|2507.07955v2|p9|0", "hnet|2507.07955v2|p9|1", "hnet|2507.07955v2|p10|0",
         "hnet|2507.07955v2|p19|0", "hnet|2507.07955v2|p20|0", "hnet|2507.07955v2|p22|0",
         "hnet|2507.07955v2|p37|0", "hnet|2507.07955v2|p37|1"),
        "Raw data is processed by a small encoder network, downsampled through a main network operating on compressed chunks, then upsampled through a decoder network.",
    ),
    (
        "hnet",
        "What drawbacks of fixed-vocabulary tokenization motivate end-to-end byte-level models?",
        ("byte-pair encoding", "character"),
        ("hnet|2507.07955v2|p1|1", "hnet|2507.07955v2|p31|0"),
        "Fixed-vocabulary byte-pair encoding causes poor character-level understanding, lack of interpretability, and degraded performance.",
    ),
    (
        "hnet",
        "Why is simple pooling (compressing every d inputs) inadequate for byte-level modeling?",
        ("data-dependent", "variable information rates"),
        ("hnet|2507.07955v2|p2|0",),
        "Simple pooling such as compressing every d inputs is not data-dependent and performs poorly on modalities with variable information rates.",
    ),
    # --- cross-paper (no single ground-truth chunk) ---
    (
        "cross",
        "How are encoder and decoder used differently in these models?",
        ("encoder", "decoder"),
        None,
        "The encoder represents or compresses the input while the decoder generates the output, and each model arranges these roles differently.",
    ),
)


def _grade(
    body: dict,
    expected: tuple[str, ...],
    ground_truth_required: bool,
    ground_truth_retrieved: bool | None,
) -> dict:
    refused = bool(body.get("refused"))
    citations = body.get("citations") or {}
    citations_valid = citations.get("valid")
    answer = str(body.get("answer") or "")
    lowered = answer.lower()
    missing = [fact for fact in expected if fact.lower() not in lowered]
    reasons: list[str] = []

    if refused:
        grade = "refused"
        reasons.append("runner refused the question")
    elif ground_truth_required and ground_truth_retrieved is False:
        grade = "incorrect"
        reasons.append("retrieval miss: no ground-truth chunk in delivered evidence")
    elif not citations_valid:
        grade = "incorrect"
        reasons.append("citations invalid")
    elif missing:
        grade = "incorrect"
        reasons.append(f"missing expected fact(s): {missing}")
    else:
        grade = "correct"
    return {
        "grade": grade,
        "citations_valid": citations_valid,
        "missing_facts": missing,
        "reasons": reasons,
    }


def _start_server(app, label: str):
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name=label)
    thread.start()
    deadline = time.monotonic() + 30.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"{label} did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


def _verify_row_invariant(runid: str, offers: int) -> int:
    """D13: every offered request must have exactly one row."""
    path = RESULTS / "results.csv"
    if not path.is_file():
        raise SystemExit("results.csv missing - rows were never written")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and r[0].startswith(runid)]
    if len(rows) != offers:
        raise SystemExit(
            f"D13 INVARIANT VIOLATION for run {runid}: {len(rows)} rows "
            f"for {offers} offered requests"
        )
    return len(rows)


async def _main() -> None:
    force_utf8_stdio()
    parser = argparse.ArgumentParser(description="WP11 evaluation harness")
    parser.add_argument("--setting", choices=("A", "B", "C"), default="B")
    parser.add_argument(
        "--limit", type=int, default=None, help="smoke: only the first N questions"
    )
    args = parser.parse_args()

    standin_server, standin_thread, standin_url = _start_server(
        create_standin_app(), "stand-in"
    )
    app, parts = build_relay_app(
        provider_name="standin", model=None, standin_base_url=standin_url
    )
    await _warmup(parts["retrievers"])
    api_server, api_thread, api_url = _start_server(app, "api")
    print(f"stack up: api={api_url} stand-in={standin_url} (provider=stand-in)")

    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    runid = f"wp11-eval-{args.setting.lower()}-{stamp}"
    questions = QUESTIONS if args.limit is None else QUESTIONS[: args.limit]
    print(f"run {runid}: {len(questions)} question(s), setting {args.setting}\n")

    retriever = parts["retrievers"][args.setting]
    records: list[dict] = []
    try:
        async with httpx.AsyncClient() as client:
            for index, (paper, question, expected, gt_chunks, fact_sentence) in enumerate(
                questions
            ):
                # ---- 1. pre-retrieve (same query the runner will use) ----
                retrieval = await retriever.retrieve(question)
                delivered_ids = [item.chunk_id for item in retrieval.items]
                ground_truth_retrieved: bool | None = (
                    any(chunk_id in delivered_ids for chunk_id in gt_chunks)
                    if gt_chunks
                    else None
                )

                # ---- 2. shape the canned answer ----
                if gt_chunks and ground_truth_retrieved:
                    target = next(
                        i for i in retrieval.items if i.chunk_id in gt_chunks
                    )
                else:
                    target = retrieval.items[0]
                canned = (
                    f"{fact_sentence} [arXiv:{target.versioned_id} p.{target.page}]"
                )
                await client.post(
                    f"{standin_url}/_admin/shaping",
                    json={"completion_text": canned},
                    timeout=10.0,
                )

                # ---- 3. offer through the real API ----
                request_id = f"{runid}-{index:03d}"
                started = time.perf_counter()
                response = await client.post(
                    f"{api_url}/ask",
                    json={
                        "question": question,
                        "retrieval_setting": args.setting,
                        "scenario": "wp11-eval",
                        "request_id": request_id,
                    },
                    timeout=120.0,
                )
                wall_ms = round((time.perf_counter() - started) * 1000.0, 1)
                try:
                    body = response.json()
                except Exception:
                    body = {}
                grade = _grade(
                    body, expected, gt_chunks is not None, ground_truth_retrieved
                )
                record = {
                    "request_id": request_id,
                    "paper_key": paper,
                    "question": question,
                    "setting": args.setting,
                    "expected_facts": list(expected),
                    "ground_truth_chunk_ids": list(gt_chunks) if gt_chunks else None,
                    "ground_truth_retrieved": ground_truth_retrieved,
                    "delivered_chunk_ids": delivered_ids,
                    "cited_chunk": f"{target.versioned_id} p.{target.page}",
                    "canned_answer": canned,
                    "http_status": response.status_code,
                    "row_status": body.get("status"),
                    "outcome_kind": body.get("outcome_kind"),
                    "refused": bool(body.get("refused")),
                    "citations_valid": grade["citations_valid"],
                    "first_token_ms": body.get("first_token_ms"),
                    "total_ms": body.get("total_ms"),
                    "wall_ms": wall_ms,
                    "tokens_in": body.get("tokens_in"),
                    "tokens_out": body.get("tokens_out"),
                    "answer": str(body.get("answer") or ""),
                    "grade": grade["grade"],
                    "missing_facts": grade["missing_facts"],
                    "reasons": grade["reasons"],
                }
                records.append(record)
                flag = {"correct": "OK ", "refused": "REF", "incorrect": "BAD"}[
                    grade["grade"]
                ]
                gt = (
                    "-"
                    if ground_truth_retrieved is None
                    else ("Y" if ground_truth_retrieved else "N")
                )
                print(
                    f"[{index + 1:02d}/{len(questions)}] {flag} gt={gt} {paper:9s}"
                    f" ttft={record['first_token_ms']} total={record['total_ms']}"
                    f" | {question[:62]}"
                )
                if grade["reasons"]:
                    print(f"        reasons: {'; '.join(grade['reasons'])}")
    finally:
        rows = _verify_row_invariant(runid, len(questions))
        api_server.should_exit = True
        standin_server.should_exit = True
        api_thread.join(timeout=10)
        standin_thread.join(timeout=10)

    # ---- merge with previous results (keyed by setting) ----
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "eval_results.json"
    merged: dict = {}
    if out.is_file():
        try:
            merged = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged.setdefault("meta", {})
    merged["meta"]["last_run"] = runid
    merged.setdefault("results", {})[args.setting] = records

    def _aggregate(recs: list[dict]) -> dict:
        counts: dict[str, int] = {}
        for r in recs:
            counts[r["grade"]] = counts.get(r["grade"], 0) + 1
        gt_known = [r for r in recs if r["ground_truth_retrieved"] is not None]
        gt_hits = sum(1 for r in gt_known if r["ground_truth_retrieved"])
        ttfts = [
            r["first_token_ms"]
            for r in recs
            if isinstance(r.get("first_token_ms"), (int, float))
        ]
        return {
            "n": len(recs),
            "rows_recorded": rows,
            "grade_counts": dict(sorted(counts.items())),
            "ground_truth_retrieved": f"{gt_hits}/{len(gt_known)}",
            "accuracy_excl_refusals": round(
                100.0
                * counts.get("correct", 0)
                / max(1, len(recs) - counts.get("refused", 0)),
                1,
            ),
            "ttft_ms_median": sorted(ttfts)[len(ttfts) // 2] if ttfts else None,
        }

    merged.setdefault("summary", {})[args.setting] = _aggregate(records)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsummary[{args.setting}]: {json.dumps(merged['summary'][args.setting])}")
    print(f"results -> {out}")


if __name__ == "__main__":
    asyncio.run(_main())
