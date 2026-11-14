"""WP10: load experiments against the REAL API stack + stand-in provider.

Run (from the project root):

    .venv\\Scripts\\python.exe -m scripts.run_load_tests --scenario all
    .venv\\Scripts\\python.exe -m scripts.run_load_tests --scenario baseline

The harness boots its own uvicorn servers (real API with real A/B/C
retrievers + the stand-in provider, identical wiring to scripts.run_api
via build_relay_app), executes the user-confirmed scenario set, and
writes:
    rows      -> results/results.csv      (via the recorder, source=stand-in)
    events    -> results/events/          (per-request, D13)
    per-req   -> results/logs/load_<scenario>.jsonl
    plots     -> results/plots/wp10_<scenario>.png
    summary   -> results/logs/wp10_summary.json

Honesty rules (D13/D16/D17): ONE row per offered request (the harness
verifies this invariant per scenario), no client-side retries, TTFT
comes from the runner's internal measurement (offer -> first content
token, includes queue wait), statuses are the row statuses, and nothing
here is presented as real-provider behavior.

Scenarios (user-confirmed 2026-11-12):
    baseline : ok-mode, ~0.8 s first-token shaping, 4 concurrent, 60 offers
    burst    : ok-mode, 25 concurrent vs in-flight limit 8 -> queue waits
    minute429: scripted minute-429 (retry-after 0.5 s) -> backoff + honest rows
    daily429 : scripted daily-429 -> D8 never-retry, instant honest rows
    breaker  : scripted 5xx -> breaker opens -> local refusals -> recovery
    real_probe : REAL Groq (user-approved; --provider groq --model <id>)
        10 sequential offers paced 30 s apart through the same API boot
        path; rows source=real; scenario label wp10-real-probe.
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
from fastapi import FastAPI

from app.core.logging_setup import force_utf8_stdio
from app.standin.service import create_app as create_standin_app
from scripts.run_api import build_relay_app, _warmup

RESULTS = Path("results")
QUESTIONS = None  # filled in _main from the spot-check probes


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# ------------------------------------------------------------- scenario cfg

SCENARIOS: dict[str, dict] = {
    "baseline": {
        "mode": "ok",
        "shaping": {"first_token_delay_ms": 800},
        "concurrency": 4,
        "requests": 60,
    },
    "burst": {
        "mode": "ok",
        "shaping": {"first_token_delay_ms": 500},
        "concurrency": 25,
        "requests": 60,
    },
    "minute429": {
        "mode": "rate_limited_minute",
        "mode_body": {"retry_after_s": 0.5},  # default 3600 s would stall to deadline
        "shaping": {},
        "concurrency": 2,
        "requests": 20,
    },
    "daily429": {
        "mode": "rate_limited_daily",
        "shaping": {},
        "concurrency": 4,
        "requests": 12,
    },
    "breaker": {
        "mode": "server_error",  # provider 503s -> retryable -> breaker counts
        "shaping": {},
        "concurrency": 4,
        "requests": 25,  # phase A; phase B (recovery) adds 10 more
        "recovery_requests": 10,
    },
    # REAL-provider probe (user-approved 2026-11-12): sequential, paced,
    # through the same API boot path with provider=groq. No stand-in
    # server, no admin shaping; every row source=real. pace_s keeps each
    # offer >30 s apart so the measured TPM 8000 (D18) is not the
    # binding constraint; a drifted offer lands as honest quota_block_local.
    "real_probe": {
        "provider": "groq",
        "label": "wp10-real-probe",
        "mode": None,  # no stand-in admin configuration
        "shaping": {},
        "concurrency": 1,
        "requests": 10,
        "pace_s": 30.0,
    },
}


# ------------------------------------------------------------------ servers


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


async def _configure_standin(base: str, cfg: dict) -> None:
    if cfg.get("mode") is None:  # real-provider scenario: no stand-in shaping
        return
    async with httpx.AsyncClient() as http:
        await http.post(f"{base}/_admin/reset", timeout=10.0)
        body = {"mode": cfg["mode"], **cfg.get("mode_body", {})}
        response = await http.post(f"{base}/_admin/mode", json=body, timeout=10.0)
        response.raise_for_status()
        if cfg.get("shaping"):
            response = await http.post(
                f"{base}/_admin/shaping", json=cfg["shaping"], timeout=10.0
            )
            response.raise_for_status()


# ------------------------------------------------------------------- offers


async def _offer(
    client: httpx.AsyncClient,
    api: str,
    request_id: str,
    question: str,
    scenario: str,
) -> dict:
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{api}/ask",
            json={
                "question": question,
                "retrieval_setting": "B",
                "scenario": scenario,
                "request_id": request_id,
            },
            timeout=90.0,
        )
    except httpx.TimeoutException:
        return {
            "request_id": request_id,
            "http_status": None,
            "error": "client_timeout",
            "wall_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
    wall_ms = (time.perf_counter() - started) * 1000.0
    try:
        body = response.json()
    except Exception:
        body = {}
    return {
        "request_id": request_id,
        "http_status": response.status_code,
        "row_status": body.get("status"),
        "outcome_kind": body.get("outcome_kind"),
        "refused": body.get("refused"),
        "first_token_ms": body.get("first_token_ms"),
        "total_ms": body.get("total_ms"),
        "retries": body.get("retries"),
        "tokens_in": body.get("tokens_in"),
        "tokens_out": body.get("tokens_out"),
        "wall_ms": round(wall_ms, 1),
    }


async def _run_offers(
    api: str,
    *,
    scenario: str,
    runid: str,
    concurrency: int,
    requests: int,
    pace_s: float = 0.0,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 2)
    records: list[dict] = []
    offer_start = time.perf_counter()

    async with httpx.AsyncClient(limits=limits) as client:

        async def one(index: int) -> None:
            question = QUESTIONS[index % len(QUESTIONS)]
            request_id = f"{runid}-{index:03d}"
            async with semaphore:
                if pace_s > 0.0 and index > 0:
                    # pacing between offer SUBMISSIONS (sequential offers:
                    # start-to-start spacing = pace_s + previous duration)
                    await asyncio.sleep(pace_s)
                record = await _offer(client, api, request_id, question, scenario)
            record["offer_index"] = index
            record["offer_offset_s"] = round(time.perf_counter() - offer_start, 2)
            records.append(record)

        await asyncio.gather(*(one(i) for i in range(requests)))

    records.sort(key=lambda r: r["offer_index"])
    return records


# -------------------------------------------------------------- aggregation


def _stat(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        index = min(len(ordered) - 1, round(p * (len(ordered) - 1)))
        return round(ordered[index], 1)

    return {
        "n": len(ordered),
        "min": round(ordered[0], 1),
        "median": pct(0.5),
        "p95": pct(0.95),
        "max": round(ordered[-1], 1),
    }


def _count_by(records: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


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


def _write_plot(name: str, records: list[dict]) -> Path | None:
    ttfts = [r["first_token_ms"] for r in records if isinstance(r.get("first_token_ms"), (int, float))]
    walls = [(r["offer_offset_s"], r["wall_ms"]) for r in records if r.get("wall_ms") is not None]
    if not ttfts and not walls:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    if ttfts:
        ax1.hist(ttfts, bins=20, color="#3b6ea5")
        ax1.set_xlabel("first_token_ms (internal, incl. queue wait)")
        ax1.set_ylabel("requests")
        ax1.set_title(f"WP10 {name}: TTFT distribution")
    if walls:
        ax2.scatter([w[0] for w in walls], [w[1] for w in walls], s=14, color="#a53b3b")
        ax2.set_xlabel("offer time (s since scenario start)")
        ax2.set_ylabel("client wall ms")
        ax2.set_title(f"WP10 {name}: end-to-end wall latency")
    fig.tight_layout()
    out = RESULTS / "plots" / f"wp10_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


async def _run_scenario(
    name: str,
    cfg: dict,
    api: str,
    standin: str,
    summary: dict,
) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    runid = f"wp10-{name}-{stamp}"
    print(f"\n=== scenario {name} (run {runid}) ===")
    await _configure_standin(standin, cfg)

    offers = cfg["requests"] + cfg.get("recovery_requests", 0)
    started = time.perf_counter()
    records = await _run_offers(
        api,
        scenario=cfg.get("label", name),
        runid=runid,
        concurrency=cfg["concurrency"],
        requests=cfg["requests"],
        pace_s=cfg.get("pace_s", 0.0),
    )
    if name == "breaker":
        # phase B: restore ok-mode, wait out the breaker's open window, recover
        print(f"  phase A done ({len(records)} offers); switching mode to ok")
        await _configure_standin(standin, {"mode": "ok", "shaping": {"first_token_delay_ms": 300}})
        await asyncio.sleep(21.0)  # open_seconds=20 -> next probe is half-open
        recovery = await _run_offers(
            api,
            scenario=f"{name}_recovery",
            runid=f"{runid}-rec",
            concurrency=2,
            requests=cfg["recovery_requests"],
        )
        records = records + recovery
    wall_s = round(time.perf_counter() - started, 1)

    rows = _verify_row_invariant(runid, offers)
    per_req = RESULTS / "logs" / f"load_{name}.jsonl"
    per_req.parent.mkdir(parents=True, exist_ok=True)
    with per_req.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    plot = _write_plot(name, records)

    ttft = _stat([r["first_token_ms"] for r in records if isinstance(r.get("first_token_ms"), (int, float))])
    total = _stat([r["total_ms"] for r in records if isinstance(r.get("total_ms"), (int, float))])
    walls = _stat([r["wall_ms"] for r in records if r.get("wall_ms") is not None])
    aggregate = {
        "scenario": name,
        "run_id": runid,
        "offers": offers,
        "rows_recorded": rows,
        "row_status_counts": _count_by(records, "row_status"),
        "http_status_counts": _count_by(records, "http_status"),
        "outcome_kind_counts": _count_by(records, "outcome_kind"),
        "retries_sum": sum(r.get("retries") or 0 for r in records if isinstance(r.get("retries"), int)),
        "ttft_ms": ttft,
        "total_ms": total,
        "wall_ms": walls,
        "wall_clock_s": wall_s,
        "plot": str(plot) if plot else None,
    }
    summary[name] = aggregate
    print(json.dumps(aggregate, indent=2))


# --------------------------------------------------------------------- main


async def _main() -> None:
    global QUESTIONS
    force_utf8_stdio()
    parser = argparse.ArgumentParser(description="WP10 load experiments")
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=tuple(SCENARIOS) + ("all",),
        default=["all"],
    )
    parser.add_argument(
        "--provider",
        choices=("standin", "groq"),
        default="standin",
        help="standin: the 5 stand-in scenarios. groq: REAL probes only "
        "(user-approved; requires --model).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id (REQUIRED with --provider groq; e.g. openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--pace-s",
        type=float,
        default=None,
        help="override the scenario's pacing between offer submissions (s)",
    )
    args = parser.parse_args()

    from scripts.retrieval_spotcheck import PROBES

    QUESTIONS = [question for _, question in PROBES]

    names = list(SCENARIOS) if args.scenario == ["all"] else args.scenario
    if args.provider == "groq":
        if not args.model:
            raise SystemExit("--model is required with --provider groq")
        if names != ["real_probe"]:
            raise SystemExit(
                "--provider groq supports only: --scenario real_probe "
                "(the stand-in scenarios need the stand-in provider)"
            )
        if "GROQ_API_KEY" not in _load_env_keys():
            raise SystemExit(
                "GROQ_API_KEY not set - create .env first (never send traffic without it)"
            )

    # --- boot the real stack (identical wiring to scripts.run_api) ---
    if args.provider == "standin":
        standin_server, standin_thread, standin_url = _start_server(
            create_standin_app(), "stand-in"
        )
    else:
        standin_server, standin_thread, standin_url = None, None, None
    app, parts = build_relay_app(
        provider_name=args.provider,
        model=args.model if args.provider == "groq" else None,
        standin_base_url=standin_url,
    )
    if args.pace_s is not None:  # CLI override wins over scenario config
        for name in names:
            SCENARIOS[name]["pace_s"] = args.pace_s
    await _warmup(parts["retrievers"])
    api_server, api_thread, api_url = _start_server(app, "api")
    print(
        f"stack up: api={api_url} stand-in={standin_url} "
        f"(provider={args.provider}, model={parts['model']}, "
        f"source={parts['source_label']})"
    )

    # merge with any previous summary so separate invocations accumulate
    out = RESULTS / "logs" / "wp10_summary.json"
    summary: dict = {}
    if out.is_file():
        try:
            summary = json.loads(out.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    try:
        for name in names:
            cfg = SCENARIOS[name]
            await _run_scenario(name, cfg, api_url, standin_url, summary)
    finally:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nsummary -> {out}")
        api_server.should_exit = True
        if standin_server is not None:
            standin_server.should_exit = True
        api_thread.join(timeout=10)
        if standin_thread is not None:
            standin_thread.join(timeout=10)


def _load_env_keys() -> set[str]:
    """Names of keys defined in .env (values NEVER read into memory here).

    utf-8-sig: PowerShell's Set-Content -Encoding utf8 writes a BOM, which
    would otherwise poison the first key name (measured: the guard caught
    this BEFORE any traffic was sent)."""
    path = Path(".env")
    if not path.is_file():
        return set()
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            names.add(stripped.split("=", 1)[0].strip())
    return names


if __name__ == "__main__":
    asyncio.run(_main())
