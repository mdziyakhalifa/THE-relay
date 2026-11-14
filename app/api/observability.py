"""Read-only observability endpoints for the frontend.

These endpoints expose existing recorded artifacts (results CSV, retrieval
settings, evaluation results, event logs) without altering core relay
behavior. They are additive and safe: if files are missing, responses
report unavailability rather than fabricating data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.core.settings import get_settings

router = APIRouter(prefix="/observability", tags=["observability"])

RESULT_COLUMNS = [
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
]

PAPERS = [
    {
        "paper_key": "attention",
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": "v7",
        "versioned_id": "1706.03762v7",
    },
    {
        "paper_key": "moshi",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037",
        "version": "v1",
        "versioned_id": "2410.00037v1",
    },
    {
        "paper_key": "hnet",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955",
        "version": "v2",
        "versioned_id": "2507.07955v2",
    },
]


def _to_num(value: str | None) -> float | int | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def _read_results(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: dict[str, Any] = {}
            for col in RESULT_COLUMNS:
                raw = row.get(col)
                if col in {"first_token_ms", "total_ms"}:
                    out[col] = _to_num(raw)
                elif col in {"retries", "tokens_in", "tokens_out"}:
                    n = _to_num(raw)
                    out[col] = int(n) if n is not None else None
                else:
                    out[col] = (raw or "").strip() if raw is not None else ""
            rows.append(out)
    return rows


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


@router.get("/settings")
def get_settings_info() -> dict[str, Any]:
    settings = get_settings()
    rs_path = settings.results_dir / "retrieval_settings.json"
    retrieval: Any = None
    if rs_path.is_file():
        try:
            retrieval = json.loads(rs_path.read_text(encoding="utf-8"))
        except Exception:
            retrieval = {"error": "unreadable"}
    return {
        "retrieval_settings": retrieval,
        "retrieval_settings_available": retrieval is not None,
        "default_retrieval_setting": "B",
        "provider_modes": {
            "current_default": "standin",
            "supported_by_api": ["standin", "groq"],
            "note": "Provider is selected at server startup; /ask does not switch provider per-request.",
        },
        "results_path": str(settings.results_dir / "results.csv"),
    }


@router.get("/results")
def get_results(
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    status: str | None = None,
    scenario: str | None = None,
    source: str | None = None,
    retrieval_setting: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    rows = _read_results(settings.results_dir / "results.csv")
    # newest first by start_time string (ISO-like) then stable
    rows = list(reversed(rows))

    def match(r: dict[str, Any]) -> bool:
        if status and r.get("status") != status:
            return False
        if scenario and r.get("scenario") != scenario:
            return False
        if source and r.get("source") != source:
            return False
        if retrieval_setting and r.get("retrieval_setting") != retrieval_setting:
            return False
        return True

    filtered = [r for r in rows if match(r)]
    total = len(filtered)
    page = filtered[offset : offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "rows": page,
        "columns": RESULT_COLUMNS,
        "available": (settings.results_dir / "results.csv").is_file(),
    }


@router.get("/summary")
def get_summary() -> dict[str, Any]:
    settings = get_settings()
    rows = _read_results(settings.results_dir / "results.csv")
    total = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    failed = [r for r in rows if r.get("status") in {"error", "timeout", "rate_limited"}]
    latencies = [float(r["total_ms"]) for r in rows if isinstance(r.get("total_ms"), (int, float))]
    ttfts = [float(r["first_token_ms"]) for r in rows if isinstance(r.get("first_token_ms"), (int, float))]
    tokens_in = sum(int(r["tokens_in"]) for r in rows if isinstance(r.get("tokens_in"), int))
    tokens_out = sum(int(r["tokens_out"]) for r in rows if isinstance(r.get("tokens_out"), int))
    recent = list(reversed(rows))[:20]

    by_status: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_scenario: dict[str, int] = {}
    by_setting: dict[str, int] = {}
    for r in rows:
        for key, bucket in [
            ("status", by_status),
            ("source", by_source),
            ("scenario", by_scenario),
            ("retrieval_setting", by_setting),
        ]:
            val = str(r.get(key) or "")
            bucket[val] = bucket.get(val, 0) + 1

    return {
        "available": (settings.results_dir / "results.csv").is_file(),
        "total_requests": total,
        "successful_requests": len(ok),
        "failed_requests": len(failed),
        "status_counts": by_status,
        "source_counts": by_source,
        "scenario_counts": by_scenario,
        "retrieval_setting_counts": by_setting,
        "latency_ms": {
            "avg": (sum(latencies) / len(latencies)) if latencies else None,
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "count": len(latencies),
        },
        "ttft_ms": {
            "avg": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "p50": _percentile(ttfts, 50),
            "p95": _percentile(ttfts, 95),
            "count": len(ttfts),
        },
        "tokens": {
            "in_total": tokens_in if any(isinstance(r.get("tokens_in"), int) for r in rows) else None,
            "out_total": tokens_out if any(isinstance(r.get("tokens_out"), int) for r in rows) else None,
        },
        "recent_requests": recent,
    }


@router.get("/evaluations")
def get_evaluations() -> dict[str, Any]:
    settings = get_settings()
    path = settings.results_dir.parent / "evaluations" / "eval_results.json"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"available": False, "error": str(e), "path": str(path)}
    return {"available": True, "path": str(path), "data": data}


@router.get("/sources")
def get_sources() -> dict[str, Any]:
    settings = get_settings()
    indexes_dir = settings.papers_dir / "indexes"
    configs = {}
    for name, dirname in [("A", "config_a"), ("B", "config_b"), ("C", "config_c")]:
        meta_path = indexes_dir / dirname / "meta.json"
        chunks_path = indexes_dir / dirname / "chunks.jsonl"
        info: dict[str, Any] = {"dir": str(indexes_dir / dirname), "available": meta_path.is_file()}
        if meta_path.is_file():
            try:
                info["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                info["meta_error"] = "unreadable"
        if chunks_path.is_file():
            try:
                # count lines cheaply enough for this corpus
                with chunks_path.open("r", encoding="utf-8") as f:
                    info["chunk_lines"] = sum(1 for _ in f)
            except Exception:
                pass
        configs[name] = info
    return {"papers": PAPERS, "indexes": configs}


@router.get("/events/{request_id}")
def get_event_log(request_id: str, tail: int = Query(200, ge=1, le=2000)) -> dict[str, Any]:
    settings = get_settings()
    # sanitize
    if not request_id or "/" in request_id or "\\" in request_id or ".." in request_id:
        raise HTTPException(status_code=400, detail="invalid request_id")
    path = settings.results_dir / "events" / f"{request_id}.jsonl"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="event log not found")
    events = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    events.append({"raw": line})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if len(events) > tail:
        events = events[-tail:]
    return {"request_id": request_id, "path": str(path), "events": events, "count": len(events)}


def create_observability_router() -> APIRouter:
    return router
