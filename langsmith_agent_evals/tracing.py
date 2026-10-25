"""Local trace capture: every LangChain/LangGraph run of the agent, written to JSONL.

``JsonlTracer`` is a ``langchain_core`` ``BaseTracer``: it receives the same run tree LangSmith's
own tracer would upload (chain, llm and tool runs with parents, timings, tags, metadata and token
usage) and writes one flattened line per run instead. Nothing leaves the machine. Values are
truncated and only JSON-safe fields are kept, so a trace file never holds keys or raw objects."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

__all__ = ["JsonlTracer", "read_traces"]

_MAX_STR = 2000
_SECRET_KEYS = ("api_key", "apikey", "authorization", "token", "secret", "password")


def _safe(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "...[truncated]"
    if isinstance(value, bool | int | float) or value is None:
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            low = key.lower()
            if any(s in low for s in _SECRET_KEYS) and "tokens" not in low:
                out[key] = "[redacted]"
            else:
                out[key] = _safe(v, depth + 1)
        return out
    if isinstance(value, list | tuple):
        return [_safe(v, depth + 1) for v in value[:50]]
    if hasattr(value, "model_dump"):
        return _safe(value.model_dump(), depth + 1)
    return _safe(str(value), depth + 1)


def _usage(run: Run) -> dict[str, int]:
    for gen_list in (run.outputs or {}).get("generations", []) or []:
        for gen in gen_list:
            msg = gen.get("message") if isinstance(gen, dict) else None
            meta = getattr(msg, "usage_metadata", None)
            if meta is None and isinstance(msg, dict):
                meta = msg.get("kwargs", {}).get("usage_metadata")
            if meta:
                return {
                    k: int(meta.get(k, 0))
                    for k in ("input_tokens", "output_tokens", "total_tokens")
                }
    return {}


class JsonlTracer(BaseTracer):
    """Appends finished runs (root and every child) to ``path`` as JSON lines."""

    def __init__(self, path: Path | str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.count = 0

    def _persist_run(self, run: Run) -> None:
        lines = [json.dumps(rec, default=str) for rec in self._flatten(run, None)]
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")
            self.count += len(lines)

    def _flatten(self, run: Run, trace_id: str | None) -> list[dict[str, Any]]:
        trace = trace_id or str(run.id)
        latency = (
            (run.end_time - run.start_time).total_seconds() * 1000
            if run.end_time and run.start_time
            else None
        )
        record = {
            "trace_id": trace,
            "id": str(run.id),
            "parent_id": str(run.parent_run_id) if run.parent_run_id else None,
            "name": run.name,
            "run_type": run.run_type,
            "start_time": run.start_time.isoformat() if run.start_time else None,
            "latency_ms": round(latency, 3) if latency is not None else None,
            "tags": list(run.tags or []),
            "metadata": _safe({k: v for k, v in (run.extra or {}).get("metadata", {}).items()}),
            "inputs": _safe(run.inputs),
            "outputs": _safe(run.outputs),
            "error": run.error,
            "usage": _usage(run),
        }
        out = [record]
        for child in run.child_runs:
            out.extend(self._flatten(child, trace))
        return out


def read_traces(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
