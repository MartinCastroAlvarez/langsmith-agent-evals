"""Online evaluation, locally: reference-free evaluators over a sample of captured traces.

In LangSmith, an *online evaluator* is a rule on a tracing project: a sample of incoming
production runs is scored by an LLM-as-judge (or code) evaluator and the scores are attached as
feedback. There is no reference answer in production, so only reference-free evaluators apply.
This module does the same over the JSONL traces ``JsonlTracer`` wrote: pick the root runs, keep a
deterministic sample, run the critic and a latency budget, and write feedback lines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from langsmith_agent_evals.evaluators import make_critic

__all__ = ["Conversation", "conversations", "evaluate_online", "sampled"]


@dataclass(frozen=True)
class Conversation:
    trace_id: str
    question: str
    answer: str
    context: str
    latency_ms: float
    tags: tuple[str, ...]


def _messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = (record.get("outputs") or {}).get("messages") or []
    return [m for m in msgs if isinstance(m, dict)]


def conversations(
    traces: Sequence[dict[str, Any]], root_name: str = "support-agent"
) -> list[Conversation]:
    """One ``Conversation`` per root agent run in a ``JsonlTracer`` file."""
    out: list[Conversation] = []
    for rec in traces:
        if rec.get("parent_id") is not None or rec.get("name") != root_name:
            continue
        msgs = _messages(rec)
        question = next((str(m.get("content", "")) for m in msgs if m.get("type") == "human"), "")
        final = msgs[-1] if msgs else {}
        answer = str(final.get("content", "")) if final.get("type") == "ai" else ""
        context = "\n".join(
            f"[{m.get('name')}] {m.get('content', '')}" for m in msgs if m.get("type") == "tool"
        )
        out.append(
            Conversation(
                trace_id=str(rec["trace_id"]),
                question=question,
                answer=answer,
                context=context,
                latency_ms=float(rec.get("latency_ms") or 0.0),
                tags=tuple(rec.get("tags") or ()),
            )
        )
    return out


def sampled(trace_id: str, rate: float) -> bool:
    """Deterministic sampling by trace id, like a sampling rule on a tracing project."""
    if rate >= 1:
        return True
    if rate <= 0:
        return False
    bucket = int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


def evaluate_online(
    traces: Sequence[dict[str, Any]],
    judge: BaseChatModel,
    *,
    rate: float = 1.0,
    latency_budget_ms: float = 5000.0,
    out: Path | None = None,
) -> list[dict[str, Any]]:
    """Score sampled conversations; returns (and optionally writes) LangSmith-shaped feedback."""
    critic = make_critic(judge)
    feedback: list[dict[str, Any]] = []
    for conv in conversations(traces):
        if not sampled(conv.trace_id, rate):
            continue
        inputs = {"question": conv.question}
        outputs = {"answer": conv.answer, "context": conv.context}
        for evaluator in critic:
            result = evaluator(inputs=inputs, outputs=outputs)
            feedback.append({"run_id": conv.trace_id, **result})
        feedback.append(
            {
                "run_id": conv.trace_id,
                "key": "latency_ok",
                "score": float(conv.latency_ms <= latency_budget_ms),
                "comment": f"{conv.latency_ms:.1f} ms (budget {latency_budget_ms:.0f} ms)",
            }
        )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for fb in feedback:
                fh.write(json.dumps(fb) + "\n")
    return feedback
