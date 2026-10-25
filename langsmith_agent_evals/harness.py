"""Run the agent over the golden set with ``langsmith.evaluate`` / ``aevaluate``, fully offline.

The LangSmith SDK runs targets and evaluators locally when it is given **local examples**
(``langsmith.schemas.Example`` objects instead of a dataset name) and ``upload_results=False``.
The client it is handed (``offline_client``) points at a discard address and carries a
pre-filled server info object, so it never needs, nor tries, a connection: the tests block every
socket to prove it. Pairwise comparison is the one SDK path that needs the API
(``evaluate_comparative`` loads both experiments from LangSmith), so ``compare`` runs the same
comparative evaluator locally through ``comparison_evaluator(...).compare_runs``.

One more offline detail: under ``upload_results=False`` the SDK traces in ``"local"`` mode, and
LangChain treats that as "tracing on" and attaches a ``LangChainTracer`` that **posts** child runs
to the client's API URL. The target therefore runs the graph inside
``tracing_context(enabled=False)`` and evaluators run with ``disable_evaluator_tracing=True``;
child runs are captured by the local ``JsonlTracer`` instead."""

from __future__ import annotations

import json
import statistics
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith import Client
from langsmith import aevaluate
from langsmith import evaluate
from langsmith import traceable
from langsmith import tracing_context
from langsmith.evaluation import EvaluationResult
from langsmith.evaluation.evaluator import comparison_evaluator
from langsmith.schemas import Example
from langsmith.schemas import LangSmithInfo
from langsmith.schemas import Run

from langsmith_agent_evals.agent import build_agent
from langsmith_agent_evals.agent import run_agent
from langsmith_agent_evals.agent import summarize_messages
from langsmith_agent_evals.config import Settings
from langsmith_agent_evals.data import Dataset
from langsmith_agent_evals.data import load_dataset
from langsmith_agent_evals.evaluators import TRAJECTORY_MODES
from langsmith_agent_evals.evaluators import Evaluator
from langsmith_agent_evals.evaluators import correctness
from langsmith_agent_evals.evaluators import cost_summary
from langsmith_agent_evals.evaluators import latency_summary
from langsmith_agent_evals.evaluators import make_critic
from langsmith_agent_evals.evaluators import make_pairwise
from langsmith_agent_evals.evaluators import make_trajectory_evaluator
from langsmith_agent_evals.evaluators import token_summary
from langsmith_agent_evals.llm import make_model
from langsmith_agent_evals.tools import OrderStore
from langsmith_agent_evals.tools import SupportTools
from langsmith_agent_evals.tracing import JsonlTracer

__all__ = [
    "OFFLINE_API_URL",
    "Comparison",
    "Experiment",
    "Row",
    "arun_experiment",
    "compare",
    "local_examples",
    "offline_client",
    "row_evaluators",
    "summary_evaluators",
    "run_experiment",
    "write_experiment",
]

#: The discard port on loopback. ``offline_client`` is never asked to send anything; if a code
#: path ever tried, it would fail locally instead of reaching a server.
OFFLINE_API_URL = "http://127.0.0.1:9"
_NAMESPACE = uuid.UUID("6f1d3c52-6a39-4a3e-9d0e-2f6c3f6b9a10")
_BETA = "'upload_results' parameter is in beta"
PASS_KEYS = ("correctness", "policy_compliance")


def offline_client() -> Client:
    """A ``Client`` that ``evaluate(upload_results=False)`` can hold without any network:
    a pre-filled ``info`` skips the ``/info`` probe, a sampling rate of 0 drops every run before
    it is sent (``aevaluate`` in langsmith 0.14 still traces its target with ``enabled=True``
    even when ``upload_results=False``), auto-batch tracing is off so no sender thread starts,
    and the URL is the loopback discard port, so a missed path fails locally."""
    return Client(
        api_url=OFFLINE_API_URL,
        api_key="offline-no-upload",
        info=LangSmithInfo(),
        auto_batch_tracing=False,
        tracing_sampling_rate=0.0,
    )


def local_examples(dataset: Dataset) -> list[Example]:
    """The golden set as in-memory LangSmith ``Example``s with stable ids."""
    dataset_id = uuid.uuid5(_NAMESPACE, dataset.name)
    created = datetime.combine(dataset.as_of, datetime.min.time(), tzinfo=UTC)
    return [
        Example(
            id=uuid.uuid5(dataset_id, e.id),
            dataset_id=dataset_id,
            inputs=e.inputs,
            outputs=e.outputs,
            metadata={"example_id": e.id, "category": e.category},
            created_at=created,
        )
        for e in dataset.examples
    ]


def row_evaluators(judge: BaseChatModel) -> list[Evaluator]:
    return [
        correctness,
        *(make_trajectory_evaluator(m) for m in TRAJECTORY_MODES),
        *make_critic(judge),
    ]


def summary_evaluators(settings: Settings) -> list[Any]:
    """Latency, tokens and cost. Typed ``Any``: the SDK accepts the ``(outputs)`` signature at
    runtime, but its type hints only list ``(runs, examples)``."""
    return [latency_summary, token_summary, cost_summary(settings.price_in, settings.price_out)]


@dataclass
class Row:
    example_id: str
    category: str
    question: str
    answer: str
    scores: dict[str, float]
    comments: dict[str, str]
    latency_ms: float
    usage: dict[str, int]
    tool_calls: list[dict[str, Any]]
    context: str
    run: Run = field(repr=False)
    example: Example = field(repr=False)

    @property
    def passed(self) -> bool:
        return all(self.scores.get(k, 0.0) >= 1.0 for k in PASS_KEYS)

    def as_json(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "category": self.category,
            "question": self.question,
            "answer": self.answer,
            "passed": self.passed,
            "scores": self.scores,
            "comments": self.comments,
            "latency_ms": round(self.latency_ms, 3),
            "usage": self.usage,
            "tool_calls": self.tool_calls,
            "run_id": str(self.run.id),
        }


@dataclass
class Experiment:
    name: str
    prompt: str
    rows: list[Row]
    summary: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def metrics(self) -> dict[str, float]:
        """Mean of every row score, the pass rate, and the summary evaluators' numbers."""
        keys = sorted({k for r in self.rows for k in r.scores})
        out = {k: round(statistics.fmean(r.scores.get(k, 0.0) for r in self.rows), 4) for k in keys}
        out["pass_rate"] = round(sum(r.passed for r in self.rows) / max(len(self.rows), 1), 4)
        out.update(self.summary)
        return out


def _scores(results: Sequence[EvaluationResult]) -> tuple[dict[str, float], dict[str, str]]:
    scores: dict[str, float] = {}
    comments: dict[str, str] = {}
    for r in results:
        scores[r.key] = float(r.score) if r.score is not None else 0.0
        comments[r.key] = r.comment or ""
    return scores, comments


def _collect(prompt: str, results: Any, metadata: dict[str, Any]) -> Experiment:
    rows: list[Row] = []
    for item in results:
        run: Run = item["run"]
        example: Example = item["example"]
        scores, comments = _scores(item["evaluation_results"]["results"])
        out = run.outputs or {}
        latency = (
            (run.end_time - run.start_time).total_seconds() * 1000
            if run.end_time and run.start_time
            else 0.0
        )
        meta = example.metadata or {}
        rows.append(
            Row(
                example_id=str(meta.get("example_id", example.id)),
                category=str(meta.get("category", "")),
                question=str((example.inputs or {}).get("question", "")),
                answer=str(out.get("answer", "")),
                scores=scores,
                comments=comments,
                latency_ms=latency,
                usage=dict(out.get("usage", {})),
                tool_calls=list(out.get("tool_calls", [])),
                context=str(out.get("context", "")),
                run=run,
                example=example,
            )
        )
    summary = {
        r.key: float(r.score) for r in results._summary_results["results"] if r.score is not None
    }
    return Experiment(results.experiment_name, prompt, rows, summary, metadata)


def _setup(
    prompt: str,
    settings: Settings,
    dataset: Dataset | None,
    agent_model: BaseChatModel | None,
    judge: BaseChatModel | None,
    trace_path: Path | None,
) -> tuple[Dataset, Any, BaseChatModel, list[BaseCallbackHandler], dict[str, Any]]:
    dataset = dataset or load_dataset()
    tools = SupportTools(orders=OrderStore(today=dataset.as_of))
    graph = build_agent(agent_model or make_model(settings, "agent"), tools.as_tools(), prompt)
    tracers: list[BaseCallbackHandler] = [JsonlTracer(trace_path)] if trace_path else []
    metadata = {
        "prompt_version": prompt,
        "model": settings.model,
        "judge_model": settings.judge_model,
        "dataset": dataset.name,
    }
    return dataset, graph, judge or make_model(settings, "judge"), tracers, metadata


def run_experiment(
    prompt: str,
    *,
    settings: Settings,
    dataset: Dataset | None = None,
    agent_model: BaseChatModel | None = None,
    judge: BaseChatModel | None = None,
    trace_path: Path | None = None,
) -> Experiment:
    """One experiment: the agent with prompt ``prompt`` over the golden set, every evaluator."""
    dataset, graph, judge, tracers, metadata = _setup(
        prompt, settings, dataset, agent_model, judge, trace_path
    )

    @traceable(name="support-agent", run_type="chain", tags=[f"prompt:{prompt}"], metadata=metadata)
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        with tracing_context(enabled=False):
            return run_agent(
                graph,
                str(inputs["question"]),
                max_steps=settings.max_steps,
                callbacks=tracers,
                tags=[f"prompt:{prompt}"],
                metadata=metadata,
            ).as_outputs()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_BETA)
        results = evaluate(
            target,
            data=local_examples(dataset),
            evaluators=row_evaluators(judge),
            summary_evaluators=summary_evaluators(settings),
            experiment_prefix=f"support-agent-{prompt}",
            metadata=metadata,
            max_concurrency=0,
            upload_results=False,
            disable_evaluator_tracing=True,
            client=offline_client(),
        )
    return _collect(prompt, results, metadata)


async def arun_experiment(
    prompt: str,
    *,
    settings: Settings,
    dataset: Dataset | None = None,
    agent_model: BaseChatModel | None = None,
    judge: BaseChatModel | None = None,
    trace_path: Path | None = None,
    max_concurrency: int = 4,
) -> Experiment:
    """``run_experiment`` on ``aevaluate``: examples run concurrently on the event loop."""
    dataset, graph, judge, tracers, metadata = _setup(
        prompt, settings, dataset, agent_model, judge, trace_path
    )

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        config: RunnableConfig = {
            "recursion_limit": 2 * settings.max_steps + 1,
            "callbacks": tracers,
            "tags": [f"prompt:{prompt}"],
            "metadata": metadata,
        }
        with tracing_context(enabled=False):
            state = await graph.ainvoke(
                {"messages": [HumanMessage(str(inputs["question"]))]}, config
            )
        return summarize_messages(state["messages"]).as_outputs()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_BETA)
        results = await aevaluate(
            target,
            data=local_examples(dataset),
            evaluators=row_evaluators(judge),
            summary_evaluators=summary_evaluators(settings),
            experiment_prefix=f"support-agent-{prompt}-async",
            metadata=metadata,
            max_concurrency=max_concurrency,
            upload_results=False,
            disable_evaluator_tracing=True,
            client=offline_client(),
        )
    rows = [item async for item in results]
    return _collect(prompt, _Materialized(rows, results), metadata)


class _Materialized:
    """An already-consumed ``AsyncExperimentResults`` that ``_collect`` can iterate."""

    def __init__(self, rows: list[Any], source: Any) -> None:
        self._rows = rows
        self.experiment_name = source.experiment_name
        self._summary_results = source._summary_results

    def __iter__(self) -> Any:
        return iter(self._rows)


# --------------------------------------------------------------------------- pairwise
@dataclass
class Comparison:
    a: str
    b: str
    per_example: list[dict[str, Any]]

    @property
    def wins(self) -> dict[str, int]:
        counts = {self.a: 0, self.b: 0, "tie": 0}
        for row in self.per_example:
            counts[row["winner"]] += 1
        return counts

    def as_json(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "wins": self.wins, "per_example": self.per_example}


def compare(a: Experiment, b: Experiment, judge: BaseChatModel) -> Comparison:
    """Pairwise A/B with the comparative evaluator, run locally, example by example."""
    evaluator = comparison_evaluator(make_pairwise(judge))
    by_id = {r.example_id: r for r in b.rows}
    per_example: list[dict[str, Any]] = []
    for row_a in a.rows:
        row_b = by_id.get(row_a.example_id)
        if row_b is None:
            continue
        with tracing_context(enabled=False):
            result = evaluator.compare_runs([row_a.run, row_b.run], row_a.example)
        score_a = float(result.scores.get(row_a.run.id, 0.0) or 0.0)
        score_b = float(result.scores.get(row_b.run.id, 0.0) or 0.0)
        winner = a.prompt if score_a > score_b else b.prompt if score_b > score_a else "tie"
        per_example.append(
            {"example_id": row_a.example_id, "winner": winner, "a": score_a, "b": score_b}
        )
    return Comparison(a.prompt, b.prompt, per_example)


# --------------------------------------------------------------------------- sink
def write_experiment(exp: Experiment, out_dir: Path) -> Path:
    """``<out>/<prompt>.jsonl`` (one row per example) and ``<out>/<prompt>.feedback.jsonl``
    (LangSmith feedback shape: run_id, key, score, comment), returning the results path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp.prompt}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in exp.rows:
            fh.write(json.dumps(row.as_json()) + "\n")
    with (out_dir / f"{exp.prompt}.feedback.jsonl").open("w", encoding="utf-8") as fh:
        for row in exp.rows:
            for key, score in row.scores.items():
                fb = {
                    "run_id": str(row.run.id),
                    "key": key,
                    "score": score,
                    "comment": row.comments.get(key, ""),
                }
                fh.write(json.dumps(fb) + "\n")
    return path
