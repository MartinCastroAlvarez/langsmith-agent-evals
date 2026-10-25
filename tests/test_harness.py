from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import requests
from langsmith import Client
from langsmith.schemas import LangSmithInfo

from langsmith_agent_evals.config import Settings
from langsmith_agent_evals.data import Dataset
from langsmith_agent_evals.fake import FakeSupportModel
from langsmith_agent_evals.harness import OFFLINE_API_URL
from langsmith_agent_evals.harness import Experiment
from langsmith_agent_evals.harness import arun_experiment
from langsmith_agent_evals.harness import compare
from langsmith_agent_evals.harness import local_examples
from langsmith_agent_evals.harness import offline_client
from langsmith_agent_evals.harness import run_experiment
from langsmith_agent_evals.harness import write_experiment
from langsmith_agent_evals.tracing import read_traces

QUALITY = (
    "correctness",
    "trajectory_strict",
    "trajectory_unordered",
    "trajectory_subset",
    "trajectory_superset",
    "helpfulness",
    "groundedness",
    "policy_compliance",
    "pass_rate",
)


def test_v2_scores_full_marks(experiments: dict[str, Experiment]) -> None:
    m = experiments["v2"].metrics()
    assert {k: m[k] for k in QUALITY} == dict.fromkeys(QUALITY, 1.0)
    assert m["cost_usd"] == 0.0 and m["tokens_total"] > 0
    assert 0 <= m["latency_p50_ms"] <= m["latency_p95_ms"]
    assert len(experiments["v2"].rows) == 13


def test_v1_is_measurably_worse(experiments: dict[str, Experiment]) -> None:
    v1, v2 = experiments["v1"].metrics(), experiments["v2"].metrics()
    assert v1["correctness"] < 0.5 < v2["correctness"]
    assert v1["trajectory_superset"] < v2["trajectory_superset"]
    assert v1["trajectory_subset"] == 1.0  # v1 only ever calls a subset of the reference tools
    assert v1["policy_compliance"] < v2["policy_compliance"]
    q13 = next(r for r in experiments["v1"].rows if r.example_id == "q13")
    assert "$500 credit" in q13.answer and q13.scores["policy_compliance"] == 0.0
    assert not q13.passed


def test_rows_carry_sdk_runs_and_examples(experiments: dict[str, Experiment]) -> None:
    row = experiments["v2"].rows[0]
    assert row.run.outputs and row.run.outputs["answer"] == row.answer
    assert row.example.metadata == {"example_id": "q01", "category": "status"}
    assert set(row.as_json()) >= {"example_id", "scores", "comments", "run_id", "passed"}
    assert experiments["v2"].metadata["prompt_version"] == "v2"


def test_local_examples_have_stable_ids(dataset: Dataset) -> None:
    a, b = local_examples(dataset), local_examples(dataset)
    assert [e.id for e in a] == [e.id for e in b]
    assert len({e.id for e in a}) == len(a)


def test_pairwise_v2_beats_v1(experiments: dict[str, Experiment]) -> None:
    result = compare(experiments["v1"], experiments["v2"], FakeSupportModel())
    wins = result.wins
    assert wins["v2"] > wins["v1"] == 0
    assert sum(wins.values()) == 13
    assert result.as_json()["a"] == "v1"
    reverse = compare(experiments["v2"], experiments["v1"], FakeSupportModel())
    assert reverse.wins == wins


def test_write_experiment(tmp_path: Path, experiments: dict[str, Experiment]) -> None:
    path = write_experiment(experiments["v2"], tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 13 and rows[0]["example_id"] == "q01"
    feedback = [json.loads(x) for x in (tmp_path / "v2.feedback.jsonl").read_text().splitlines()]
    assert len(feedback) == 13 * 8
    assert set(feedback[0]) == {"run_id", "key", "score", "comment"}


def test_traces_capture_the_whole_run_tree(
    tmp_path: Path, settings: Settings, dataset: Dataset
) -> None:
    trace = tmp_path / "traces.jsonl"
    small = Dataset(dataset.name, dataset.description, dataset.as_of, dataset.examples[:2])
    run_experiment("v2", settings=settings, dataset=small, trace_path=trace)
    records = read_traces(trace)
    roots = [r for r in records if r["parent_id"] is None]
    assert len(roots) == 2 and {r["name"] for r in roots} == {"support-agent"}
    types = {r["run_type"] for r in records}
    assert {"chain", "llm", "tool"} <= types
    llm = next(r for r in records if r["run_type"] == "llm")
    assert llm["usage"]["total_tokens"] > 0
    assert llm["metadata"]["prompt_version"] == "v2"
    assert "prompt:v2" in roots[0]["tags"]
    tool = next(r for r in records if r["run_type"] == "tool")
    assert tool["name"] == "lookup_order"
    assert read_traces(tmp_path / "missing.jsonl") == []


def test_aevaluate_matches_evaluate(settings: Settings, experiments: dict[str, Experiment]) -> None:
    exp = asyncio.run(arun_experiment("v2", settings=settings, max_concurrency=4))
    sync = experiments["v2"].metrics()
    assert {k: exp.metrics()[k] for k in QUALITY} == {k: sync[k] for k in QUALITY}
    assert exp.name != experiments["v2"].name


def test_offline_client_never_samples() -> None:
    client = offline_client()
    assert client.api_url == OFFLINE_API_URL
    assert client.tracing_sample_rate == 0.0
    assert not client._should_sample("anything")


def test_sdk_pitfall_aevaluate_traces_with_upload_results_false(
    settings: Settings, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """langsmith 0.14: ``aevaluate(upload_results=False)`` still traces its target with
    ``enabled=True``. With an ordinary client those runs head for the API; ``offline_client``
    drops them by sampling. The HTTP layer is stubbed, so nothing is ever sent."""
    sent: list[str] = []

    def fake_request(self: Client, method: str, pathname: str, *args: Any, **kwargs: Any) -> Any:
        sent.append(f"{method} {pathname}")
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        return response

    monkeypatch.setattr(Client, "request_with_retries", fake_request)
    small = Dataset(dataset.name, dataset.description, dataset.as_of, dataset.examples[:1])
    ordinary = Client(
        api_url=OFFLINE_API_URL, api_key="x", info=LangSmithInfo(), auto_batch_tracing=False
    )
    import langsmith_agent_evals.harness as harness

    monkeypatch.setattr(harness, "offline_client", lambda: ordinary)
    asyncio.run(arun_experiment("v2", settings=settings, dataset=small))
    assert any(s.startswith("POST") and "/runs" in s for s in sent), sent
    sent.clear()
    monkeypatch.undo()
    monkeypatch.setattr(Client, "request_with_retries", fake_request)
    asyncio.run(arun_experiment("v2", settings=settings, dataset=small))
    assert sent == []


def test_nothing_reached_langsmith(no_network: list[Any]) -> None:
    assert not [a for a in no_network if "smith.langchain.com" in str(a)]
    assert no_network == []


def test_network_guard_is_active(no_network: list[Any]) -> None:
    import socket

    before = len(no_network)
    with pytest.raises(OSError):
        socket.create_connection(("api.smith.langchain.com", 443), timeout=1)
    assert no_network[before:] == [("getaddrinfo", "api.smith.langchain.com")]
    del no_network[before:]
