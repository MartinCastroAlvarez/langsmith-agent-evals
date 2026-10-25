"""The optional LangSmith upload path. **Never executed by this project or its CI.**

Everything else in the repo runs offline. These functions are what a reader runs, with their own
``LANGSMITH_API_KEY``, to see the same dataset, experiments, pairwise comparison and feedback in
the LangSmith UI. The CLI only reaches them through ``agent-evals upload --yes-upload`` with the
key set; the tests drive them with a stub client and assert the calls, with sockets blocked."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langsmith import Client
from langsmith import evaluate
from langsmith.evaluation import evaluate_comparative

from langsmith_agent_evals.agent import build_agent
from langsmith_agent_evals.agent import run_agent
from langsmith_agent_evals.config import Settings
from langsmith_agent_evals.data import Dataset
from langsmith_agent_evals.evaluators import make_pairwise
from langsmith_agent_evals.harness import row_evaluators
from langsmith_agent_evals.harness import summary_evaluators
from langsmith_agent_evals.tools import OrderStore
from langsmith_agent_evals.tools import SupportTools

__all__ = [
    "UploadRefused",
    "compare_in_langsmith",
    "push_dataset",
    "real_client",
    "upload_experiment",
]


class UploadRefused(RuntimeError):
    """Uploading needs the reader's own key and an explicit opt-in."""


def real_client(env: Mapping[str, str] | None = None) -> Client:
    env = os.environ if env is None else env
    if not env.get("LANGSMITH_API_KEY"):
        raise UploadRefused("set LANGSMITH_API_KEY (your own key) to upload to LangSmith")
    return Client()  # pragma: no cover - would talk to LangSmith


def push_dataset(client: Client, dataset: Dataset) -> str:
    """``create_dataset`` + ``create_examples`` (idempotent on the dataset name)."""
    if client.has_dataset(dataset_name=dataset.name):
        return dataset.name
    created = client.create_dataset(dataset.name, description=dataset.description)
    client.create_examples(
        dataset_id=created.id,
        examples=[
            {
                "inputs": e.inputs,
                "outputs": e.outputs,
                "metadata": {"example_id": e.id, "category": e.category},
            }
            for e in dataset.examples
        ],
    )
    return dataset.name


def upload_experiment(
    client: Client,
    prompt: str,
    *,
    settings: Settings,
    dataset: Dataset,
    agent_model: BaseChatModel,
    judge: BaseChatModel,
) -> str:
    """The same experiment as ``harness.run_experiment``, uploaded: returns its name."""
    tools = SupportTools(orders=OrderStore(today=dataset.as_of))
    graph = build_agent(agent_model, tools.as_tools(), prompt)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_agent(graph, str(inputs["question"]), max_steps=settings.max_steps).as_outputs()

    results = evaluate(
        target,
        data=dataset.name,
        evaluators=row_evaluators(judge),
        summary_evaluators=summary_evaluators(settings),
        experiment_prefix=f"support-agent-{prompt}",
        metadata={"prompt_version": prompt, "model": settings.model},
        client=client,
    )
    return str(results.experiment_name)


def compare_in_langsmith(client: Client, a: str, b: str, judge: BaseChatModel) -> Any:
    """``evaluate_comparative`` over two uploaded experiments, with the same pairwise judge."""
    return evaluate_comparative(
        (a, b),
        evaluators=[make_pairwise(judge)],
        client=client,
        randomize_order=False,  # make_pairwise already asks in both orders
    )
