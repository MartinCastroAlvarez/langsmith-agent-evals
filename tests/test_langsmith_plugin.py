"""The ``langsmith`` pytest plugin, offline: ``LANGSMITH_TEST_TRACKING=false`` (set in conftest)
runs ``@pytest.mark.langsmith`` tests and ``expect`` assertions locally without sending results.
With your own key and tracking on, the same tests become a LangSmith experiment."""

from __future__ import annotations

from datetime import date

import pytest
from langsmith import expect
from langsmith import testing as t

from langsmith_agent_evals.agent import build_agent
from langsmith_agent_evals.agent import run_agent
from langsmith_agent_evals.data import load_dataset
from langsmith_agent_evals.evaluators import correctness
from langsmith_agent_evals.evaluators import make_critic
from langsmith_agent_evals.evaluators import make_trajectory_evaluator
from langsmith_agent_evals.fake import FakeSupportModel
from langsmith_agent_evals.tools import OrderStore
from langsmith_agent_evals.tools import SupportTools

DATASET = load_dataset()
GRAPH = build_agent(
    FakeSupportModel(), SupportTools(orders=OrderStore(today=date(2026, 10, 25))).as_tools(), "v2"
)
SUPERSET = make_trajectory_evaluator("superset")
CRITIC = make_critic(FakeSupportModel())


@pytest.mark.langsmith
@pytest.mark.parametrize("example", DATASET.examples, ids=[e.id for e in DATASET.examples])
def test_golden_example(example: object) -> None:
    ex = example  # GoldenExample
    t.log_inputs(ex.inputs)  # type: ignore[attr-defined]
    t.log_reference_outputs(ex.outputs)  # type: ignore[attr-defined]
    out = run_agent(GRAPH, ex.inputs["question"]).as_outputs()  # type: ignore[attr-defined]
    t.log_outputs({"answer": out["answer"]})

    score = correctness(out, ex.outputs)["score"]  # type: ignore[attr-defined]
    t.log_feedback(key="correctness", score=score)
    expect.score(score).to_be_greater_than(0.99)
    expect.value(SUPERSET(out, ex.outputs)["score"]).to_equal(1.0)  # type: ignore[attr-defined]
    for critic in CRITIC:
        result = critic(inputs=ex.inputs, outputs=out)  # type: ignore[attr-defined]
        expect.score(result["score"], key=result["key"]).to_be_greater_than(0.99)


@pytest.mark.langsmith
def test_expect_matchers() -> None:
    answer = run_agent(GRAPH, "Do you ship to Australia?").answer
    expect.value(answer).to_contain("European Union")
    expect.score(0.5).to_be_approximately(0.5)
    expect.score(0.5).to_be_between(0.0, 1.0)
    expect.value(None).to_be_none()
    with pytest.raises(AssertionError):
        expect.value(answer).to_contain("Australia is supported")
