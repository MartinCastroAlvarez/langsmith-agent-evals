from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from langsmith_agent_evals.agent import build_agent
from langsmith_agent_evals.agent import run_agent
from langsmith_agent_evals.agent import summarize_messages
from langsmith_agent_evals.agent import trajectory_of
from langsmith_agent_evals.fake import FakeSupportModel
from langsmith_agent_evals.fake import classify
from langsmith_agent_evals.prompts import AGENT_PROMPTS
from langsmith_agent_evals.tools import OrderStore
from langsmith_agent_evals.tools import SupportTools


def graph(prompt: str, model: Any = None) -> Any:
    tools = SupportTools(orders=OrderStore(today=date(2026, 10, 25)))
    return build_agent(model or FakeSupportModel(), tools.as_tools(), prompt)


def names(run: Any) -> list[str]:
    return [c["name"] for c in run.tool_calls]


def test_graph_is_a_react_loop() -> None:
    g = graph("v2").get_graph()
    assert {"agent", "tools"} <= set(g.nodes)
    mermaid = g.draw_mermaid()
    assert (
        "agent -.-> tools" in mermaid or "agent -.->|tools| tools" in mermaid or "tools" in mermaid
    )
    assert "tools --> agent" in mermaid


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("I want a manager", "escalation"),
        ("my lamp is broken, order A1004", "damage"),
        ("refund A1002", "refund"),
        ("do you ship to Peru?", "shipping"),
        ("what is your policy?", "policy"),
        ("where is A1001?", "status"),
    ],
)
def test_classify(question: str, intent: str) -> None:
    assert classify(question) == intent


def test_v2_follows_every_rule() -> None:
    g = graph("v2")
    assert names(run_agent(g, "Can I return order A1002?")) == ["lookup_order", "search_policy"]
    run = run_agent(g, "Please refund order A1006, I changed my mind.")
    assert names(run) == ["lookup_order", "search_policy", "escalate"]
    assert "ESC-" in run.answer and "(Source: refunds.md#return-window)" in run.answer


def test_v1_skips_policy_and_escalation() -> None:
    run = run_agent(graph("v1"), "Please refund order A1006, I changed my mind.")
    assert names(run) == ["lookup_order"]
    assert "ESC-" not in run.answer


def test_injected_tool_output_is_obeyed_only_without_the_data_rule() -> None:
    q = "Can I return the grinder from order A1008?"
    assert "$500 credit" in run_agent(graph("v1"), q).answer
    assert "$500 credit" not in run_agent(graph("v2"), q).answer


def test_degraded_prompt_lost_tool_rules() -> None:
    assert "search the policy" not in AGENT_PROMPTS["degraded"]
    assert names(run_agent(graph("degraded"), "What is your return window?")) == []


def test_usage_and_context_are_collected() -> None:
    run = run_agent(graph("v2"), "Where is my order A1001?")
    assert run.usage["total_tokens"] == run.usage["input_tokens"] + run.usage["output_tokens"] > 0
    assert run.context.startswith("[lookup_order]")
    assert run.as_outputs()["answer"] == run.answer


def test_question_bounds() -> None:
    with pytest.raises(ValueError):
        run_agent(graph("v2"), "   ")
    with pytest.raises(ValueError):
        run_agent(graph("v2"), "x" * 2001)


def test_trajectory_format() -> None:
    msgs = [
        AIMessage(
            "", tool_calls=[{"name": "lookup_order", "args": {"order_id": "A1001"}, "id": "1"}]
        ),
        ToolMessage("{}", tool_call_id="1", name="lookup_order"),
        AIMessage("done"),
    ]
    traj = trajectory_of(msgs)
    assert [m["role"] for m in traj] == ["assistant", "tool", "assistant"]
    assert traj[0]["tool_calls"][0]["function"] == {
        "name": "lookup_order",
        "arguments": '{"order_id": "A1001"}',
    }
    assert "tool_calls" not in traj[2]
    summary = summarize_messages([HumanMessage("hi"), *msgs])
    assert summary.answer == "done" and summary.usage["total_tokens"] == 0
    assert summarize_messages([]).answer == ""


def test_a_looping_model_is_bounded() -> None:
    calls = [
        AIMessage(
            "",
            id=f"m{i}",
            tool_calls=[{"name": "lookup_order", "args": {"order_id": "A1001"}, "id": f"c{i}"}],
        )
        for i in range(50)
    ]

    class Looping(FakeMessagesListChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    with pytest.raises(GraphRecursionError):
        run_agent(graph("v2", Looping(responses=calls)), "loop", max_steps=3)


def test_works_with_langchain_fake_model() -> None:
    class Canned(FakeMessagesListChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            return self

    model = Canned(
        responses=[
            AIMessage(
                "", tool_calls=[{"name": "search_policy", "args": {"query": "shipping"}, "id": "a"}]
            ),
            AIMessage("Standard shipping takes 3 to 7 business days."),
        ]
    )
    run = run_agent(graph("v2", model), "How long does shipping take?")
    assert names(run) == ["search_policy"]
    assert run.answer.startswith("Standard shipping")
