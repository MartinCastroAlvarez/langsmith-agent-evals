"""The agent under test: a small LangGraph ReAct loop (model <-> tools) for customer support."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from langsmith_agent_evals.prompts import AGENT_PROMPTS

__all__ = ["AgentRun", "build_agent", "run_agent", "summarize_messages", "trajectory_of"]

MAX_QUESTION = 2000


def build_agent(
    model: BaseChatModel, tools: Sequence[BaseTool], prompt: str
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """``START -> agent -> (tools -> agent)* -> END``. ``prompt`` is a version key or full text."""
    system = SystemMessage(AGENT_PROMPTS.get(prompt, prompt))
    bound = model.bind_tools(list(tools))

    def agent(state: MessagesState) -> dict[str, list[BaseMessage]]:
        return {"messages": [bound.invoke([system, *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(name="support-agent")


@dataclass
class AgentRun:
    """What one question produced, in plain JSON-able types."""

    answer: str
    trajectory: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    context: str
    usage: dict[str, int] = field(default_factory=dict)

    def as_outputs(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "trajectory": self.trajectory,
            "tool_calls": self.tool_calls,
            "context": self.context,
            "usage": self.usage,
        }


def trajectory_of(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """OpenAI-format messages for agentevals: assistant turns with their tool calls, tool turns,
    and the final assistant turn. Contents are dropped: trajectory evals compare tool use."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": ""}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                    for c in m.tool_calls
                ]
            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": ""})
    return out


def run_agent(
    graph: CompiledStateGraph[Any, Any, Any, Any],
    question: str,
    *,
    max_steps: int = 8,
    callbacks: Sequence[BaseCallbackHandler] = (),
    tags: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
) -> AgentRun:
    if not question.strip() or len(question) > MAX_QUESTION:
        raise ValueError(f"question must be 1..{MAX_QUESTION} characters")
    config: RunnableConfig = {
        "recursion_limit": 2 * max_steps + 1,
        "callbacks": list(callbacks),
        "tags": list(tags),
        "metadata": dict(metadata or {}),
        "run_name": "support-agent",
    }
    state = graph.invoke({"messages": [HumanMessage(question)]}, config)
    return summarize_messages(state["messages"])


def summarize_messages(messages: Sequence[BaseMessage]) -> AgentRun:
    """Fold a finished conversation into an ``AgentRun`` (answer, trajectory, usage, context)."""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    calls: list[dict[str, Any]] = []
    context: list[str] = []
    for m in messages:
        if isinstance(m, AIMessage):
            meta = m.usage_metadata
            if meta:
                usage["input_tokens"] += meta["input_tokens"]
                usage["output_tokens"] += meta["output_tokens"]
                usage["total_tokens"] += meta["total_tokens"]
            calls.extend({"name": c["name"], "args": c["args"]} for c in m.tool_calls)
        elif isinstance(m, ToolMessage):
            context.append(f"[{m.name}] {m.content}")
    final = messages[-1] if messages else None
    answer = final.text if isinstance(final, AIMessage) else ""
    return AgentRun(
        answer=answer,
        trajectory=trajectory_of([m for m in messages if not isinstance(m, HumanMessage)]),
        tool_calls=calls,
        context="\n".join(context),
        usage=usage,
    )
