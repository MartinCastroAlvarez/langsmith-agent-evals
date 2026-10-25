"""``python -m langsmith_agent_evals "<question>"``: ask the support agent one question."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langsmith_agent_evals.agent import build_agent
from langsmith_agent_evals.agent import run_agent
from langsmith_agent_evals.config import PROVIDERS
from langsmith_agent_evals.config import Settings
from langsmith_agent_evals.data import load_dataset
from langsmith_agent_evals.llm import ModelConfigError
from langsmith_agent_evals.llm import make_model
from langsmith_agent_evals.prompts import AGENT_PROMPTS
from langsmith_agent_evals.tools import OrderStore
from langsmith_agent_evals.tools import SupportTools
from langsmith_agent_evals.tracing import JsonlTracer

__all__ = ["main"]


def _settings(provider: str | None) -> Settings:
    base = Settings.from_env()
    if provider is None:
        return base
    return Settings(
        provider=provider,
        judge_provider=provider,
        ollama_host=base.ollama_host,
        allow_remote_ollama=base.allow_remote_ollama,
        temperature=base.temperature,
        max_steps=base.max_steps,
        price_in=base.price_in,
        price_out=base.price_out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m langsmith_agent_evals", description=__doc__)
    parser.add_argument("question", nargs="?", help="the customer's message")
    parser.add_argument("--prompt", default="v2", choices=sorted(AGENT_PROMPTS))
    parser.add_argument("--provider", choices=PROVIDERS, help="default: LSE_PROVIDER or ollama")
    parser.add_argument("--trace", type=Path, help="append the run tree to this JSONL file")
    parser.add_argument("--mermaid", action="store_true", help="print the graph as Mermaid")
    parser.add_argument("-v", "--verbose", action="store_true", help="print tool calls and usage")
    args = parser.parse_args(argv)

    try:
        settings = _settings(args.provider)
        model = make_model(settings, "agent")
    except (ModelConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    dataset = load_dataset()
    graph = build_agent(
        model, SupportTools(orders=OrderStore(today=dataset.as_of)).as_tools(), args.prompt
    )
    if args.mermaid:
        print(graph.get_graph().draw_mermaid())
        return 0
    if not args.question:
        parser.error("a question is required (or --mermaid)")
    tracers = [JsonlTracer(args.trace)] if args.trace else []
    try:
        result = run_agent(graph, args.question, max_steps=settings.max_steps, callbacks=tracers)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.answer)
    if args.verbose:
        for call in result.tool_calls:
            print(f"  tool: {call['name']} {json.dumps(call['args'])}")
        print(f"  usage: {json.dumps(result.usage)}")
    return 0
