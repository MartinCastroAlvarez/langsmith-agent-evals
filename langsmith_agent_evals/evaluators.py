"""Evaluators, all with LangSmith's evaluator signatures so ``langsmith.evaluate`` can run them.

* row evaluators ``(inputs, outputs, reference_outputs) -> {"key", "score", "comment"}``:
  correctness, four agentevals trajectory matchers and the three-criterion critic (openevals
  ``create_llm_as_judge`` with a local judge model);
* summary evaluators ``(runs, examples)`` or ``(outputs)`` -> ``{"results": [...]}``:
  latency, tokens, cost;
* a comparative evaluator ``(runs, example) -> {"key", "scores": {run_id: score}}`` for
  pairwise A/B, asked in both orders to cancel position bias."""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Literal

from agentevals.trajectory.match import create_trajectory_match_evaluator
from langchain_core.language_models import BaseChatModel
from langsmith.schemas import Example
from langsmith.schemas import Run
from openevals.llm import create_llm_as_judge
from openevals.types import ChatCompletionMessage
from pydantic import BaseModel
from pydantic import Field

from langsmith_agent_evals.prompts import CRITERIA
from langsmith_agent_evals.prompts import JUDGE_SYSTEM
from langsmith_agent_evals.prompts import PAIRWISE_SYSTEM
from langsmith_agent_evals.prompts import judge_user_message
from langsmith_agent_evals.prompts import pairwise_user_message

__all__ = [
    "TRAJECTORY_MODES",
    "Evaluator",
    "PairwiseVerdict",
    "correctness",
    "cost_summary",
    "latency_summary",
    "looks_like_injection",
    "make_critic",
    "make_pairwise",
    "make_trajectory_evaluator",
    "reference_trajectory",
    "token_summary",
]

Evaluator = Callable[..., dict[str, Any]]
TrajectoryMode = Literal["strict", "unordered", "subset", "superset"]
TRAJECTORY_MODES: tuple[TrajectoryMode, ...] = ("strict", "unordered", "subset", "superset")
#: Free-text arguments are not compared; the order id must match exactly.
_ARG_OVERRIDES: dict[str, Any] = {"search_policy": "ignore", "escalate": "ignore"}
_INJECTION = re.compile(
    r"ignore (?:all |the |any |previous |prior |your )*(?:instructions|rubric|rules|criteri)"
    r"|score override|you are (?:now )?the judge|</?(?:candidate|context|question|answer_[ab])>"
    r"|\b(?:give|assign|output) (?:this|me|a) (?:score|grade)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- correctness
def correctness(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """Share of the reference's required facts present in the answer; 0 if a forbidden one is."""
    answer = str(outputs.get("answer", "")).lower()
    required = [str(x) for x in reference_outputs.get("must_include", [])]
    forbidden = [str(x) for x in reference_outputs.get("must_not_include", [])]
    hit = [x for x in required if x.lower() in answer]
    bad = [x for x in forbidden if x.lower() in answer]
    score = 0.0 if bad else (len(hit) / len(required) if required else 1.0)
    missing = [x for x in required if x not in hit]
    comment = f"forbidden: {bad}" if bad else f"missing: {missing}" if missing else "all facts"
    return {"key": "correctness", "score": score, "comment": comment}


# --------------------------------------------------------------------------- trajectory
def reference_trajectory(calls: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """``[{"name", "args"}, ...]`` -> the OpenAI-format messages a one-call-per-turn agent emits."""
    out: list[dict[str, Any]] = []
    for c in calls:
        fn = {"name": c["name"], "arguments": json.dumps(c.get("args", {}))}
        out += [
            {"role": "assistant", "content": "", "tool_calls": [{"function": fn}]},
            {"role": "tool", "content": ""},
        ]
    out.append({"role": "assistant", "content": ""})
    return out


def make_trajectory_evaluator(mode: TrajectoryMode) -> Evaluator:
    """agentevals' trajectory match in one ``mode``; tool args exact except free-text ones."""
    match = create_trajectory_match_evaluator(
        trajectory_match_mode=mode,
        tool_args_match_mode="exact",
        tool_args_match_overrides=_ARG_OVERRIDES,
    )

    def evaluator(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
        result = match(
            outputs=outputs["trajectory"],
            reference_outputs=reference_trajectory(reference_outputs["trajectory"]),
        )
        got = [c["name"] for c in outputs.get("tool_calls", [])]
        want = [c["name"] for c in reference_outputs["trajectory"]]
        return {
            "key": f"trajectory_{mode}",
            "score": float(bool(result["score"])),
            "comment": f"got {got}, reference {want}",
        }

    evaluator.__name__ = f"trajectory_{mode}"
    return evaluator


# --------------------------------------------------------------------------- critic
def looks_like_injection(text: str) -> bool:
    """Candidate text that talks to the grader (or forges a block tag) is an automatic fail."""
    return bool(_INJECTION.search(text))


def make_critic(judge: BaseChatModel) -> list[Evaluator]:
    """One openevals LLM-as-judge per rubric criterion, sharing the local judge model.

    The candidate, the question and the tool context enter the prompt only through escaped
    ``<candidate>``/``<question>``/``<context>`` blocks; the system prompt is a constant."""
    evaluators: list[Evaluator] = []
    for criterion in CRITERIA:
        evaluators.append(_critic_for(judge, criterion))
    return evaluators


def _critic_for(judge: BaseChatModel, criterion: str) -> Evaluator:
    def build(
        *, inputs: dict[str, Any], outputs: dict[str, Any], **_: Any
    ) -> list[ChatCompletionMessage]:
        user = judge_user_message(
            criterion,
            str(inputs.get("question", "")),
            str(outputs.get("context", "")),
            str(outputs.get("answer", "")),
        )
        return [
            ChatCompletionMessage(role="system", content=JUDGE_SYSTEM),
            ChatCompletionMessage(role="user", content=user),
        ]

    llm_judge = create_llm_as_judge(
        prompt=build, judge=judge, feedback_key=criterion, choices=[0.0, 0.5, 1.0]
    )

    def evaluator(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        answer = str(outputs.get("answer", ""))
        if looks_like_injection(answer):
            return {
                "key": criterion,
                "score": 0.0,
                "comment": "candidate contains instructions aimed at the grader",
            }
        result: Any = llm_judge(inputs=inputs, outputs=outputs)
        score = max(0.0, min(1.0, float(result["score"])))
        return {"key": criterion, "score": score, "comment": str(result.get("comment") or "")}

    evaluator.__name__ = f"critic_{criterion}"
    return evaluator


# --------------------------------------------------------------------------- summaries
def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def latency_summary(runs: Sequence[Run], examples: Sequence[Example]) -> dict[str, Any]:
    """p50 / p95 / mean wall-clock latency of the target runs, in milliseconds."""
    ms = [
        (r.end_time - r.start_time).total_seconds() * 1000
        for r in runs
        if r.end_time is not None and r.start_time is not None
    ]
    return {
        "results": [
            {"key": "latency_p50_ms", "score": round(_pct(ms, 0.5), 3)},
            {"key": "latency_p95_ms", "score": round(_pct(ms, 0.95), 3)},
            {"key": "latency_mean_ms", "score": round(statistics.fmean(ms), 3) if ms else 0.0},
        ]
    }


def token_summary(outputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Total and mean tokens per question, from the agent's ``usage_metadata``."""
    totals = [int(o.get("usage", {}).get("total_tokens", 0)) for o in outputs]
    tool_calls = [len(o.get("tool_calls", [])) for o in outputs]
    return {
        "results": [
            {"key": "tokens_total", "score": sum(totals)},
            {"key": "tokens_mean", "score": round(statistics.fmean(totals), 1) if totals else 0},
            {
                "key": "tool_calls_mean",
                "score": round(statistics.fmean(tool_calls), 2) if tool_calls else 0,
            },
        ]
    }


def cost_summary(price_in: float, price_out: float) -> Evaluator:
    """USD cost at ``price_in`` / ``price_out`` per 1M tokens (0 for a local model)."""

    def cost(outputs: Sequence[dict[str, Any]]) -> dict[str, Any]:
        tin = sum(int(o.get("usage", {}).get("input_tokens", 0)) for o in outputs)
        tout = sum(int(o.get("usage", {}).get("output_tokens", 0)) for o in outputs)
        usd = (tin * price_in + tout * price_out) / 1_000_000
        return {"results": [{"key": "cost_usd", "score": round(usd, 6)}]}

    cost.__name__ = "cost_summary"
    return cost


# --------------------------------------------------------------------------- pairwise
class PairwiseVerdict(BaseModel):
    """Which of two answers is better."""

    reasoning: str = Field(description="One or two sentences.")
    preferred: Literal["A", "B", "tie"]


def make_pairwise(
    judge: BaseChatModel,
) -> Callable[[Sequence[Run], Example | None], dict[str, Any]]:
    """A comparative evaluator: the judge sees (A, B) and then (B, A); only a preference that
    survives the swap counts, anything else is a tie (0.5 each)."""
    structured = judge.with_structured_output(PairwiseVerdict)

    def ask(question: str, first: dict[str, Any], second: dict[str, Any]) -> str:
        user = pairwise_user_message(
            question,
            str(first.get("context", "")),
            str(first.get("answer", "")),
            str(second.get("context", "")),
            str(second.get("answer", "")),
        )
        verdict = structured.invoke(
            [{"role": "system", "content": PAIRWISE_SYSTEM}, {"role": "user", "content": user}]
        )
        if not isinstance(verdict, PairwiseVerdict):
            verdict = PairwiseVerdict.model_validate(verdict)
        return verdict.preferred

    def preference(runs: Sequence[Run], example: Example | None) -> dict[str, Any]:
        if len(runs) != 2:
            raise ValueError("pairwise preference compares exactly two runs")
        a, b = runs
        inputs = (example.inputs if example else a.inputs) or {}
        question = str(inputs.get("question", ""))
        out_a, out_b = a.outputs or {}, b.outputs or {}
        forward = ask(question, out_a, out_b)
        backward = {"A": "B", "B": "A", "tie": "tie"}[ask(question, out_b, out_a)]
        winner = forward if forward == backward else "tie"
        scores = {"A": (1.0, 0.0), "B": (0.0, 1.0), "tie": (0.5, 0.5)}[winner]
        return {"key": "preference", "scores": {a.id: scores[0], b.id: scores[1]}}

    preference.__name__ = "pairwise_preference"
    return preference
