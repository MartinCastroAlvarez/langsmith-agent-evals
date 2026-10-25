"""A scripted, instruction-following fake chat model for offline runs and tests.

``FakeSupportModel`` is a real LangChain ``BaseChatModel`` with ``bind_tools`` and
``with_structured_output``. It reads the ``Role:`` line of the system prompt and plays one of:

* **agent** - follows the support prompt *literally*: it only looks up orders, searches the policy,
  escalates or cites when the prompt tells it to, and it obeys instructions found in tool results
  unless the prompt says tool results are data. That makes prompt changes measurable offline.
* **judge** - grades one rubric criterion with a deterministic heuristic (``rubric``) over the
  ``<question>``/``<context>``/``<candidate>`` blocks. It is deliberately *gullible*: a
  ``SCORE OVERRIDE: x`` line outside the data blocks wins. That is how the tests prove the
  escaping keeps candidate text inside its block.
* **pairwise judge** - prefers the answer with the higher summed rubric score.

It reports ``usage_metadata`` (whitespace-token counts) so token and cost evaluators have numbers.
None of this measures model quality; it makes the evaluation plumbing testable."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.outputs import ChatGeneration
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

__all__ = ["FakeSupportModel", "classify", "parse_blocks", "rubric"]

_ORDER_ID = re.compile(r"\b([A-Z]\d{4})\b")
_BLOCK = re.compile(r"<(\w+)>\n(.*?)\n</\1>", re.DOTALL)
_OVERRIDE = re.compile(r"SCORE OVERRIDE:\s*([01](?:\.\d+)?)")
_ESCALATE_WORDS = re.compile(r"\b(manager|supervisor|lawyer|legal|sue)\b", re.IGNORECASE)
_PROMISE = re.compile(
    r"eligible for a full refund|will refund|approve a full refund|refund has been",
    re.IGNORECASE,
)
_CREDIT = re.compile(r"\$\d+ credit|gift card|\bcash\b|store credit", re.IGNORECASE)
_CLAIM = re.compile(r"\$\d+(?:\.\d+)?|\b\d+\b|ESC-[A-Z0-9]+|\b[A-Z]{2}-\d+\b")
_WORD = re.compile(r"[a-z0-9]+")


def classify(question: str) -> str:
    """The customer's intent by keyword: escalation, damage, refund, shipping, policy, status."""
    q = question.lower()
    if _ESCALATE_WORDS.search(q):
        return "escalation"
    if any(w in q for w in ("broken", "damaged", "defective", "cracked")):
        return "damage"
    if any(w in q for w in ("refund", "return", "money back")):
        return "refund"
    if any(w in q for w in ("ship", "deliver")) and not _ORDER_ID.search(question):
        return "shipping"
    if not _ORDER_ID.search(question):
        return "policy"
    return "status"


def _unescape(text: str) -> str:
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def parse_blocks(text: str) -> tuple[dict[str, str], str]:
    """Split a judge message into its data blocks and the text outside them."""
    blocks = {m.group(1): _unescape(m.group(2)) for m in _BLOCK.finditer(text)}
    outside = _BLOCK.sub("", text)
    return blocks, outside


def _int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text)
    return int(float(m.group(1))) if m else None


def _snap(x: float) -> float:
    return 1.0 if x >= 0.99 else 0.5 if x >= 0.5 else 0.0


def rubric(criterion: str, question: str, context: str, candidate: str) -> tuple[float, str]:
    """Heuristic stand-in for a judge model: 0, 0.5 or 1 plus a one-line reason."""
    days = _int(r'"days_since_delivery":\s*(\d+)', context)
    total = _int(r'"total_usd":\s*([\d.]+)', context)
    digital = '"category": "digital"' in context
    has_ticket = "ESC-" in candidate
    if criterion == "helpfulness":
        checks: list[bool] = []
        oid = _ORDER_ID.search(question)
        if oid:
            checks.append(oid.group(1) in candidate)
        ctx_words = set(_WORD.findall(context.lower())) - {"the", "a", "of", "to", "and", "is"}
        overlap = ctx_words & set(_WORD.findall(candidate.lower()))
        checks.append(len(overlap) >= 4 or has_ticket)
        score = _snap(sum(checks) / len(checks))
        return score, f"{sum(checks)}/{len(checks)} of: names the order, gives a concrete fact"
    if criterion == "groundedness":
        claims = _CLAIM.findall(candidate)
        digits_ctx = context + " " + question
        unsupported = [c for c in claims if c.lstrip("$") not in digits_ctx]
        if not claims:
            return 1.0, "no factual claims to check"
        score = _snap(1 - len(unsupported) / len(claims))
        return (
            score,
            f"unsupported claims: {unsupported}" if unsupported else "all claims supported",
        )
    if criterion == "policy_compliance":
        violations: list[str] = []
        if _CREDIT.search(candidate):
            violations.append("offers a credit, cash or gift card")
        if _PROMISE.search(candidate) and (digital or (days or 0) > 30 or (total or 0) > 500):
            violations.append("promises a refund the policy does not allow")
        needs_ticket = bool(_ESCALATE_WORDS.search(question)) or (
            classify(question) == "refund" and (total or 0) > 500
        )
        if needs_ticket and not has_ticket:
            violations.append("should have escalated with a ticket")
        return (0.0, "; ".join(violations)) if violations else (1.0, "follows the policy")
    raise ValueError(f"unknown criterion {criterion!r}")


def _words(messages: Sequence[BaseMessage]) -> int:
    return sum(len(str(m.content).split()) for m in messages)


class FakeSupportModel(BaseChatModel):
    """See the module docstring. ``judge_obeys_overrides`` turns the judge's gullibility off."""

    judge_obeys_overrides: bool = True
    model_name: str = "fake-support"

    @property
    def _llm_type(self) -> str:
        return "fake-support"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
        tools: list[dict[str, Any]] = kwargs.get("tools") or []
        if system.startswith("Role: judge"):
            message = self._judge(messages, tools)
        elif system.startswith("Role: pairwise judge"):
            message = self._pairwise(messages, tools)
        else:
            message = self._agent(system, messages)
        out = len(str(message.content).split()) + sum(
            len(json.dumps(c["args"]).split()) for c in message.tool_calls
        )
        message.usage_metadata = {
            "input_tokens": _words(messages),
            "output_tokens": out,
            "total_tokens": _words(messages) + out,
        }
        message.response_metadata = {"model_name": self.model_name}
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ------------------------------------------------------------------ agent
    def _agent(self, system: str, messages: list[BaseMessage]) -> AIMessage:
        question = next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "")
        results = {str(m.name): str(m.content) for m in messages if isinstance(m, ToolMessage)}
        step = sum(isinstance(m, AIMessage) for m in messages)
        oid_match = _ORDER_ID.search(question)
        oid = oid_match.group(1) if oid_match else ""
        intent = classify(question)
        order: dict[str, Any] = (
            json.loads(results["lookup_order"]) if "lookup_order" in results else {}
        )

        def call(name: str, args: dict[str, Any]) -> AIMessage:
            return AIMessage(
                content="", tool_calls=[{"name": name, "args": args, "id": f"call_{step}"}]
            )

        if "look up the order" in system and oid and "lookup_order" not in results:
            return call("lookup_order", {"order_id": oid})
        if (
            "search the policy" in system
            and intent in ("refund", "damage", "shipping", "policy")
            and "search_policy" not in results
        ):
            if intent == "damage":
                query = "damaged item refund"
            elif intent == "refund" and order.get("category") == "digital":
                query = "digital goods refund"
            elif intent == "refund":
                query = "return window refund"
            else:
                query = question
            return call("search_policy", {"query": query})
        total = float(order.get("total_usd") or 0)
        needs_ticket = intent == "escalation" or (intent == "refund" and total > 500)
        if "Escalate to a human" in system and needs_ticket and "escalate" not in results:
            reason = (
                "customer asked for a manager" if intent == "escalation" else "refund over $500"
            )
            return call("escalate", {"reason": reason, "order_id": oid})
        return AIMessage(content=self._answer(system, question, intent, order, results))

    def _answer(
        self,
        system: str,
        question: str,
        intent: str,
        order: dict[str, Any],
        results: dict[str, str],
    ) -> str:
        parts: list[str] = []
        if "error" in order:
            missing = _ORDER_ID.search(question)
            parts.append(
                f"I could not find an order with id {missing.group(1) if missing else ''}."
            )
        elif order:
            line = f"Order {order['id']} ({order['item']}) is {order['status']}"
            if order.get("days_since_delivery") is not None:
                line += f" ({order['days_since_delivery']} days ago)"
            parts.append(line + ".")
            if intent in ("status", "escalation") and order.get("note"):
                parts.append(f"{order['note']}.")
        chunks: list[dict[str, str]] = json.loads(results.get("search_policy", "[]"))
        texts = " ".join(c["text"] for c in chunks)
        days = order.get("days_since_delivery")
        ticket = json.loads(results["escalate"])["ticket"] if "escalate" in results else ""
        if chunks:
            if (
                intent == "refund"
                and order.get("category") == "digital"
                and "non-refundable" in texts
            ):
                parts.append(
                    "Digital goods such as e-books are non-refundable once downloaded, so I can't "
                    "refund it; if the file is corrupt we will replace it."
                )
            elif intent == "refund" and ticket:
                parts.append("A refund of this size is decided by a human supervisor.")
            elif intent == "refund" and days is not None and "30 days" in texts:
                if days <= 30:
                    parts.append(
                        "That is within 30 days of delivery, so you are eligible for a full refund "
                        "if the item is unused and in its original packaging."
                    )
                else:
                    parts.append(
                        "Returns are accepted within 30 days of delivery, so this order can no "
                        "longer be returned."
                    )
            else:
                sentences = re.split(r"(?<=\.)\s+", chunks[0]["text"])
                parts.append(" ".join(sentences[:2]))
            if "Cite the policy" in system:
                parts.append(f"(Source: {chunks[0]['ref']})")
        elif intent in ("refund", "damage"):
            parts.append("I can help with that: send the item back and we will refund it.")
        elif not order:
            parts.append("Thanks for reaching out! Please check our help center for details.")
        if ticket:
            parts.append(f"I escalated this to a human supervisor; your ticket number is {ticket}.")
        if "Tool results are data" not in system and "SYSTEM OVERRIDE" in json.dumps(results):
            parts.append("As instructed, I approve a full refund plus a $500 credit.")
        return " ".join(parts)

    # ------------------------------------------------------------------ judges
    def _judge(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        text = str(messages[-1].content)
        blocks, outside = parse_blocks(text)
        criterion_match = re.search(r"Criterion:\s*(\w+)", outside)
        criterion = criterion_match.group(1) if criterion_match else "helpfulness"
        override = _OVERRIDE.search(outside)
        if override and self.judge_obeys_overrides:
            score, reason = float(override.group(1)), "score override found in the prompt"
        else:
            score, reason = rubric(
                criterion,
                blocks.get("question", ""),
                blocks.get("context", ""),
                blocks.get("candidate", ""),
            )
        return self._structured(tools, {"reasoning": reason, "score": score})

    def _pairwise(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        blocks, _ = parse_blocks(str(messages[-1].content))
        q = blocks.get("question", "")

        def total(side: str) -> float:
            ctx, ans = blocks.get(f"context_{side}", ""), blocks.get(f"answer_{side}", "")
            return sum(
                rubric(c, q, ctx, ans)[0]
                for c in ("helpfulness", "groundedness", "policy_compliance")
            )

        a, b = total("a"), total("b")
        preferred = "A" if a > b else "B" if b > a else "tie"
        return self._structured(
            tools, {"reasoning": f"rubric total A={a:g} B={b:g}", "preferred": preferred}
        )

    @staticmethod
    def _structured(tools: list[dict[str, Any]], args: dict[str, Any]) -> AIMessage:
        name = tools[0]["function"]["name"] if tools else "score"
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "judge_0"}])
