"""The agent's three local tools: order lookup (read-only SQLite), refund-policy search (BM25 over
the bundled markdown) and escalation to a human. Tool input comes from the model and is untrusted:
ids are validated, SQL is parameterised, queries and reasons are length-capped."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool

from langsmith_agent_evals.data import ORDERS_DB
from langsmith_agent_evals.data import POLICY_DIR

__all__ = [
    "ORDER_ID",
    "EscalationDesk",
    "OrderStore",
    "PolicyChunk",
    "PolicyIndex",
    "SupportTools",
    "tokenize",
]

ORDER_ID = re.compile(r"^[A-Z]\d{4}$")
#: Columns the agent may see. ``customer`` is deliberately absent: the agent never needs a name.
_VISIBLE = ("id", "item", "category", "total_usd", "status", "ordered_on", "delivered_on", "note")
_SELECT = (
    "SELECT id, item, category, total_usd, status, ordered_on, delivered_on, note "
    "FROM orders WHERE id = ?"
)
_MAX_QUERY = 200
_MAX_REASON = 300
_WORD = re.compile(r"[a-z0-9$]+")
_STOP_WORDS = (
    "a an and are be can do does for from how i in is it my of on or our the to we what "
    "when where which who will with you your"
)
_STOP = frozenset(_STOP_WORDS.split())


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class OrderStore:
    """Read-only orders: ``mode=ro`` URI, ``query_only``, one fixed parameterised statement."""

    def __init__(self, path: Path = ORDERS_DB, *, today: date | None = None) -> None:
        self._path = path
        self._today = today or date.today()

    def lookup(self, order_id: str) -> dict[str, Any]:
        order_id = order_id.strip().upper()
        if not ORDER_ID.fullmatch(order_id):
            return {"error": "invalid order id: expected one letter and four digits, e.g. A1001"}
        conn = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        try:
            conn.execute("PRAGMA query_only = ON")
            row = conn.execute(_SELECT, (order_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return {"error": f"no order with id {order_id}"}
        order = dict(zip(_VISIBLE, row, strict=True))
        if order["delivered_on"]:
            delivered = date.fromisoformat(order["delivered_on"])
            order["days_since_delivery"] = (self._today - delivered).days
        return order


@dataclass(frozen=True)
class PolicyChunk:
    source: str
    section: str
    text: str

    @property
    def ref(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.section.lower()).strip("-")
        return f"{self.source}#{slug}"


class PolicyIndex:
    """BM25 over ``##`` sections of the policy markdown. Small, exact and dependency-free."""

    def __init__(self, directory: Path = POLICY_DIR, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks: list[PolicyChunk] = []
        for path in sorted(directory.glob("*.md")):
            section, lines = "", list[str]()
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("## "):
                    self._add(path.name, section, lines)
                    section, lines = line[3:].strip(), list[str]()
                elif not line.startswith("# "):
                    lines.append(line)
            self._add(path.name, section, lines)
        self._docs = [Counter(tokenize(f"{c.section} {c.text}")) for c in self.chunks]
        self._avg = sum(sum(d.values()) for d in self._docs) / max(len(self._docs), 1)
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._k1, self._b = k1, b

    def _add(self, source: str, section: str, lines: list[str]) -> None:
        text = " ".join(x.strip() for x in lines if x.strip())
        if section and text:
            self.chunks.append(PolicyChunk(source, section, text))

    def search(self, query: str, k: int = 2) -> list[PolicyChunk]:
        terms = tokenize(query[:_MAX_QUERY])
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            length = sum(doc.values())
            score = 0.0
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    norm = tf + self._k1 * (1 - self._b + self._b * length / self._avg)
                    score += self._idf[t] * tf * (self._k1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [self.chunks[i] for _, i in scored[:k]]


@dataclass
class EscalationDesk:
    """Hands a conversation to a human. Tickets are deterministic so evaluations are repeatable."""

    tickets: list[dict[str, str]] = field(default_factory=list)

    def escalate(self, reason: str, order_id: str = "") -> dict[str, str]:
        reason = " ".join(reason.split())[:_MAX_REASON] or "no reason given"
        digest = hashlib.sha256(f"{order_id}|{reason}".encode()).hexdigest()[:6].upper()
        ticket = {"ticket": f"ESC-{digest}", "status": "queued for a human supervisor"}
        self.tickets.append({**ticket, "reason": reason, "order_id": order_id})
        return ticket


class SupportTools:
    """The three tools as LangChain ``StructuredTool``s, sharing one store, index and desk."""

    def __init__(
        self,
        *,
        orders: OrderStore | None = None,
        policy: PolicyIndex | None = None,
        desk: EscalationDesk | None = None,
    ) -> None:
        self.orders = orders or OrderStore()
        self.policy = policy or PolicyIndex()
        self.desk = desk or EscalationDesk()

    def lookup_order(self, order_id: str) -> str:
        """Look up one order by id (one letter and four digits, e.g. A1001)."""
        return json.dumps(self.orders.lookup(order_id))

    def search_policy(self, query: str) -> str:
        """Search the store's refund, shipping and escalation policy. Returns the top sections."""
        hits = self.policy.search(query)
        return json.dumps([{"ref": c.ref, "text": c.text} for c in hits])

    def escalate(self, reason: str, order_id: str = "") -> str:
        """Hand the conversation to a human supervisor. Returns a ticket number."""
        return json.dumps(self.desk.escalate(reason, order_id))

    def as_tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(self.lookup_order, name="lookup_order"),
            StructuredTool.from_function(self.search_policy, name="search_policy"),
            StructuredTool.from_function(self.escalate, name="escalate"),
        ]
