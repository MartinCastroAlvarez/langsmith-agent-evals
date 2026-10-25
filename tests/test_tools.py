from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from langsmith_agent_evals.data import ORDERS_DB
from langsmith_agent_evals.tools import EscalationDesk
from langsmith_agent_evals.tools import OrderStore
from langsmith_agent_evals.tools import PolicyIndex
from langsmith_agent_evals.tools import SupportTools
from langsmith_agent_evals.tools import tokenize

TODAY = date(2026, 10, 25)


def test_lookup_returns_visible_fields_and_days() -> None:
    order = OrderStore(today=TODAY).lookup("a1002")
    assert order["id"] == "A1002"
    assert order["days_since_delivery"] == 13
    assert "customer" not in order  # names are never exposed to the model


def test_undelivered_order_has_no_day_count() -> None:
    assert "days_since_delivery" not in OrderStore(today=TODAY).lookup("A1001")


@pytest.mark.parametrize(
    "bad",
    ["A1001' OR '1'='1", "A1001; DROP TABLE orders", "1001", "AA1001", "", "A10011", "*"],
)
def test_invalid_ids_never_reach_sql(bad: str) -> None:
    assert "invalid order id" in OrderStore(today=TODAY).lookup(bad)["error"]


def test_unknown_id() -> None:
    assert OrderStore().lookup("Z9999") == {"error": "no order with id Z9999"}


def test_database_file_is_opened_read_only() -> None:
    conn = sqlite3.connect(f"file:{ORDERS_DB}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM orders")
    conn.close()


@pytest.mark.parametrize(
    ("query", "ref"),
    [
        ("return window refund", "refunds.md#return-window"),
        ("digital goods refund", "refunds.md#digital-goods"),
        ("damaged item refund", "refunds.md#damaged-or-defective-items"),
        ("How long does standard shipping take?", "shipping.md#delivery-times"),
        ("Do you ship to Australia?", "shipping.md#international-shipping"),
        ("I want a manager", "escalation.md#when-to-escalate"),
    ],
)
def test_policy_search_ranks_the_right_section_first(query: str, ref: str) -> None:
    assert PolicyIndex().search(query)[0].ref == ref


def test_policy_search_no_match_and_caps() -> None:
    index = PolicyIndex()
    assert index.search("zzzz qqqq") == []
    assert len(index.search("refund", k=1)) == 1
    assert index.search("refund " * 500)  # long queries are truncated, not rejected


def test_tokenize_drops_stop_words() -> None:
    assert tokenize("What is the $500 Refund?") == ["$500", "refund"]


def test_escalation_ticket_is_deterministic_and_reason_capped() -> None:
    desk = EscalationDesk()
    a = desk.escalate("x" * 1000, "A1005")
    b = desk.escalate("x" * 1000, "A1005")
    assert a == b
    assert a["ticket"].startswith("ESC-")
    assert len(desk.tickets[0]["reason"]) == 300
    assert desk.escalate("   ")["ticket"].startswith("ESC-")


def test_tools_wrap_as_structured_tools() -> None:
    tools = {t.name: t for t in SupportTools(orders=OrderStore(today=TODAY)).as_tools()}
    assert set(tools) == {"lookup_order", "search_policy", "escalate"}
    assert json.loads(tools["lookup_order"].invoke({"order_id": "A1004"}))["status"] == "delivered"
    hits = json.loads(tools["search_policy"].invoke({"query": "digital"}))
    assert hits[0]["ref"] == "refunds.md#digital-goods"
    ticket = json.loads(tools["escalate"].invoke({"reason": "manager"}))
    assert ticket["status"] == "queued for a human supervisor"
