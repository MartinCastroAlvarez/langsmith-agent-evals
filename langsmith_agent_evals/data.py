"""Bundled data: the orders database (built from ``orders.json``), the refund-policy markdown and
the golden dataset. ``python -m langsmith_agent_evals.data --check`` verifies the SQLite file."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

__all__ = [
    "DATA_DIR",
    "DATASET_PATH",
    "ORDERS_DB",
    "POLICY_DIR",
    "Dataset",
    "GoldenExample",
    "build_orders_db",
    "load_dataset",
    "main",
]

DATA_DIR = Path(__file__).resolve().parent / "data"
ORDERS_JSON = DATA_DIR / "orders.json"
ORDERS_DB = DATA_DIR / "orders.sqlite"
POLICY_DIR = DATA_DIR / "policy"
DATASET_PATH = DATA_DIR / "dataset.json"

_SCHEMA = """
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    customer TEXT NOT NULL,
    item TEXT NOT NULL,
    category TEXT NOT NULL CHECK (category IN ('physical', 'digital')),
    total_usd REAL NOT NULL,
    status TEXT NOT NULL,
    ordered_on TEXT NOT NULL,
    delivered_on TEXT,
    note TEXT NOT NULL DEFAULT ''
)
"""
_COLUMNS = (
    "id",
    "customer",
    "item",
    "category",
    "total_usd",
    "status",
    "ordered_on",
    "delivered_on",
    "note",
)
_INSERT = "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"


def _orders() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(ORDERS_JSON.read_text(encoding="utf-8"))["orders"]
    return rows


def build_orders_db(target: Path) -> None:
    """Write the orders table to ``target`` (replaced if present)."""
    target.unlink(missing_ok=True)
    with sqlite3.connect(target) as conn:
        conn.execute(_SCHEMA)
        conn.executemany(_INSERT, [tuple(o[c] for c in _COLUMNS) for o in _orders()])
    conn.close()


def _dump(path: Path) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return list(conn.execute("SELECT * FROM orders ORDER BY id"))
    finally:
        conn.close()


@dataclass(frozen=True)
class GoldenExample:
    """One row of the golden set, in LangSmith's ``inputs`` / ``outputs`` shape."""

    id: str
    category: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    as_of: date
    examples: tuple[GoldenExample, ...]


def load_dataset(path: Path = DATASET_PATH) -> Dataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    examples = tuple(
        GoldenExample(
            id=str(e["id"]),
            category=str(e["category"]),
            inputs=dict(e["inputs"]),
            outputs=dict(e["outputs"]),
        )
        for e in raw["examples"]
    )
    ids = [e.id for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate example ids in dataset")
    return Dataset(
        name=str(raw["name"]),
        description=str(raw["description"]),
        as_of=date.fromisoformat(raw["as_of"]),
        examples=examples,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m langsmith_agent_evals.data")
    parser.add_argument("--check", action="store_true", help="fail if orders.sqlite is stale")
    args = parser.parse_args(argv)
    if not args.check:
        build_orders_db(ORDERS_DB)
        print(f"wrote {ORDERS_DB.name}: {len(_orders())} orders")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp) / "orders.sqlite"
        build_orders_db(fresh)
        same = ORDERS_DB.exists() and _dump(fresh) == _dump(ORDERS_DB)
    print("orders.sqlite is up to date" if same else "orders.sqlite is stale: rebuild it")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
