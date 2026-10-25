from __future__ import annotations

import json
from pathlib import Path

import pytest

from langsmith_agent_evals import data
from langsmith_agent_evals.data import load_dataset

TOOLS = {"lookup_order", "search_policy", "escalate"}


def test_dataset_shape() -> None:
    ds = load_dataset()
    assert len(ds.examples) == 13
    for ex in ds.examples:
        assert ex.inputs["question"]
        assert ex.outputs["must_include"]
        assert {c["name"] for c in ex.outputs["trajectory"]} <= TOOLS


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    raw = json.loads(data.DATASET_PATH.read_text())
    raw["examples"].append(raw["examples"][0])
    path = tmp_path / "d.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(path)


def test_bundled_db_is_fresh() -> None:
    assert data.main(["--check"]) == 0


def test_build_and_stale_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "orders.sqlite"
    monkeypatch.setattr(data, "ORDERS_DB", target)
    assert data.main(["--check"]) == 1  # missing file is stale
    assert data.main([]) == 0
    assert data.main(["--check"]) == 0
