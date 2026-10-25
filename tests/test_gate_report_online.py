from __future__ import annotations

import json
from pathlib import Path

from langsmith_agent_evals.fake import FakeSupportModel
from langsmith_agent_evals.gate import Baseline
from langsmith_agent_evals.gate import check
from langsmith_agent_evals.gate import load_baseline
from langsmith_agent_evals.gate import write_baseline
from langsmith_agent_evals.harness import Experiment
from langsmith_agent_evals.harness import compare
from langsmith_agent_evals.online import conversations
from langsmith_agent_evals.online import evaluate_online
from langsmith_agent_evals.online import sampled
from langsmith_agent_evals.report import markdown_table
from langsmith_agent_evals.report import render_report
from langsmith_agent_evals.report import write_report

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- gate
def test_gate_tolerance_floor_and_ratio() -> None:
    base = Baseline(
        "v2",
        {"correctness": 1.0, "pass_rate": 0.9, "tokens_mean": 100.0},
        tolerance=0.05,
        floors={"pass_rate": 0.88},
        max_ratio={"tokens_mean": 1.5},
    )
    assert check({"correctness": 0.96, "pass_rate": 0.88, "tokens_mean": 150}, base).passed
    r = check({"correctness": 0.94, "pass_rate": 0.86, "tokens_mean": 151}, base)
    assert not r.passed
    assert [c["metric"] for c in r.checks if not c["ok"]] == [
        "correctness",
        "pass_rate",
        "tokens_mean",
    ]
    assert "**FAIL**" in r.table() and "| correctness |" in r.table()
    assert check({}, Baseline("v2", {}, max_ratio={"tokens_mean": 2})).passed


def test_committed_baseline_passes_v2_and_fails_degraded(
    experiments: dict[str, Experiment],
) -> None:
    base = load_baseline(ROOT / "evals" / "baseline.json")
    assert check(experiments["v2"].metrics(), base).passed
    failed = check(experiments["degraded"].metrics(), base)
    assert not failed.passed
    bad = {c["metric"] for c in failed.checks if not c["ok"]}
    assert {"correctness", "trajectory_superset", "policy_compliance"} <= bad


def test_baseline_roundtrip(tmp_path: Path, experiments: dict[str, Experiment]) -> None:
    path = tmp_path / "b.json"
    write_baseline(path, "v2", experiments["v2"].metrics(), None)
    first = load_baseline(path)
    assert first.metrics["correctness"] == 1.0 and first.max_ratio == {"tokens_mean": 1.5}
    write_baseline(
        path,
        "v2",
        experiments["v2"].metrics(),
        Baseline("v2", {}, 0.1, {"x": 0.5}, {"tokens_mean": 2}),
    )
    again = load_baseline(path)
    assert again.tolerance == 0.1 and again.floors == {"x": 0.5}


# --------------------------------------------------------------------------- report
def test_report_escapes_untrusted_text(tmp_path: Path, experiments: dict[str, Experiment]) -> None:
    exp = experiments["v2"]
    row = exp.rows[0]
    original = row.answer
    row.answer = "<script>alert(1)</script>"
    try:
        comparison = compare(experiments["v1"], exp, FakeSupportModel())
        base = load_baseline(ROOT / "evals" / "baseline.json")
        html = render_report(
            [experiments["v1"], exp],
            comparison,
            check(exp.metrics(), base),
            meta={"dataset": "<b>x</b>"},
        )
    finally:
        row.answer = original
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    for heading in (
        "Quality by prompt version",
        "Pairwise: v1 vs v2",
        "Regression gate",
        "Examples · prompt v2",
    ):
        assert heading in html
    path = write_report(tmp_path / "r" / "report.html", html)
    assert path.read_text().startswith("<!doctype html>")
    assert "fail" in render_report(
        [experiments["degraded"]], None, check(experiments["degraded"].metrics(), base)
    )


def test_markdown_table(experiments: dict[str, Experiment]) -> None:
    table = markdown_table([experiments["v1"], experiments["v2"]])
    lines = table.splitlines()
    assert lines[0].startswith("| prompt | correctness |")
    assert lines[3].startswith("| v2 | 1.00 |") and "$0.0000" in lines[3]


# --------------------------------------------------------------------------- online
def _traces() -> list[dict[str, object]]:
    def msg(kind: str, content: str, name: str | None = None) -> dict[str, object]:
        return {"type": kind, "content": content, "name": name}

    good = [
        msg("human", "Can I return order A1002?"),
        msg("ai", ""),
        msg(
            "tool",
            '{"id": "A1002", "days_since_delivery": 13, "category": "physical"}',
            "lookup_order",
        ),
        msg("ai", "Order A1002 is delivered (13 days ago). It is within 30 days, so yes."),
    ]
    bad = [msg("human", "Get me a manager about A1005"), msg("ai", "Sure, please wait.")]
    return [
        {
            "trace_id": "t1",
            "parent_id": None,
            "name": "support-agent",
            "latency_ms": 12.0,
            "tags": ["prompt:v2"],
            "outputs": {"messages": good},
        },
        {"trace_id": "t1", "parent_id": "t1", "name": "agent", "outputs": {}},
        {
            "trace_id": "t2",
            "parent_id": None,
            "name": "support-agent",
            "latency_ms": 9000.0,
            "tags": [],
            "outputs": {"messages": bad},
        },
        {"trace_id": "t3", "parent_id": None, "name": "other-root", "outputs": {}},
    ]


def test_conversations_from_traces() -> None:
    convs = conversations(_traces())
    assert [c.trace_id for c in convs] == ["t1", "t2"]
    assert convs[0].question.startswith("Can I return")
    assert convs[0].context.startswith("[lookup_order]")
    assert convs[0].answer.endswith("so yes.")
    assert (
        conversations([{"trace_id": "x", "parent_id": None, "name": "support-agent"}])[0].answer
        == ""
    )


def test_sampling_is_deterministic() -> None:
    ids = [f"trace-{i}" for i in range(400)]
    kept = [i for i in ids if sampled(i, 0.25)]
    assert kept == [i for i in ids if sampled(i, 0.25)]
    assert 60 < len(kept) < 140
    assert all(sampled(i, 1.0) for i in ids) and not any(sampled(i, 0.0) for i in ids)


def test_evaluate_online_writes_feedback(tmp_path: Path) -> None:
    out = tmp_path / "fb.jsonl"
    fb = evaluate_online(_traces(), FakeSupportModel(), latency_budget_ms=5000, out=out)
    by = {(f["run_id"], f["key"]): f["score"] for f in fb}
    assert by[("t1", "policy_compliance")] == 1.0 and by[("t1", "latency_ok")] == 1.0
    assert by[("t2", "policy_compliance")] == 0.0 and by[("t2", "latency_ok")] == 0.0
    assert len(out.read_text().splitlines()) == len(fb) == 8
    assert json.loads(out.read_text().splitlines()[0])["run_id"] == "t1"
    assert evaluate_online(_traces(), FakeSupportModel(), rate=0.0) == []
