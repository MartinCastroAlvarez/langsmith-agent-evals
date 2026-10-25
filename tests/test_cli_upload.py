from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel

from langsmith_agent_evals import cli
from langsmith_agent_evals import evals
from langsmith_agent_evals import upload
from langsmith_agent_evals.config import Settings
from langsmith_agent_evals.data import load_dataset
from langsmith_agent_evals.fake import FakeSupportModel
from langsmith_agent_evals.llm import ModelConfigError
from langsmith_agent_evals.llm import is_loopback_url
from langsmith_agent_evals.llm import make_model

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- config / llm
def test_settings_defaults_and_validation() -> None:
    s = Settings()
    assert (s.provider, s.model, s.judge_provider, s.judge_model) == (
        "ollama",
        "qwen3:4b",
        "ollama",
        "qwen3:4b",
    )
    a = Settings(provider="anthropic")
    assert a.model == "claude-sonnet-5-5" and a.judge_model == "claude-haiku-4-5-20251001"
    for bad in (
        {"provider": "openai"},
        {"judge_provider": "x"},
        {"max_steps": 0},
        {"price_in": -1},
    ):
        with pytest.raises(ValueError):
            Settings(**bad)  # type: ignore[arg-type]


def test_settings_from_env() -> None:
    s = Settings.from_env(
        {
            "LSE_PROVIDER": "fake",
            "LSE_JUDGE_PROVIDER": "ollama",
            "LSE_PRICE_IN": "0.5",
            "LSE_MAX_STEPS": "4",
            "LSE_ALLOW_REMOTE_OLLAMA": "1",
        }
    )
    assert (s.provider, s.judge_provider, s.price_in, s.max_steps, s.allow_remote_ollama) == (
        "fake",
        "ollama",
        0.5,
        4,
        True,
    )
    assert Settings.from_env({}).provider == "ollama"


def test_make_model() -> None:
    assert isinstance(make_model(Settings(provider="fake")), FakeSupportModel)
    ollama = make_model(Settings(provider="ollama"), "judge")
    assert isinstance(ollama, BaseChatModel) and type(ollama).__name__ == "ChatOllama"
    with pytest.raises(ModelConfigError, match="loopback"):
        make_model(Settings(provider="ollama", ollama_host="http://10.0.0.5:11434"))
    assert isinstance(
        make_model(
            Settings(provider="ollama", ollama_host="http://10.0.0.5:1", allow_remote_ollama=True)
        ),
        BaseChatModel,
    )


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        ("http://127.0.0.1:11434", True),
        ("localhost:11434", True),
        ("http://[::1]:1", True),
        ("http://example.com", False),
        ("http://10.1.1.1", False),
    ],
)
def test_is_loopback_url(url: str, ok: bool) -> None:
    assert is_loopback_url(url) is ok


# --------------------------------------------------------------------------- cli
def test_cli_answers(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    trace = tmp_path / "t.jsonl"
    assert (
        cli.main(["--provider", "fake", "-v", "--trace", str(trace), "Where is my order A1001?"])
        == 0
    )
    out = capsys.readouterr().out
    assert "NW-48213" in out and "tool: lookup_order" in out and "usage:" in out
    assert trace.read_text().count("\n") > 3


def test_cli_mermaid_and_errors(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli.main(["--provider", "fake", "--mermaid"]) == 0
    assert "graph TD" in capsys.readouterr().out
    assert cli.main(["--provider", "fake", "x" * 3000]) == 2
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.9:11434")
    assert cli.main(["--provider", "ollama", "hi"]) == 2
    monkeypatch.setenv("LSE_PROVIDER", "fake")
    monkeypatch.delenv("OLLAMA_HOST")
    assert cli.main(["Where is A1001?"]) == 0
    with pytest.raises(SystemExit):
        cli.main(["--provider", "fake"])


# --------------------------------------------------------------------------- evals CLI
def test_evals_run_gate_baseline_online(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "run"
    baseline = tmp_path / "baseline.json"
    shutil.copy(ROOT / "evals" / "baseline.json", baseline)
    assert (
        evals.main(["--provider", "fake", "run", "--out", str(out), "--baseline", str(baseline)])
        == 0
    )
    printed = capsys.readouterr().out
    assert "| v2 | 1.00 |" in printed and "gate (v2" in printed and "PASS" in printed
    for name in (
        "report.html",
        "summary.json",
        "traces.jsonl",
        "v1.jsonl",
        "v2.jsonl",
        "v2.feedback.jsonl",
    ):
        assert (out / name).exists(), name
    summary = json.loads((out / "summary.json").read_text())
    assert summary["gate"]["passed"] is True and summary["pairwise"]["wins"]["v1"] == 0

    assert (
        evals.main(["--provider", "fake", "gate", "--prompt", "v2", "--baseline", str(baseline)])
        == 0
    )
    assert (
        evals.main(
            ["--provider", "fake", "gate", "--prompt", "degraded", "--baseline", str(baseline)]
        )
        == 1
    )
    assert "gate: FAIL (prompt degraded)" in capsys.readouterr().out

    fb = tmp_path / "online.jsonl"
    assert (
        evals.main(
            [
                "--provider",
                "fake",
                "online",
                "--traces",
                str(out / "traces.jsonl"),
                "--out",
                str(fb),
            ]
        )
        == 0
    )
    assert "scored 26 sampled conversations" in capsys.readouterr().out

    fresh = tmp_path / "new" / "b.json"
    assert evals.main(["--provider", "fake", "baseline", "--baseline", str(fresh)]) == 0
    assert json.loads(fresh.read_text())["metrics"]["correctness"] == 1.0


def test_evals_run_async_single_prompt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "a"
    code = evals.main(
        [
            "--provider",
            "fake",
            "run",
            "--async",
            "--prompts",
            "v2",
            "--out",
            str(out),
            "--baseline",
            str(tmp_path / "none.json"),
        ]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["meta"]["runner"] == "langsmith.aevaluate" and summary["pairwise"] is None


def test_evals_errors(tmp_path: Path) -> None:
    assert (
        evals.main(["--provider", "fake", "gate", "--baseline", str(tmp_path / "missing.json")])
        == 2
    )


# --------------------------------------------------------------------------- upload (stubbed)
class StubClient:
    def __init__(self, exists: bool = False) -> None:
        self.exists = exists
        self.calls: list[tuple[str, Any]] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        self.calls.append(("has_dataset", dataset_name))
        return self.exists

    def create_dataset(self, name: str, *, description: str) -> Any:
        self.calls.append(("create_dataset", name))
        return SimpleNamespace(id="ds-1")

    def create_examples(self, *, dataset_id: str, examples: list[dict[str, Any]]) -> None:
        self.calls.append(("create_examples", (dataset_id, len(examples))))


def test_upload_refuses_without_a_key() -> None:
    with pytest.raises(upload.UploadRefused):
        upload.real_client({})


def test_push_dataset_creates_examples_once() -> None:
    ds = load_dataset()
    client = StubClient()
    assert upload.push_dataset(client, ds) == ds.name  # type: ignore[arg-type]
    assert client.calls[1:] == [("create_dataset", ds.name), ("create_examples", ("ds-1", 13))]
    existing = StubClient(exists=True)
    upload.push_dataset(existing, ds)  # type: ignore[arg-type]
    assert [c[0] for c in existing.calls] == ["has_dataset"]


def test_upload_experiment_and_comparison_use_the_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_evaluate(target: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        seen["answer"] = target({"question": "Where is my order A1001?"})["answer"]
        return SimpleNamespace(experiment_name="support-agent-v2-abc")

    def fake_comparative(experiments: Any, **kwargs: Any) -> Any:
        seen["compared"] = experiments
        return "ok"

    monkeypatch.setattr(upload, "evaluate", fake_evaluate)
    monkeypatch.setattr(upload, "evaluate_comparative", fake_comparative)
    s = Settings(provider="fake")
    name = upload.upload_experiment(
        object(),
        "v2",
        settings=s,
        dataset=load_dataset(),  # type: ignore[arg-type]
        agent_model=FakeSupportModel(),
        judge=FakeSupportModel(),
    )
    assert name == "support-agent-v2-abc"
    assert seen["data"] == "support-agent-golden" and "upload_results" not in seen
    assert "NW-48213" in seen["answer"]
    assert upload.compare_in_langsmith(object(), "a", "b", FakeSupportModel()) == "ok"  # type: ignore[arg-type]
    assert seen["compared"] == ("a", "b")


def test_evals_upload_refuses_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    assert evals.main(["--provider", "fake", "upload"]) == 2
    monkeypatch.setenv("LANGSMITH_API_KEY", "not-a-real-key")
    assert evals.main(["--provider", "fake", "upload"]) == 2  # still needs --yes-upload
