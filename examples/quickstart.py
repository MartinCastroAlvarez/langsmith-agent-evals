"""Offline quickstart: two prompt versions through ``langsmith.evaluate``, A/B, gate.

python -m examples.quickstart                      # fake model, no network
LSE_PROVIDER=ollama python -m examples.quickstart  # your local ollama model
"""

from __future__ import annotations

import os
from pathlib import Path

from langsmith_agent_evals.config import Settings
from langsmith_agent_evals.gate import check
from langsmith_agent_evals.gate import load_baseline
from langsmith_agent_evals.harness import compare
from langsmith_agent_evals.harness import run_experiment
from langsmith_agent_evals.llm import make_model
from langsmith_agent_evals.report import markdown_table


def main() -> None:
    provider = os.environ.get("LSE_PROVIDER", "fake")
    settings = Settings.from_env({**os.environ, "LSE_PROVIDER": provider})
    v1 = run_experiment("v1", settings=settings)
    v2 = run_experiment("v2", settings=settings)
    print(markdown_table([v1, v2]))

    ab = compare(v1, v2, make_model(settings, "judge"))
    print(f"\npairwise v1 vs v2 (both orders): {ab.wins}")

    worst = min(v1.rows, key=lambda r: sum(r.scores.values()))
    print(f"\nv1's weakest answer ({worst.example_id}): {worst.answer}")
    print(f"  {worst.comments['correctness']}; policy: {worst.comments['policy_compliance']}")

    baseline = load_baseline(Path(__file__).resolve().parents[1] / "evals" / "baseline.json")
    for exp in (v2, v1):
        verdict = "PASS" if check(exp.metrics(), baseline).passed else "FAIL"
        print(f"\ngate for {exp.prompt}: {verdict}")


if __name__ == "__main__":
    main()
