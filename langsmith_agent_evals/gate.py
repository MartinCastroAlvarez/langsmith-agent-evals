"""Regression gate: an experiment's metrics against a committed baseline.

A metric fails when it drops more than ``tolerance`` below the baseline or below its absolute
floor; cost-like metrics (``max_ratio``) fail when they grow by more than that factor. CI runs
the gate on every push, so a prompt edit that breaks tool use or policy compliance can't merge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

__all__ = ["Baseline", "GateResult", "check", "load_baseline", "write_baseline"]

#: Higher is better for these; they are compared against baseline - tolerance and the floor.
QUALITY_KEYS = (
    "correctness",
    "trajectory_superset",
    "trajectory_strict",
    "helpfulness",
    "groundedness",
    "policy_compliance",
    "pass_rate",
)


@dataclass(frozen=True)
class Baseline:
    prompt: str
    metrics: dict[str, float]
    tolerance: float = 0.05
    floors: dict[str, float] = field(default_factory=dict)
    max_ratio: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GateResult:
    passed: bool
    checks: list[dict[str, Any]]

    def table(self) -> str:
        lines = ["| metric | baseline | current | limit | status |", "|---|---:|---:|---:|---|"]
        for c in self.checks:
            status = "ok" if c["ok"] else "**FAIL**"
            lines.append(
                f"| {c['metric']} | {c['baseline']:.3f} | {c['current']:.3f} | "
                f"{c['op']} {c['limit']:.3f} | {status} |"
            )
        return "\n".join(lines)


def load_baseline(path: Path) -> Baseline:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Baseline(
        prompt=str(raw["prompt"]),
        metrics={k: float(v) for k, v in raw["metrics"].items()},
        tolerance=float(raw.get("tolerance", 0.05)),
        floors={k: float(v) for k, v in raw.get("floors", {}).items()},
        max_ratio={k: float(v) for k, v in raw.get("max_ratio", {}).items()},
    )


def write_baseline(
    path: Path, prompt: str, metrics: dict[str, float], previous: Baseline | None
) -> None:
    keys = [*QUALITY_KEYS, *(previous.max_ratio if previous else {"tokens_mean": 1.5})]
    body = {
        "prompt": prompt,
        "metrics": {k: metrics[k] for k in keys if k in metrics},
        "tolerance": previous.tolerance if previous else 0.05,
        "floors": previous.floors if previous else {"correctness": 0.8, "policy_compliance": 0.8},
        "max_ratio": previous.max_ratio if previous else {"tokens_mean": 1.5},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check(metrics: dict[str, float], baseline: Baseline) -> GateResult:
    checks: list[dict[str, Any]] = []
    for key in QUALITY_KEYS:
        if key not in baseline.metrics:
            continue
        base = baseline.metrics[key]
        limit = max(base - baseline.tolerance, baseline.floors.get(key, 0.0))
        current = float(metrics.get(key, 0.0))
        checks.append(
            {
                "metric": key,
                "baseline": base,
                "current": current,
                "op": ">=",
                "limit": limit,
                "ok": current >= limit - 1e-9,
            }
        )
    for key, ratio in baseline.max_ratio.items():
        if key not in baseline.metrics:
            continue
        base = baseline.metrics[key]
        limit = base * ratio
        current = float(metrics.get(key, 0.0))
        checks.append(
            {
                "metric": key,
                "baseline": base,
                "current": current,
                "op": "<=",
                "limit": limit,
                "ok": current <= limit + 1e-9,
            }
        )
    return GateResult(all(c["ok"] for c in checks), checks)
