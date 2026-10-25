"""A self-contained HTML report (no JavaScript, no external assets). Every value from a run is
passed through ``html.escape``: answers and questions are untrusted text."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from langsmith_agent_evals.gate import GateResult
from langsmith_agent_evals.harness import Comparison
from langsmith_agent_evals.harness import Experiment

__all__ = ["METRIC_COLUMNS", "markdown_table", "render_report", "write_report"]

METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("correctness", "correctness"),
    ("trajectory_strict", "traj. strict"),
    ("trajectory_unordered", "traj. unordered"),
    ("trajectory_subset", "traj. subset"),
    ("trajectory_superset", "traj. superset"),
    ("helpfulness", "helpfulness"),
    ("groundedness", "groundedness"),
    ("policy_compliance", "policy"),
    ("pass_rate", "pass rate"),
)
SUMMARY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("latency_p50_ms", "p50 ms"),
    ("latency_p95_ms", "p95 ms"),
    ("tokens_mean", "tokens / q"),
    ("tool_calls_mean", "tools / q"),
    ("cost_usd", "cost USD"),
)
ROW_KEYS = (
    "correctness",
    "trajectory_strict",
    "trajectory_superset",
    "helpfulness",
    "groundedness",
    "policy_compliance",
)

_CSS = """
:root{--bg:#0b1020;--panel:#121a33;--line:#24304f;--text:#e6e9f2;--muted:#8b95b0;
--accent:#f59e0b;--good:#34d399;--mid:#fbbf24;--bad:#f87171;color-scheme:dark}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f7f7fb;--panel:#fff;
--line:#dde1ec;--text:#141a2e;--muted:#5d6785;--accent:#b45309;--good:#047857;--mid:#a16207;
--bad:#b91c1c;color-scheme:light}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1240px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px;color:var(--accent)}
.sub{color:var(--muted);margin:0 0 18px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:28px;font-weight:700;margin-top:2px}.card .n{color:var(--muted);font-size:12px}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--panel);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:6px 7px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--accent);
vertical-align:middle;margin-left:6px}
.chip{display:inline-block;min-width:34px;text-align:center;border-radius:6px;padding:1px 5px;
font-size:12px;font-weight:600;margin:1px}
.s1{background:color-mix(in srgb,var(--good) 22%,transparent);color:var(--good)}
.s5{background:color-mix(in srgb,var(--mid) 22%,transparent);color:var(--mid)}
.s0{background:color-mix(in srgb,var(--bad) 22%,transparent);color:var(--bad)}
.ans{color:var(--muted);font-size:12px;max-width:520px}.tools{font-size:12px;color:var(--muted)}
.strip{display:flex;flex-wrap:wrap;gap:6px}.pw{background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:6px 10px;font-size:12px}.pw b{display:block;font-size:13px}
.ok{color:var(--good);font-weight:700}.fail{color:var(--bad);font-weight:700}
"""


def _chip(score: float | None) -> str:
    if score is None:
        return '<span class="chip">-</span>'
    cls = "s1" if score >= 0.99 else "s5" if score >= 0.5 else "s0"
    return f'<span class="chip {cls}">{score:.2g}</span>'


def _fmt(key: str, value: float | None) -> str:
    if value is None:
        return "-"
    if key == "cost_usd":
        return f"${value:.4f}"
    if key.endswith("_ms"):
        return f"{value:.1f}"
    if key in ("tokens_mean", "tool_calls_mean"):
        return f"{value:.1f}"
    return f"{value:.2f}"


def markdown_table(experiments: list[Experiment]) -> str:
    """The README results table: one row per prompt version."""
    cols = [*METRIC_COLUMNS, *SUMMARY_COLUMNS]
    head = "| prompt | " + " | ".join(label for _, label in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    rows = []
    for exp in experiments:
        m = exp.metrics()
        rows.append(f"| {exp.prompt} | " + " | ".join(_fmt(k, m.get(k)) for k, _ in cols) + " |")
    return "\n".join([head, sep, *rows])


def render_report(
    experiments: list[Experiment],
    comparison: Comparison | None = None,
    gate: GateResult | None = None,
    *,
    title: str = "Agent evaluation report",
    meta: dict[str, Any] | None = None,
) -> str:
    meta = meta or {}
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main>",
        f"<h1>{escape(title)}</h1>",
        "<p class='sub'>" + escape(" · ".join(f"{k}: {v}" for k, v in meta.items())) + "</p>",
        "<div class='cards'>",
    ]
    for exp in experiments:
        m = exp.metrics()
        parts.append(
            "<div class='card'>"
            f"<div class='k'>prompt {escape(exp.prompt)}</div>"
            f"<div class='v'>{m.get('pass_rate', 0) * 100:.0f}%</div>"
            f"<div class='n'>pass rate · correctness {m.get('correctness', 0):.2f} · "
            f"policy {m.get('policy_compliance', 0):.2f} · "
            f"{m.get('tokens_mean', 0):.0f} tok/q</div>"
            "</div>"
        )
    if comparison is not None:
        w = comparison.wins
        parts.append(
            "<div class='card'><div class='k'>pairwise "
            f"{escape(comparison.a)} vs {escape(comparison.b)}</div>"
            f"<div class='v'>{w[comparison.b]}–{w[comparison.a]}</div>"
            f"<div class='n'>{escape(comparison.b)} wins – {escape(comparison.a)} wins · "
            f"{w['tie']} ties (judged in both orders)</div></div>"
        )
    if gate is not None:
        cls, label = ("ok", "PASS") if gate.passed else ("fail", "FAIL")
        failed = [c["metric"] for c in gate.checks if not c["ok"]]
        parts.append(
            "<div class='card'><div class='k'>regression gate</div>"
            f"<div class='v {cls}'>{label}</div>"
            f"<div class='n'>{escape(', '.join(failed) or 'every metric within tolerance')}</div>"
            "</div>"
        )
    parts.append("</div>")
    for heading, cols in (
        ("Quality by prompt version", METRIC_COLUMNS),
        ("Latency, tokens and cost (summary evaluators)", SUMMARY_COLUMNS),
    ):
        parts.append(f"<h2>{escape(heading)}</h2><div class='wrap'><table><tr><th>prompt</th>")
        parts.extend(f"<th>{escape(label)}</th>" for _, label in cols)
        parts.append("</tr>")
        for exp in experiments:
            m = exp.metrics()
            parts.append(f"<tr><td><b>{escape(exp.prompt)}</b></td>")
            for key, _ in cols:
                value = m.get(key)
                bar = ""
                if cols is METRIC_COLUMNS and value is not None:
                    bar = f"<span class='bar' style='width:{max(2, round(value * 28))}px'></span>"
                parts.append(f"<td class='num'>{escape(_fmt(key, value))}{bar}</td>")
            parts.append("</tr>")
        parts.append("</table></div>")
    if comparison is not None:
        parts.append(
            f"<h2>Pairwise: {escape(comparison.a)} vs {escape(comparison.b)} "
            "(winner per example)</h2><div class='strip'>"
        )
        for pick in comparison.per_example:
            cls = "s5" if pick["winner"] == "tie" else "s1"
            parts.append(
                f"<div class='pw'><b>{escape(pick['example_id'])}</b>"
                f"<span class='chip {cls}'>{escape(pick['winner'])}</span></div>"
            )
        parts.append("</div>")
    for exp in experiments:
        parts.append(
            f"<h2>Examples · prompt {escape(exp.prompt)}</h2><div class='wrap'><table><tr>"
            "<th>id</th><th>question / answer</th><th>tools</th>"
            + "".join(f"<th>{escape(k.replace('trajectory_', 'traj. '))}</th>" for k in ROW_KEYS)
            + "</tr>"
        )
        for row in exp.rows:
            tools = " → ".join(c["name"] for c in row.tool_calls) or "none"
            parts.append(
                f"<tr><td>{escape(row.example_id)}</td><td>{escape(row.question)}"
                f"<div class='ans'>{escape(row.answer)}</div></td>"
                f"<td class='tools'>{escape(tools)}</td>"
                + "".join(f"<td>{_chip(row.scores.get(k))}</td>" for k in ROW_KEYS)
                + "</tr>"
            )
        parts.append("</table></div>")
    if gate is not None:
        parts.append(
            "<h2>Regression gate</h2><div class='wrap'><table><tr><th>metric</th>"
            "<th>baseline</th><th>current</th><th>limit</th><th>status</th></tr>"
        )
        for c in gate.checks:
            status = "<span class='ok'>ok</span>" if c["ok"] else "<span class='fail'>FAIL</span>"
            parts.append(
                f"<tr><td>{escape(c['metric'])}</td><td class='num'>{c['baseline']:.3f}</td>"
                f"<td class='num'>{c['current']:.3f}</td>"
                f"<td class='num'>{escape(c['op'])} {c['limit']:.3f}</td><td>{status}</td></tr>"
            )
        parts.append("</table></div>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_report(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
