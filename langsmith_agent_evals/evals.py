"""``python -m langsmith_agent_evals.evals``: experiments, A/B, the regression gate, reports.

Subcommands (all offline unless stated):

* ``run``      - every prompt version over the golden set, pairwise A/B of the first two, gate
                 the last one, write JSONL results + feedback + traces + ``report.html``.
* ``gate``     - one prompt version against ``evals/baseline.json``; exit 1 on a regression.
* ``baseline`` - rewrite the baseline from a fresh run (review the diff before committing).
* ``online``   - reference-free evaluators over a sample of captured traces.
* ``upload``   - the optional LangSmith path; needs your ``LANGSMITH_API_KEY`` and
                 ``--yes-upload``. Never run by this project."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from langsmith_agent_evals.cli import _settings
from langsmith_agent_evals.config import PROVIDERS
from langsmith_agent_evals.config import Settings
from langsmith_agent_evals.data import load_dataset
from langsmith_agent_evals.gate import check
from langsmith_agent_evals.gate import load_baseline
from langsmith_agent_evals.gate import write_baseline
from langsmith_agent_evals.harness import Experiment
from langsmith_agent_evals.harness import arun_experiment
from langsmith_agent_evals.harness import compare
from langsmith_agent_evals.harness import run_experiment
from langsmith_agent_evals.harness import write_experiment
from langsmith_agent_evals.llm import ModelConfigError
from langsmith_agent_evals.llm import make_model
from langsmith_agent_evals.online import evaluate_online
from langsmith_agent_evals.prompts import AGENT_PROMPTS
from langsmith_agent_evals.report import markdown_table
from langsmith_agent_evals.report import render_report
from langsmith_agent_evals.report import write_report
from langsmith_agent_evals.tracing import read_traces

__all__ = ["main"]

DEFAULT_BASELINE = Path("evals/baseline.json")


def _experiment(prompt: str, settings: Settings, trace: Path | None, use_async: bool) -> Experiment:
    if use_async:
        return asyncio.run(arun_experiment(prompt, settings=settings, trace_path=trace))
    return run_experiment(prompt, settings=settings, trace_path=trace)


def _cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    trace = out / "traces.jsonl"
    trace.unlink(missing_ok=True)
    experiments = [_experiment(p, settings, trace, args.use_async) for p in args.prompts]
    for exp in experiments:
        write_experiment(exp, out)
    comparison = None
    if len(experiments) >= 2:
        comparison = compare(experiments[0], experiments[1], make_model(settings, "judge"))
    gate = None
    if args.baseline.exists():
        gate = check(experiments[-1].metrics(), load_baseline(args.baseline))
    dataset = load_dataset()
    meta = {
        "dataset": f"{dataset.name} ({len(dataset.examples)} examples, as of {dataset.as_of})",
        "agent": f"{settings.provider}:{settings.model}",
        "judge": f"{settings.judge_provider}:{settings.judge_model}",
        "runner": "langsmith.aevaluate" if args.use_async else "langsmith.evaluate",
        "upload_results": "False",
    }
    html = render_report(experiments, comparison, gate, meta=meta)
    write_report(out / "report.html", html)
    summary = {
        "meta": meta,
        "experiments": {e.prompt: {"name": e.name, "metrics": e.metrics()} for e in experiments},
        "pairwise": comparison.as_json() if comparison else None,
        "gate": {"passed": gate.passed, "checks": gate.checks} if gate else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(markdown_table(experiments))
    if comparison:
        w = comparison.wins
        print(f"\npairwise {comparison.a} vs {comparison.b}: {w}")
    if gate:
        print(
            f"\ngate ({experiments[-1].prompt} vs {args.baseline}): "
            f"{'PASS' if gate.passed else 'FAIL'}"
        )
    print(f"\nwrote {out}/report.html, summary.json, traces.jsonl and per-prompt JSONL")
    return 0


def _cmd_gate(args: argparse.Namespace, settings: Settings) -> int:
    exp = _experiment(args.prompt, settings, None, False)
    result = check(exp.metrics(), load_baseline(args.baseline))
    print(result.table())
    print(f"\ngate: {'PASS' if result.passed else 'FAIL'} (prompt {args.prompt})")
    return 0 if result.passed else 1


def _cmd_baseline(args: argparse.Namespace, settings: Settings) -> int:
    exp = _experiment(args.prompt, settings, None, False)
    previous = load_baseline(args.baseline) if args.baseline.exists() else None
    write_baseline(args.baseline, args.prompt, exp.metrics(), previous)
    print(f"wrote {args.baseline} from prompt {args.prompt}")
    return 0


def _cmd_online(args: argparse.Namespace, settings: Settings) -> int:
    feedback = evaluate_online(
        read_traces(args.traces),
        make_model(settings, "judge"),
        rate=args.rate,
        latency_budget_ms=args.latency_budget_ms,
        out=args.out,
    )
    runs = {f["run_id"] for f in feedback}
    by_key: dict[str, list[float]] = {}
    for f in feedback:
        by_key.setdefault(f["key"], []).append(float(f["score"]))
    print(f"scored {len(runs)} sampled conversations -> {args.out}")
    for key, scores in sorted(by_key.items()):
        print(f"  {key}: {sum(scores) / len(scores):.2f}")
    return 0


def _cmd_upload(args: argparse.Namespace, settings: Settings) -> int:  # pragma: no cover
    from langsmith_agent_evals.upload import compare_in_langsmith
    from langsmith_agent_evals.upload import push_dataset
    from langsmith_agent_evals.upload import real_client
    from langsmith_agent_evals.upload import upload_experiment

    if not args.yes_upload or not os.environ.get("LANGSMITH_API_KEY"):
        print(
            "refusing to upload: pass --yes-upload and set your own LANGSMITH_API_KEY",
            file=sys.stderr,
        )
        return 2
    client = real_client()
    dataset = load_dataset()
    push_dataset(client, dataset)
    names = [
        upload_experiment(
            client,
            prompt,
            settings=settings,
            dataset=dataset,
            agent_model=make_model(settings, "agent"),
            judge=make_model(settings, "judge"),
        )
        for prompt in args.prompts
    ]
    if len(names) >= 2:
        compare_in_langsmith(client, names[0], names[1], make_model(settings, "judge"))
    print("uploaded experiments:", ", ".join(names))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m langsmith_agent_evals.evals",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", choices=PROVIDERS, help="default: LSE_PROVIDER or ollama")
    sub = parser.add_subparsers(dest="cmd", required=True)
    prompts = sorted(AGENT_PROMPTS)

    run = sub.add_parser("run", help="experiments + A/B + gate + report")
    run.add_argument("--prompts", nargs="+", default=["v1", "v2"], choices=prompts)
    run.add_argument("--out", type=Path, default=Path("runs/latest"))
    run.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    run.add_argument("--async", dest="use_async", action="store_true", help="use aevaluate")

    gate = sub.add_parser("gate", help="fail on a regression against the baseline")
    gate.add_argument("--prompt", default="v2", choices=prompts)
    gate.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)

    base = sub.add_parser("baseline", help="rewrite the baseline from a fresh run")
    base.add_argument("--prompt", default="v2", choices=prompts)
    base.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)

    online = sub.add_parser("online", help="reference-free evaluators over captured traces")
    online.add_argument("--traces", type=Path, default=Path("runs/latest/traces.jsonl"))
    online.add_argument("--rate", type=float, default=1.0, help="sampling rate 0..1")
    online.add_argument("--latency-budget-ms", type=float, default=5000.0)
    online.add_argument("--out", type=Path, default=Path("runs/latest/online_feedback.jsonl"))

    up = sub.add_parser("upload", help="(optional) upload to LangSmith with your own key")
    up.add_argument("--prompts", nargs="+", default=["v1", "v2"], choices=prompts)
    up.add_argument("--yes-upload", action="store_true", help="confirm sending data to LangSmith")

    args = parser.parse_args(argv)
    try:
        settings = _settings(args.provider)
        handlers = {
            "run": _cmd_run,
            "gate": _cmd_gate,
            "baseline": _cmd_baseline,
            "online": _cmd_online,
            "upload": _cmd_upload,
        }
        return handlers[args.cmd](args, settings)
    except (ModelConfigError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
