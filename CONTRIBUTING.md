# Contributing to langsmith-agent-evals

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install                       # add -E anthropic for the optional adapter
poetry run pytest
poetry run ruff check langsmith_agent_evals tests examples
poetry run ruff format --check langsmith_agent_evals tests examples
poetry run mypy langsmith_agent_evals
poetry run bandit -c pyproject.toml -q -r langsmith_agent_evals
poetry run pip-audit
poetry run python -m langsmith_agent_evals.data --check
poetry run python -m langsmith_agent_evals.evals --provider fake gate --prompt v2
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests that call a real model, LangSmith, or any network.
- Turning tracing or upload on by default, or removing a layer of the offline setup
  (`upload_results=False`, `offline_client`, `tracing_context(enabled=False)`,
  `disable_evaluator_tracing=True`).
- Judge prompts that put untrusted text outside an escaped block.
- Hand-edited result tables, or fake-model numbers presented as model quality.
- Loosening the baseline to make a regression pass.

## What it does accept

- New golden examples in `langsmith_agent_evals/data/dataset.json` (question, reference
  answer, `must_include`, `must_not_include`, reference trajectory).
- New evaluators with LangSmith's signatures, with tests for their scoring.
- New injection cases for the agent (in the data) or the judge (in the tests).

## Changing the data

Edit `langsmith_agent_evals/data/orders.json`, then `python -m langsmith_agent_evals.data` to
rebuild the SQLite file. Keep it fictional.

## Updating the baseline and the README numbers

```bash
poetry run python -m langsmith_agent_evals.evals --provider fake baseline --prompt v2   # review the diff
poetry run python -m langsmith_agent_evals.evals --provider fake run --prompts v1 v2 --out runs/docs
cp runs/docs/report.html runs/docs/summary.json docs/results/
```

Paste the printed table into the README with the date and versions, and re-render
`docs/report.png` from `docs/results/report.html`.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
