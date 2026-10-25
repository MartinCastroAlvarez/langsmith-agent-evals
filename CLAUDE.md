# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`langsmith-agent-evals` shows how to evaluate and observe a LangGraph agent with the current
LangSmith SDK (`evaluate`/`aevaluate`, custom, summary and comparative evaluators, openevals,
agentevals, the pytest plugin, run trees, feedback) entirely offline, as small, correct,
tested code.

## The rules

1. **Offline by default.** Never call LangSmith, a hosted model or any network from code
   paths that run by default, in tests or in CI. Keep every layer of the offline setup
   (local examples, `upload_results=False`, `offline_client`, `tracing_context(enabled=False)`
   around the graph, `disable_evaluator_tracing=True`). Upload stays behind
   `--yes-upload` + the reader's key.
2. **Untrusted text stays in blocks.** Anything from a user, a tool or a candidate enters a
   judge prompt only through `prompts.data_block`. System prompts stay constants.
3. **Tools validate their input.** Fixed parameterised SQL, read-only connection, caps.
4. **Offline tests with fakes** (`FakeSupportModel`, LangChain fake chat models), sockets
   blocked by `tests/conftest.py`.
5. **Honest numbers.** README tables come from `agent-evals run` output in
   `docs/results/`. Fake-model numbers are plumbing checks, never model quality.
6. **The baseline only moves on purpose**, in its own commit, with the reason.
7. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or loosen
   lint/type/coverage config.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`, `mypy` (strict),
`bandit`, `pip-audit`, `python -m langsmith_agent_evals.data --check`, `python -m
examples.quickstart` and `agent-evals --provider fake gate --prompt v2` all green; README and
CHANGELOG updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets or `.env`.
