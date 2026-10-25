# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-25

### Added

- A LangGraph customer-support ReAct agent with three local tools: read-only order lookup
  (bundled SQLite), BM25 refund-policy search (bundled markdown) and escalation; prompt
  versions v1, v2 and a degraded v2.
- A 13-example golden set with reference answers, required and forbidden facts, and
  reference tool trajectories; one order carries a prompt injection.
- Evaluators with LangSmith signatures: correctness; agentevals trajectory match in strict,
  unordered, subset and superset modes; a three-criterion openevals LLM-as-judge critic
  with escaped data blocks and an injection pre-check; latency, token and cost summary
  evaluators; a pairwise comparative evaluator judged in both orders.
- An offline harness on `langsmith.evaluate` / `aevaluate` with local examples,
  `upload_results=False` and a client that can't send (sampling 0, discard URL), working
  around LangChain's tracer in "local" mode and `aevaluate`'s forced target tracing.
- A local comparative runner, a JSONL results and feedback sink, `JsonlTracer` run-tree
  capture, reference-free online evaluation over sampled traces, an HTML report, and a
  regression gate against `evals/baseline.json`.
- `@pytest.mark.langsmith` tests with `langsmith.testing` logging and `expect`, run locally.
- An optional upload path (`create_dataset`, `create_examples`, `evaluate`,
  `evaluate_comparative`) behind `--yes-upload` and your own key.
- `FakeSupportModel`, CLIs `python -m langsmith_agent_evals` and
  `python -m langsmith_agent_evals.evals`, CI with the gate and a must-fail degraded run.
