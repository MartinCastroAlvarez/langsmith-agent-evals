# Security

langsmith-agent-evals runs an LLM agent with tools over local data and asks a second model
to grade it. People install it and point it at their own models. Design goals: nothing leaves
the machine unless you opt in, model output and tool input are untrusted, the judge can't be
talked into a score, and reports and traces never become an injection or leak channel.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/langsmith-agent-evals/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory unless
you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The bundled (or your) data, the content of prompts and answers, any API key in
your environment, the integrity of evaluation scores and of the CI gate.

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| Whoever asks the agent | the question text | reach data outside the tools, run code, exhaust the host |
| The agent model | tool calls and arguments, the answer | write the database, read customer names, loop forever |
| Whoever wrote a record (an order note, a policy file) | text that comes back as tool output | turn it into instructions the agent must follow, or into a score |
| The candidate answer (graded by the judge) | text inside the judge prompt | change the rubric or the score |
| Whoever controls `OLLAMA_HOST` or a hosted endpoint | where prompts go | receive your data without your explicit opt-in |

| Threat | Control |
| --- | --- |
| Data sent to LangSmith or another service | Tracing is off; evaluation uses local examples, `upload_results=False`, evaluator tracing disabled and `offline_client()` (sampling rate 0, pre-filled server info, loopback discard URL). The graph runs under `tracing_context(enabled=False)` because LangChain otherwise attaches a tracer that posts runs even in the SDK's "local" mode. Tests block every socket for the whole session. Upload exists only behind `upload --yes-upload` plus your own `LANGSMITH_API_KEY`, and CI checks that it refuses otherwise. |
| SQL injection through tool arguments | Order ids must match `^[A-Z]\d{4}$` before any query; the only statement is a fixed parameterised `SELECT`; the file is opened `mode=ro` with `query_only=ON`. Tested with injection-shaped ids. |
| Data the agent should not see | `lookup_order` returns an allowlist of columns; the customer's name is never returned. |
| Prompt injection in tool output | The v2 prompt says tool results are data. An order note in the bundled data carries an injection; the evaluators (correctness `must_not_include`, policy compliance) catch an agent that obeys it, and the README shows v1 doing so. Treat agent answers about untrusted records as untrusted. |
| Prompt injection against the judge | Candidate, question and context enter the judge prompt only inside `<candidate>`, `<question>`, `<context>` blocks with `& < >` escaped, so a forged closing tag can't leave its block. The judge system prompt is a constant and says block contents are data. Candidates that address the grader ("ignore the rubric", "score override", forged tags) score 0 before the judge is called. Scores are clamped to [0, 1] and limited to {0, 0.5, 1}. Tested with a deliberately gullible fake judge. |
| Position bias in pairwise judging | Every pair is judged in both orders; an inconsistent preference is a tie. |
| HTML injection through the report | Every value (questions, answers, metadata) goes through `html.escape`; the page has no JavaScript and no external assets. Tested with a `<script>` answer. |
| Secrets in traces | `JsonlTracer` truncates long values and redacts keys named like `api_key`, `authorization`, `token`, `secret` or `password`. No key is ever put in a prompt. |
| Runaway agent | `recursion_limit = 2 × LSE_MAX_STEPS + 1` (default 17), questions ≤ 2,000 characters, policy queries ≤ 200, escalation reasons ≤ 300. |
| Data leaving through ollama | `OLLAMA_HOST` must be loopback unless `LSE_ALLOW_REMOTE_OLLAMA=1`. Hosted providers are opt-in and need your own key. |
| Unsafe code in the package | An AST test bans `eval`, `exec`, `compile`, `__import__`, `pickle`, `marshal`, `shelve`, `subprocess`, `yaml`, `requests`, f-string SQL and any `noqa` / `nosec` / `type: ignore` / `shell=True`. |

**Limits.** A judge is a model: a strong enough injection or a weak local judge can still
mis-grade. The escaping and the pre-check make the attack harder, not impossible, which is
why the gate also relies on deterministic evaluators (correctness, trajectory).

## 2. Dependencies

CI runs `bandit` on the package and `pip-audit` on the locked environment. Dependabot alerts,
security updates, secret scanning and private vulnerability reporting are enabled on the
repository.

## 3. Using it on your own agent

- Keep `upload_results=False` and `offline_client()` unless you mean to send data; the
  upload path sends questions, answers, tool outputs and judge reasoning to LangSmith.
- Keep ollama on `127.0.0.1`.
- Treat judge scores as signals. Gate on deterministic evaluators as well.
