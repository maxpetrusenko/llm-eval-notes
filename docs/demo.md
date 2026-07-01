# Demo Runbook

Use this flow to reproduce the README demo without API keys.

## Offline Eval Harness

```bash
uv sync
uv run llm-eval list-evals
uv run llm-eval run all --provider mock
```

Expected behavior:

- CLI lists hallucination, brittleness, structured, tool-use, reasoning, safety, streaming, and all.
- `run all --provider mock` emits a Markdown report.
- The mock report includes both passing and failing signals, which proves the reporting path does not hide regressions.

## Failure Review

Use the failure examples in the README as the review template:

1. Identify the case family and case id.
2. Record expected output, observed output, and metric impact.
3. Assign a root-cause label such as `structured_output.missing_required_field` or `tool_use.argument_extraction`.
4. Decide whether the change blocks release using the README threshold table.

## Real Model Comparison

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
uv run llm-eval compare --models gpt-4o,claude-3-5-sonnet-20241022
```

The comparison command writes Markdown and JSON artifacts under `experiments/` when given an output path. Do not publish raw provider logs or prompts that contain private data.
