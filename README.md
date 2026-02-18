# llm-eval-notes

Public LLM evaluation artifacts. Real tests, real metrics.

Tests hallucination, prompt brittleness, structured output, and tool-use across multiple models.

## What This Tests

- **Hallucination**: Ground truth comparison, detecting claims not supported by context
- **Prompt Brittleness**: Consistency across phrasing variations
- **Structured Output**: JSON schema validation, Pydantic enforcement
- **Tool Use**: Tool selection and argument extraction accuracy

See [EVALS.md](EVALS.md) for detailed evaluation taxonomy.

## Quick Start

```bash
# Install
uv sync

# Run all evals with mock provider (no API keys needed)
uv run llm-eval run all --provider mock

# Run specific eval
uv run llm-eval run hallucination --provider mock
uv run llm-eval run tool-use --provider mock

# Compare models (requires API keys)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
uv run llm-eval compare --models gpt-4o,claude-3-5-sonnet-20241022
```

## Sample Results

Generated with mock provider. For real model results, see `experiments/`.

### Hallucination Tests
| Model | Exact Match | Safe Rate | Hallucination Rate | Refusal Rate |
|-------|-------------|-----------|-------------------|--------------|
| mock/mock-model | 0% | 100% | 0% | 0% |

### Prompt Brittleness
| Model | Consistency Rate | Avg Unique Answers | Refusals |
|-------|-----------------|---------------------|----------|
| mock/mock-model | 100% | 1.0 | 0 |

### Structured Output
| Model | Valid JSON | Schema Valid | Retry Success |
|-------|------------|--------------|---------------|
| mock/mock-model | 0% | 0% | 0% |

### Tool Use
| Model | Tool Selection | Parameter Accuracy | Both Correct |
|-------|---------------|---------------------|--------------|
| mock/mock-model | 0% | 0% | 0% |

## Project Structure

```
llm-eval-notes/
├── src/llm_eval/
│   ├── providers/     # OpenAI, Anthropic, Mock
│   ├── evals/         # Hallucination, Brittleness, Structured, Tool Use
│   └── cli.py         # CLI entry point
├── tests/             # pytest suite (48 tests)
├── experiments/       # Results by date
├── .github/workflows/ # CI
├── EVALS.md           # Evaluation taxonomy
└── pyproject.toml
```

## Adding Evals

1. Add test cases to `src/llm_eval/evals/<type>.py`
2. Add tests to `tests/test_<type>.py`
3. Run `uv run pytest tests/` to verify
4. Update `EVALS.md` with new taxonomy if needed

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/llm_eval/

# List available evals
uv run llm-eval list-evals
```

## Why This Matters

Frontier AI companies care deeply about:

1. **Reliability**: Models that hallucinate less, respond consistently
2. **Tool Use**: Correct tool selection is critical for agents
3. **Structured Output**: APIs need valid JSON, every time
4. **Evaluation**: You can't improve what you don't measure

This repo demonstrates applied AI engineering: systematic evaluation, not hype.

## License

MIT
