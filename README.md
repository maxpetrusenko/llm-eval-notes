# llm-eval-notes

Public LLM evaluation artifacts. Real tests, real metrics.

Tests hallucination, prompt brittleness, and structured output across multiple models.

## What This Tests

- **Hallucination**: Ground truth comparison, detecting claims not supported by context
- **Prompt Brittleness**: Consistency across phrasing variations
- **Structured Output**: JSON schema validation, Pydantic enforcement

## Quick Start

```bash
# Install
uv sync

# Run all evals with mock provider (no API keys needed)
uv run llm-eval run all --provider mock

# Run specific eval
uv run llm-eval run hallucination --provider mock

# Compare models (requires API keys)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
uv run llm-eval compare
```

## Sample Results

### Hallucination Tests
| Model | Exact Match | Safe Rate | Hallucination Rate |
|-------|-------------|-----------|-------------------|
| Mock (baseline) | 60% | 80% | 20% |

### Prompt Brittleness
| Model | Consistency Rate | Avg Unique Answers |
|-------|-----------------|-------------------|
| Mock (baseline) | 85% | 1.3 |

### Structured Output
| Model | Valid JSON | Schema Valid |
|-------|------------|--------------|
| Mock (baseline) | 83% | 67% |

## Project Structure

```
llm-eval-notes/
├── src/llm_eval/
│   ├── providers/     # OpenAI, Anthropic, Mock
│   ├── evals/         # Hallucination, Brittleness, Structured
│   └── cli.py         # CLI entry point
├── tests/             # pytest suite
├── experiments/       # Configs and results
└── pyproject.toml
```

## Adding Evals

1. Add test cases to `src/llm_eval/evals/<type>.py`
2. Run `uv run pytest tests/` to verify
3. Update results in README

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## License

MIT
