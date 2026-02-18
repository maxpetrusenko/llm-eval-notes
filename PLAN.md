# llm-eval-notes Implementation Plan

## Goal
Create a public artifact demonstrating AI engineering capability. Not a README dump—real tests, real metrics, clean code.

## Stack
- **Python** (lingua franca for evals)
- **uv** for packaging (fast, modern)
- **pytest** for tests
- **pydantic** for schemas
- **ruff + mypy** for hygiene
- **typer** for CLI (optional but nice)

## Repo Structure
```
llm-eval-notes/
├── pyproject.toml
├── src/llm_eval/
│   ├── __init__.py
│   ├── providers/
│   │   ├── base.py          # Provider protocol
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── xai.py
│   ├── evals/
│   │   ├── hallucination.py
│   │   ├── brittleness.py
│   │   └── structured_output.py
│   └── cli.py
├── tests/
│   ├── test_hallucination.py
│   ├── test_brittleness.py
│   └── test_structured_output.py
├── experiments/
│   ├── results/             # Generated results
│   └── configs/             # Eval configs
└── README.md
```

## 1. Project Skeleton

### pyproject.toml
```toml
[project]
name = "llm-eval-notes"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.0",
    "anthropic>=0.40",
    "pydantic>=2.0",
    "pytest>=8.0",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.8",
    "mypy>=1.0",
]

[tool.uv]
dev-dependencies = [
    "pytest-asyncio>=0.24",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
strict = true
```

### Provider Protocol
```python
# src/llm_eval/providers/base.py
from typing import Protocol, override
import pydantic

class Message(pydantic.BaseModel):
    role: str
    content: str

class CompletionResult(pydantic.BaseModel):
    content: str
    model: str
    tokens_used: int | None = None

class LLMProvider(Protocol):
    def complete(self, messages: list[Message]) -> CompletionResult: ...
    @property
    def name(self) -> str: ...
```

## 2. Hallucination Tests

### Concept
Ground truth comparison—model says something not in source material.

### Data Structure
```python
# experiments/configs/hallucination.yaml
cases:
  - question: "What year was the company founded?"
    context: "Acme Corp was founded in 2019 by Jane Doe."
    ground_truth:
      answer: "2019"
      allowed_variants: ["in 2019", "2019 by Jane Doe"]
    hallucination_patterns:
      - "2020"
      - "John Smith"
```

### Metrics
- **Exact match rate**: `%` where answer matches exactly
- **Semantic similarity**: Embedding cosine similarity
- **Hallucination rate**: `%` with unsupported claims
- **Refusal rate**: `%` where model says "I don't know"

### Implementation Sketch
```python
def eval_hallucination(provider: LLMProvider, case: HallucinationCase) -> Result:
    response = provider.complete([
        Message(role="user", content=f"Context: {case.context}\n\nQ: {case.question}")
    ])
    return {
        "exact_match": response.content in case.ground_truth.allowed_variants,
        "has_hallucination": any(p in response.content for p in case.hallucination_patterns),
        "semantic_score": cosine_similarity(embed(response.content), embed(case.ground_truth.answer)),
    }
```

## 3. Prompt Brittleness Tests

### Concept
Same question, different phrasings—does the answer stay consistent?

### Data Structure
```python
# experiments/configs/brittleness.yaml
scenarios:
  - id: "capital-of-france"
    variations:
      - "What is the capital of France?"
      - "France's capital is...?"
      - "Name the capital city of France."
      - "Tell me: what's the capital of France?"
    expected_answer: "Paris"
    tolerance: "exact"  # or "fuzzy"
```

### Metrics
- **Consistency rate**: `%` giving same answer across variations
- **Refusal variance**: `%` refusing some but not all
- **Token efficiency**: Tokens per answer (bloat?)

### Edge Cases to Test
- One word changes ("not" vs "now")
- Order effects (Q1 then Q2 vs Q2 then Q1)
- Temperature sensitivity (0 vs 0.7 vs 1.0)

## 4. Structured Output Validation

### Concept
Force JSON output—count failures, track schema violations.

### Data Structure
```python
# experiments/configs/structured_output.yaml
schemas:
  - name: "person"
    pydantic_model: |
      class Person(BaseModel):
          name: str
          age: int
          email: str | None
          addresses: list[Address]

  - name: "nested-optional"
    pydantic_model: |
      class ComplexResponse(BaseModel):
          required_field: str
          optional_with_default: int = 42
          union_field: str | int
          nested: dict[str, list[str]]

prompts:
  - schema: "person"
    prompt: "Extract person info from: John is 30 years old."
  - schema: "nested-optional"
    prompt: "Return a valid response with nested data."
```

### Metrics
- **Parse success rate**: `%` valid JSON
- **Schema validation rate**: `%` matching Pydantic model
- **Common violations**: Missing fields, wrong types, extra fields
- **Retry effectiveness**: `%` fixed after "fix this JSON" nudge

### Implementation Sketch
```python
def eval_structured_output(provider: LLMProvider, case: StructuredCase) -> Result:
    response = provider.complete([
        Message(role="user", content=f"{case.prompt}\n\nRespond with JSON only.")
    ])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        return {"valid_json": False, "schema_valid": False}

    try:
        case.pydantic_model.validate_json(response.content)
        return {"valid_json": True, "schema_valid": True}
    except pydantic.ValidationError as e:
        return {"valid_json": True, "schema_valid": False, "errors": e.errors()}
```

## 5. Cross-Model Comparison

### Run Matrix
```
          | Hallucination | Brittleness | Structured
----------|----------------|-------------|-----------
GPT-4o    |       ✓        |      ✓      |     ✓
Claude 3.5|       ✓        |      ✓      |     ✓
Grok-2    |       ✓        |      ✓      |     ✓
```

### Report Format
```markdown
# Eval Results: 2025-02-18

## Hallucination Tests
| Model    | Exact Match | Hallucination Rate |
|----------|-------------|-------------------|
| GPT-4o   | 92%         | 3%                |
| Claude   | 95%         | 2%                |
| Grok     | 88%         | 8%                |

## Brittleness Tests
| Model    | Consistency | Refusal Variance |
|----------|-------------|------------------|
| GPT-4o   | 94%         | 1%               |
| Claude   | 97%         | 0%               |
| Grok     | 89%         | 5%               |

## Structured Output
| Model    | Valid JSON | Schema Valid |
|----------|------------|--------------|
| GPT-4o   | 99%        | 96%          |
| Claude   | 98%        | 95%          |
| Grok     | 92%        | 85%          |
```

## 6. CLI

```bash
# Run all evals across all models
uv run llm-eval run-all

# Run specific eval
uv run llm-eval run --eval hallucination --model gpt-4o

# Compare models
uv run llm-eval compare --models gpt-4o,claude-3.5-sonnet

# Generate report
uv run llm-eval report --format markdown > results.md
```

## 7. README Content

```markdown
# llm-eval-notes

Public evaluation artifacts. Real tests, real metrics.

## What This Tests

- **Hallucination**: Ground truth comparison, citation checking
- **Prompt Brittleness**: Phrasing variations, order effects
- **Structured Output**: JSON schema validation, Pydantic enforcement

## Quick Start

```bash
uv sync
uv run llm-eval run-all
```

## Sample Results

[Include actual results from running the evals]

## Adding Evals

See CONTRIBUTING.md. Short version: add YAML to `experiments/configs/`.
```

## Implementation Order

1. **Skeleton**: pyproject.toml, basic layout, CI (ruff + mypy)
2. **Provider layer**: OpenAI, Anthropic, xAI implementations
3. **Hallucination eval**: First working test
4. **Structured output**: Pydantic validation
5. **Brittleness eval**: Variations framework
6. **CLI**: typer interface
7. **Report generation**: Markdown output
8. **README**: Fill with real results

## Signals This Sends

- **Discipline**: Type hints, strict mode, tests
- **Pragmatism**: uv not poetry, YAML not bespoke DSL
- **Depth**: Multiple eval types, not one demo
- **Transparency**: Real metrics shown, failures included

## Anti-Patterns to Avoid

- Generic README without real results
- Only one model tested
- No metric definitions
- Untested code in an eval repo (ironic)
- Over-engineered abstractions

## Next Steps

1. Initialize repo with `uv init`
2. Create first eval (hallucination is easiest)
3. Get one provider working (OpenAI)
4. Expand to other evals
5. Add other providers
6. Generate initial results for README
