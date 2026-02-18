# Evaluation Taxonomy

## Hallucination Tests

### What Counts as Hallucination

We classify hallucinations into three categories:

1. **Unsupported Claim**: The model asserts something not present in the provided context.
   - Example: Context says "founded in 2019", model says "founded in 2020"
   - Example: Context mentions "Jane Doe", model claims "John Smith"

2. **Wrong Entity**: The model correctly identifies a category but substitutes the wrong specific entity.
   - Example: Context says "CEO of Acme is Jane", model says "CEO is Sarah"

3. **Wrong Number/Value**: The model produces a numerically incorrect value.
   - Example: Context says "$50M revenue", model says "$500M"

### Refusal Behavior

A "good refusal" is when the model says "I don't know" or "not mentioned" for information truly absent from context. A "bad refusal" is refusing when the answer IS present.

### Metrics

- **Exact Match Rate**: Response matches ground truth exactly
- **Allowed Variant Rate**: Response matches an acceptable paraphrase
- **Hallucination Rate**: Response contains unsupported claims
- **Refusal Rate**: Model refuses to answer (good for absent info)
- **Safe Rate**: No hallucinations detected

## Prompt Brittleness Tests

### Measurement

We measure brittleness through **response variance** across semantically equivalent prompts.

### Variance Metric

For a scenario with N variations:
- **Consistency Rate**: Fraction of variations yielding identical (normalized) answers
- **Unique Answers**: Count of distinct responses after normalization
- **Keywords Present Rate**: For fuzzy tolerance, fraction of expected keywords appearing

### Brittleness Categories

1. **Phrasing Variance**: Same question, different words
   - "What is the capital?" vs "France's capital is...?"

2. **Word-Order Sensitivity**: Semantic meaning preserved, order changed
   - "Cat sat on mat" vs "On the mat sat a cat"

3. **Minimal Perturbation**: One-word changes that flip meaning
   - "not" vs "now", "hot" vs "not"

### Metrics

- **Consistency Rate**: Higher is better (1.0 = perfectly consistent)
- **Avg Unique Answers**: Lower is better (1.0 = all same answer)
- **Refusal Variance**: Number of variations that triggered refusal

## Structured Output Tests

### What We Validate

1. **JSON Validity**: Response is parseable JSON
2. **Schema Conformance**: Response matches expected Pydantic model
3. **Type Correctness**: Fields have correct types (int, str, array, object)
4. **Required Fields**: All required fields present
5. **No Extra Fields**: Strict mode (no undefined properties)

### Failure Modes

1. **Not JSON**: Response is plain text, YAML, or malformed
2. **Missing Required**: Schema-required field absent
3. **Type Mismatch**: String where int expected, etc.
4. **Unexpected Field**: Extra properties not in schema
5. **Nested Violation**: Sub-objects or arrays fail validation

### Metrics

- **Valid JSON Rate**: Response is parseable
- **Schema Valid Rate**: Passes Pydantic validation
- **Retry Success Rate**: Fixed after "fix this JSON" nudge

## Tool-Use Evaluation (Experimental)

### What We Test

1. **Tool Selection**: Correct function chosen for task
2. **Argument Extraction**: Correct parameters passed
3. **Argument Format**: Types match tool schema

### Example Case

```
Task: "What's the weather in Tokyo?"

Expected:
- tool: "get_weather"
- arguments: {"location": "Tokyo"}

Failure modes:
- Wrong tool: get_time, search_web
- Missing argument: {}
- Wrong type: {"location": 123}
```

## Running Your Own Evals

```bash
# Run all evals with mock (no API keys)
uv run llm-eval run all --provider mock

# Run with real API
export OPENAI_API_KEY=sk-...
uv run llm-eval run all --provider openai --model gpt-4o

# Compare models
uv run llm-eval compare --models gpt-4o,claude-3-5-sonnet-20241022
```

## Result Format

Results are stored in `experiments/YYYY-MM-DD/` as:
- `results.jsonl`: One JSON line per evaluation run
- `summary.md`: Aggregated metrics table
