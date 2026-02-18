"""Tests for structured output evaluation."""


from llm_eval.evals.structured import (
    DEFAULT_CASES,
    NESTED_SCHEMA,
    PERSON_SCHEMA,
    StructuredCase,
    StructuredOutputEval,
    StructuredResult,
)
from llm_eval.providers.base import MockProvider


def test_structured_case_model() -> None:
    """Test StructuredCase model."""
    case = StructuredCase(
        id="test",
        prompt="Extract info",
        schema_name="Person",
        schema_definition=PERSON_SCHEMA,
    )

    assert case.id == "test"
    assert case.schema_name == "Person"


def test_structured_result_model() -> None:
    """Test StructuredResult model."""
    result = StructuredResult(
        case_id="test",
        raw_response='{"name": "John", "age": 30}',
        valid_json=True,
        schema_valid=True,
        passed_after_retry=False,
    )

    assert result.case_id == "test"
    assert result.valid_json
    assert result.schema_valid


def test_structured_eval_valid_json() -> None:
    """Test valid JSON detection."""
    provider = MockProvider(default_response='{"name": "John", "age": 30}')

    eval = StructuredOutputEval(
        cases=[
            StructuredCase(
                id="person",
                prompt="Extract person",
                schema_name="Person",
                schema_definition=PERSON_SCHEMA,
            )
        ]
    )

    results = eval.run(provider)
    assert len(results) == 1
    assert results[0].valid_json


def test_structured_eval_invalid_json() -> None:
    """Test invalid JSON detection."""
    provider = MockProvider(default_response='{"name": "John", "age":}')  # Invalid

    eval = StructuredOutputEval(
        cases=[
            StructuredCase(
                id="person",
                prompt="Extract person",
                schema_name="Person",
                schema_definition=PERSON_SCHEMA,
            )
        ]
    )

    results = eval.run(provider)
    assert not results[0].valid_json
    assert results[0].parse_error is not None


def test_structured_eval_schema_validation() -> None:
    """Test schema validation."""
    # Missing required field 'age'
    provider = MockProvider(default_response='{"name": "John"}')

    eval = StructuredOutputEval(
        cases=[
            StructuredCase(
                id="person",
                prompt="Extract person",
                schema_name="Person",
                schema_definition=PERSON_SCHEMA,
            )
        ]
    )

    results = eval.run(provider)
    assert results[0].valid_json  # JSON is valid
    assert not results[0].schema_valid  # But schema fails
    assert len(results[0].validation_errors) > 0


def test_structured_eval_unexpected_field() -> None:
    """Test unexpected field detection."""
    provider = MockProvider(default_response='{"name": "John", "age": 30, "extra": "field"}')

    eval = StructuredOutputEval(
        cases=[
            StructuredCase(
                id="person",
                prompt="Extract person",
                schema_name="Person",
                schema_definition=PERSON_SCHEMA,
            )
        ]
    )

    results = eval.run(provider)
    assert results[0].valid_json
    # Should flag unexpected field
    assert any(e["error"] == "unexpected_field" for e in results[0].validation_errors)


def test_structured_eval_nested_schema() -> None:
    """Test nested schema validation."""
    provider = MockProvider(default_response='{"title": "Test", "tags": ["a", "b"], "metadata": {"created": "2025-01-01"}}')

    eval = StructuredOutputEval(
        cases=[
            StructuredCase(
                id="nested",
                prompt="Create nested object",
                schema_name="Nested",
                schema_definition=NESTED_SCHEMA,
            )
        ]
    )

    results = eval.run(provider)
    assert results[0].valid_json
    assert results[0].schema_valid


def test_structured_eval_type_mismatch() -> None:
    """Test type mismatch detection."""
    # age should be int, not string
    provider = MockProvider(default_response='{"name": "John", "age": "thirty"}')

    eval = StructuredOutputEval(
        cases=[
            StructuredCase(
                id="person",
                prompt="Extract person",
                schema_name="Person",
                schema_definition=PERSON_SCHEMA,
            )
        ]
    )

    results = eval.run(provider)
    assert results[0].valid_json
    assert not results[0].schema_valid
    assert any("expected_integer" in e["error"] for e in results[0].validation_errors)


def test_structured_metrics() -> None:
    """Test metrics calculation."""
    eval = StructuredOutputEval(cases=DEFAULT_CASES[:3])

    # Mix of valid and invalid
    provider = MockProvider(
        responses={
            "Extract person info from: John is 30 years old and his email is john@example.com\n\nRespond with valid JSON only.": '{"name": "John", "age": 30}',
            "Extract person info from: Jane works at Acme Corp\n\nRespond with valid JSON only.": "not json",
            'Create a blog post object with title "Hello World" and tags ["tech", "python"]\n\nRespond with valid JSON only.': '{"title": "Hello", "tags": ["tech"]}',
        },
        default_response='{"name": "Default", "age": 0}',
    )

    results = eval.run(provider)
    metrics = eval.calculate_metrics(results)

    assert "total_cases" in metrics
    assert "valid_json_rate" in metrics
    assert "schema_valid_rate" in metrics
    assert metrics["total_cases"] == 3


def test_default_cases_exist() -> None:
    """Test that default cases are defined."""
    assert len(DEFAULT_CASES) > 0
    assert all(isinstance(c, StructuredCase) for c in DEFAULT_CASES)


def test_person_schema() -> None:
    """Test person schema definition."""
    assert "properties" in PERSON_SCHEMA
    assert "name" in PERSON_SCHEMA["properties"]
    assert "age" in PERSON_SCHEMA["properties"]
    assert "name" in PERSON_SCHEMA["required"]
    assert "age" in PERSON_SCHEMA["required"]
