"""Structured output validation tests."""

import json
from typing import Any

import pydantic

from llm_eval.providers.base import LLMProvider, Message


class StructuredCase(pydantic.BaseModel):
    """A structured output test case."""

    id: str
    prompt: str
    schema_name: str
    schema_definition: dict[str, Any]
    retry_prompt: str = "Fix the JSON. Respond with valid JSON only."


class StructuredResult(pydantic.BaseModel):
    """Result of a structured output test."""

    case_id: str
    raw_response: str
    valid_json: bool
    schema_valid: bool
    passed_after_retry: bool
    parse_error: str | None = None
    validation_errors: list[dict[str, Any]] = []


class StructuredOutputEval(pydantic.BaseModel):
    """Structured output evaluation runner."""

    cases: list[StructuredCase]

    def run(self, provider: LLMProvider) -> list[StructuredResult]:
        """Run all structured output tests."""
        results = []
        for case in self.cases:
            result = self._run_case(case, provider)
            results.append(result)
        return results

    def _run_case(self, case: StructuredCase, provider: LLMProvider) -> StructuredResult:
        """Run a single test case."""
        prompt = f"{case.prompt}\n\nRespond with valid JSON only."

        response = provider.complete(
            [Message(role="user", content=prompt)],
            temperature=0.0,
            json_mode=True,
        )

        # Try to parse JSON
        try:
            parsed = json.loads(response.content)
            valid_json = True
            parse_error = None
        except json.JSONDecodeError as e:
            valid_json = False
            parsed = None
            parse_error = str(e)

        # Validate against schema
        if valid_json and parsed is not None:
            validation_errors = self._validate_schema(parsed, case.schema_definition)
            schema_valid = len(validation_errors) == 0
        else:
            validation_errors = []
            schema_valid = False

        # Retry if failed
        passed_after_retry = False
        if not (valid_json and schema_valid):
            retry_response = provider.complete(
                [
                    Message(role="user", content=prompt),
                    Message(role="assistant", content=response.content),
                    Message(role="user", content=case.retry_prompt),
                ],
                temperature=0.0,
            )
            try:
                json.loads(retry_response.content)
                passed_after_retry = True
            except json.JSONDecodeError:
                pass

        return StructuredResult(
            case_id=case.id,
            raw_response=response.content,
            valid_json=valid_json,
            schema_valid=schema_valid,
            passed_after_retry=passed_after_retry,
            parse_error=parse_error,
            validation_errors=validation_errors,
        )

    def _validate_schema(self, data: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate data against schema definition."""
        errors: list[dict[str, Any]] = []

        # Check required fields
        for field_name, field_def in schema.get("properties", {}).items():
            if field_name in schema.get("required", []):
                if field_name not in data:
                    errors.append({
                        "field": field_name,
                        "error": "missing_required_field",
                    })

            # Check type if present
            if field_name in data:
                expected_type = field_def.get("type")
                if expected_type == "string" and not isinstance(data[field_name], str):
                    errors.append({
                        "field": field_name,
                        "error": f"expected_string_got_{type(data[field_name]).__name__}",
                    })
                elif expected_type == "integer" and not isinstance(data[field_name], int):
                    errors.append({
                        "field": field_name,
                        "error": f"expected_integer_got_{type(data[field_name]).__name__}",
                    })
                elif expected_type == "array" and not isinstance(data[field_name], list):
                    errors.append({
                        "field": field_name,
                        "error": f"expected_array_got_{type(data[field_name]).__name__}",
                    })
                elif expected_type == "object" and not isinstance(data[field_name], dict):
                    errors.append({
                        "field": field_name,
                        "error": f"expected_object_got_{type(data[field_name]).__name__}",
                    })

        # Check for unexpected fields (strict mode)
        defined_fields = set(schema.get("properties", {}).keys())
        for field_name in data:
            if field_name not in defined_fields:
                errors.append({
                    "field": field_name,
                    "error": "unexpected_field",
                })

        return errors

    def calculate_metrics(self, results: list[StructuredResult]) -> dict[str, float]:
        """Calculate aggregate metrics."""
        if not results:
            return {}

        total = len(results)
        return {
            "total_cases": total,
            "valid_json_rate": sum(r.valid_json for r in results) / total,
            "schema_valid_rate": sum(r.schema_valid for r in results) / total,
            "retry_success_rate": sum(r.passed_after_retry for r in results) / total,
        }


# Test schemas
PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
    },
    "required": ["name", "age"],
}

NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "metadata": {
            "type": "object",
            "properties": {
                "created": {"type": "string"},
                "author": {"type": "string"},
            },
            "required": ["created"],
        },
    },
    "required": ["title"],
}

UNION_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "value": {"type": "integer"},  # Should also accept string in practice
        "active": {"type": "boolean"},
    },
    "required": ["id", "value"],
}

_default_cases = [
    StructuredCase(
        id="person-simple",
        prompt="Extract person info from: John is 30 years old and his email is john@example.com",
        schema_name="Person",
        schema_definition=PERSON_SCHEMA,
    ),
    StructuredCase(
        id="person-missing-field",
        prompt="Extract person info from: Jane works at Acme Corp",
        schema_name="Person",
        schema_definition=PERSON_SCHEMA,
    ),
    StructuredCase(
        id="nested-simple",
        prompt='Create a blog post object with title "Hello World" and tags ["tech", "python"]',
        schema_name="Nested",
        schema_definition=NESTED_SCHEMA,
    ),
    StructuredCase(
        id="nested-complex",
        prompt='Create an object: title="Deep Dive", tags=["ai", "ml"], metadata={"created":"2025-01-01"}',
        schema_name="Nested",
        schema_definition=NESTED_SCHEMA,
    ),
    StructuredCase(
        id="union-numeric",
        prompt='Return object with id="123", value=42, active=true',
        schema_name="Union",
        schema_definition=UNION_SCHEMA,
    ),
    StructuredCase(
        id="union-string-value",
        prompt='Return object with id="abc", value="not-a-number", active=false',
        schema_name="Union",
        schema_definition=UNION_SCHEMA,
    ),
]

DEFAULT_CASES = _default_cases
