"""Tests for tool-use evaluation."""

from llm_eval.evals.tool_use import (
    DEFAULT_CASES,
    Tool,
    ToolUseCase,
    ToolUseEval,
    ToolUseResult,
)
from llm_eval.providers.base import MockProvider


def test_tool_model() -> None:
    """Test Tool model."""
    tool = Tool(
        name="test",
        description="A test tool",
        parameters={"type": "object", "properties": {}},
    )
    assert tool.name == "test"


def test_tool_use_case_model() -> None:
    """Test ToolUseCase model."""
    case = ToolUseCase(
        id="test",
        task="Do something",
        expected_tool="tool1",
        expected_params={"x": 1},
        available_tools=[],
    )
    assert case.id == "test"
    assert case.expected_tool == "tool1"


def test_tool_use_result_model() -> None:
    """Test ToolUseResult model."""
    result = ToolUseResult(
        case_id="test",
        raw_response='{"tool": "t1", "parameters": {}}',
        tool_selected="t1",
        parameters_extracted={},
        correct_tool=True,
        correct_params=True,
    )
    assert result.correct_tool
    assert result.correct_params


def test_tool_use_eval_with_mock() -> None:
    """Test tool-use eval with mock provider."""
    provider = MockProvider(
        default_response='{"tool": "get_weather", "parameters": {"location": "Tokyo"}}'
    )

    eval = ToolUseEval(
        cases=[
            ToolUseCase(
                id="weather",
                task="Weather in Tokyo?",
                expected_tool="get_weather",
                expected_params={"location": "Tokyo"},
                available_tools=[],
            )
        ]
    )

    results = eval.run(provider)
    assert len(results) == 1
    assert results[0].correct_tool
    assert results[0].correct_params


def test_tool_use_eval_wrong_tool() -> None:
    """Test wrong tool detection."""
    provider = MockProvider(
        default_response='{"tool": "search", "parameters": {"query": "Tokyo"}}'
    )

    eval = ToolUseEval(
        cases=[
            ToolUseCase(
                id="weather",
                task="Weather in Tokyo?",
                expected_tool="get_weather",
                expected_params={"location": "Tokyo"},
                available_tools=[],
            )
        ]
    )

    results = eval.run(provider)
    assert not results[0].correct_tool


def test_tool_use_eval_parse_error() -> None:
    """Test parse error handling."""
    provider = MockProvider(default_response="I think it's sunny in Tokyo")

    eval = ToolUseEval(
        cases=[
            ToolUseCase(
                id="weather",
                task="Weather in Tokyo?",
                expected_tool="get_weather",
                expected_params={"location": "Tokyo"},
                available_tools=[],
            )
        ]
    )

    results = eval.run(provider)
    assert results[0].parse_error is not None
    assert not results[0].correct_tool


def test_tool_use_eval_with_code_blocks() -> None:
    """Test parsing JSON from markdown code blocks."""
    provider = MockProvider(
        default_response='```json\n{"tool": "get_weather", "parameters": {"location": "Tokyo"}}\n```'
    )

    eval = ToolUseEval(
        cases=[
            ToolUseCase(
                id="weather",
                task="Weather in Tokyo?",
                expected_tool="get_weather",
                expected_params={"location": "Tokyo"},
                available_tools=[],
            )
        ]
    )

    results = eval.run(provider)
    assert results[0].correct_tool


def test_tool_use_metrics() -> None:
    """Test metrics calculation."""
    eval = ToolUseEval(cases=DEFAULT_CASES[:3])

    # Mix of correct and incorrect
    provider = MockProvider(
        responses={
            "What's the weather in Tokyo?": '{"tool": "get_weather", "parameters": {"location": "Tokyo"}}',
            "Check the weather in Paris in fahrenheit": '{"tool": "get_weather", "parameters": {"location": "Paris"}}',  # Missing unit
            "Search for information about quantum computing": '{"tool": "search", "parameters": {"query": "quantum computing"}}',
        },
        default_response='{"tool": "unknown", "parameters": {}}',
    )

    results = eval.run(provider)
    metrics = eval.calculate_metrics(results)

    assert "total_cases" in metrics
    assert "tool_selection_accuracy" in metrics
    assert "parameter_accuracy" in metrics
    assert metrics["total_cases"] == 3


def test_default_cases_exist() -> None:
    """Test that default cases are defined."""
    assert len(DEFAULT_CASES) > 0
    assert all(isinstance(c, ToolUseCase) for c in DEFAULT_CASES)
