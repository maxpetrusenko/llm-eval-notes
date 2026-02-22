"""Tests for reasoning chain evaluation."""

import pytest
from llm_eval.evals.reasoning import (
    ReasoningCase,
    ReasoningResult,
    ReasoningEval,
    ReasoningStep,
)


def test_reasoning_step_model():
    """Test ReasoningStep dataclass."""
    step = ReasoningStep(content="First, I analyze the input", step_number=1)
    assert step.content == "First, I analyze the input"
    assert step.step_number == 1


def test_reasoning_case_model():
    """Test ReasoningCase model."""
    case = ReasoningCase(
        prompt="Solve: 2 + 2",
        expected_steps=["Analyze the numbers", "Add them together", "State the result"],
        expected_conclusion="4",
    )
    assert case.prompt == "Solve: 2 + 2"
    assert len(case.expected_steps) == 3


def test_reasoning_result_model():
    """Test ReasoningResult model."""
    result = ReasoningResult(
        case_id="test-1",
        prompt="Test",
        response="Step 1: Analyze. Step 2: Compute.",
        steps_found=2,
        steps_expected=3,
        missing_steps=["State result"],
        has_logical_error=False,
        step_coherence_rate=0.8,
        logical_error_rate=0.0,
    )
    assert result.steps_found == 2
    assert result.steps_expected == 3
    assert len(result.missing_steps) == 1


def test_reasoning_eval_with_mock():
    """Test reasoning eval with mock provider."""
    from llm_eval.providers.base import MockProvider

    provider = MockProvider(
        default_response="Let me think step by step.\n1. First, I analyze the input.\n2. Then I compute the result.\n3. The answer is 4."
    )

    cases = [
        ReasoningCase(
            prompt="Solve: 2 + 2",
            expected_steps=["analyze", "compute"],
            expected_conclusion="4",
        )
    ]

    eval_runner = ReasoningEval(cases)
    results = eval_runner.run(provider)

    assert len(results) == 1
    assert results[0].steps_found >= 1


def test_reasoning_eval_detects_missing_steps():
    """Test that eval detects missing reasoning steps."""
    from llm_eval.providers.base import MockProvider

    # Response missing step 3
    provider = MockProvider(
        default_response="1. First, I analyze.\n2. Then I compute."
    )

    cases = [
        ReasoningCase(
            prompt="Solve: 2 + 2",
            expected_steps=["analyze", "compute", "conclude"],
            expected_conclusion="4",
        )
    ]

    eval_runner = ReasoningEval(cases)
    results = eval_runner.run(provider)

    assert len(results[0].missing_steps) >= 1


def test_reasoning_eval_detects_logical_error():
    """Test detection of logical errors."""
    from llm_eval.providers.base import MockProvider

    # Response with logical error (2+2=5 is wrong)
    provider = MockProvider(
        default_response="1. I analyze 2 + 2.\n2. The answer is 5."
    )

    cases = [
        ReasoningCase(
            prompt="Solve: 2 + 2",
            expected_steps=["analyze", "answer"],
            expected_conclusion="4",
        )
    ]

    eval_runner = ReasoningEval(cases)
    results = eval_runner.run(provider)

    assert results[0].has_logical_error is True


def test_reasoning_metrics():
    """Test metrics calculation."""
    results = [
        ReasoningResult(
            case_id="1",
            prompt="Test",
            response="Response",
            steps_found=3,
            steps_expected=3,
            missing_steps=[],
            has_logical_error=False,
            step_coherence_rate=1.0,
            logical_error_rate=0.0,
        ),
        ReasoningResult(
            case_id="2",
            prompt="Test",
            response="Response",
            steps_found=2,
            steps_expected=3,
            missing_steps=["step3"],
            has_logical_error=True,
            step_coherence_rate=0.5,
            logical_error_rate=1.0,
        ),
    ]

    eval_runner = ReasoningEval([])
    metrics = eval_runner.calculate_metrics(results)

    assert "avg_step_coherence_rate" in metrics
    assert "avg_logical_error_rate" in metrics
    assert metrics["avg_step_coherence_rate"] == 0.75  # (1.0 + 0.5) / 2
    assert metrics["avg_logical_error_rate"] == 0.5  # (0 + 1) / 2


def test_default_reasoning_cases_exist():
    """Test that default cases are available."""
    cases = ReasoningEval.default_cases()
    assert len(cases) >= 3
    assert all(isinstance(c, ReasoningCase) for c in cases)
