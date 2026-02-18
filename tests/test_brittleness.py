"""Tests for brittleness evaluation."""

from llm_eval.evals.brittleness import (
    DEFAULT_SCENARIOS,
    BrittlenessEval,
    BrittlenessResult,
    BrittlenessScenario,
    PromptVariation,
)
from llm_eval.providers.base import MockProvider


def test_prompt_variation_model() -> None:
    """Test PromptVariation model."""
    variation = PromptVariation(text="What is 2+2?")
    assert variation.text == "What is 2+2?"
    assert variation.expected_answer is None


def test_brittleness_scenario_model() -> None:
    """Test BrittlenessScenario model."""
    scenario = BrittlenessScenario(
        id="test",
        name="Test Scenario",
        variations=[PromptVariation(text="Test")],
        tolerance="exact",
    )

    assert scenario.id == "test"
    assert scenario.tolerance == "exact"


def test_brittleness_result_model() -> None:
    """Test BrittlenessResult model."""
    result = BrittlenessResult(
        scenario_id="test",
        responses=["A", "A", "A"],
        consistent=True,
        consistency_rate=1.0,
        refused_count=0,
        unique_answers=1,
        keywords_present_rate=1.0,
    )

    assert result.scenario_id == "test"
    assert result.consistent
    assert result.unique_answers == 1


def test_brittleness_eval_exact_consistency() -> None:
    """Test exact consistency detection."""
    provider = MockProvider(default_response="Paris")

    eval = BrittlenessEval(
        scenarios=[
            BrittlenessScenario(
                id="capital",
                name="Capital",
                variations=[
                    PromptVariation(text="What is the capital of France?"),
                    PromptVariation(text="France's capital is?"),
                ],
                tolerance="exact",
            )
        ]
    )

    results = eval.run(provider)
    assert len(results) == 1
    assert results[0].consistent
    assert results[0].unique_answers == 1


def test_brittleness_eval_inconsistency() -> None:
    """Test inconsistency detection."""
    provider = MockProvider(
        responses={
            "What is the capital of France?": "Paris",
            "France's capital is?": "The capital is Paris",
        },
        default_response="Paris",
    )

    eval = BrittlenessEval(
        scenarios=[
            BrittlenessScenario(
                id="capital",
                name="Capital",
                variations=[
                    PromptVariation(text="What is the capital of France?"),
                    PromptVariation(text="France's capital is?"),
                ],
                tolerance="exact",
            )
        ]
    )

    results = eval.run(provider)
    assert not results[0].consistent
    assert results[0].unique_answers == 2


def test_brittleness_eval_fuzzy_keywords() -> None:
    """Test fuzzy keyword matching."""
    provider = MockProvider(default_response="The answer is Paris, France")

    eval = BrittlenessEval(
        scenarios=[
            BrittlenessScenario(
                id="capital",
                name="Capital",
                variations=[
                    PromptVariation(text="What is the capital of France?"),
                ],
                tolerance="fuzzy",
                keywords=["paris"],
            )
        ]
    )

    results = eval.run(provider)
    assert results[0].consistent
    assert results[0].keywords_present_rate == 1.0


def test_brittleness_refusal_detection() -> None:
    """Test refusal detection."""
    provider = MockProvider(
        responses={
            "Q1": "I don't know",
            "Q2": "Answer",
        },
        default_response="Answer",
    )

    eval = BrittlenessEval(
        scenarios=[
            BrittlenessScenario(
                id="test",
                name="Test",
                variations=[
                    PromptVariation(text="Q1"),
                    PromptVariation(text="Q2"),
                ],
                tolerance="exact",
            )
        ]
    )

    results = eval.run(provider)
    assert results[0].refused_count == 1


def test_brittleness_metrics() -> None:
    """Test metrics calculation."""
    eval = BrittlenessEval(scenarios=DEFAULT_SCENARIOS[:2])

    provider = MockProvider(default_response="42")  # Consistent response

    results = eval.run(provider)
    metrics = eval.calculate_metrics(results)

    assert "total_scenarios" in metrics
    assert "consistent_scenarios" in metrics
    assert "avg_consistency_rate" in metrics
    assert metrics["total_scenarios"] == 2


def test_default_scenarios_exist() -> None:
    """Test that default scenarios are defined."""
    assert len(DEFAULT_SCENARIOS) > 0
    assert all(isinstance(s, BrittlenessScenario) for s in DEFAULT_SCENARIOS)
