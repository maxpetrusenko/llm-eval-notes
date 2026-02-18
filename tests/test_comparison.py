"""Tests for comparison framework."""

from llm_eval.evals.brittleness import BrittlenessEval, BrittlenessScenario, PromptVariation
from llm_eval.evals.comparison import ComparisonRunner
from llm_eval.evals.hallucination import HallucinationCase, HallucinationEval
from llm_eval.providers.base import MockProvider


def test_comparison_runner_creation() -> None:
    """Test ComparisonRunner creation."""
    runner = ComparisonRunner()
    assert runner.hallucination is None
    assert runner.brittleness is None
    assert runner.structured is None


def test_comparison_runner_with_defaults() -> None:
    """Test ComparisonRunner.with_defaults()."""
    runner = ComparisonRunner.with_defaults()

    assert runner.hallucination is not None
    assert runner.brittleness is not None
    assert runner.structured is not None


def test_comparison_runner_run_all() -> None:
    """Test running all evals."""
    runner = ComparisonRunner(
        hallucination=HallucinationEval(
            cases=[
                HallucinationCase(
                    id="test",
                    question="What year?",
                    context="2019",
                    ground_truth="2019",
                    allowed_variants=["2019"],
                    hallucination_patterns=[],
                )
            ]
        ),
        brittleness=BrittlenessEval(
            scenarios=[
                BrittlenessScenario(
                    id="test",
                    name="Test",
                    variations=[PromptVariation(text="Test")],
                    tolerance="exact",
                )
            ]
        ),
    )

    provider = MockProvider(default_response="2019")
    report = runner.run_all([provider])

    assert "timestamp" in report.model_dump()
    assert "runs" in report.model_dump()
    assert "summary" in report.model_dump()


def test_comparison_report_to_markdown() -> None:
    """Test markdown generation."""
    runner = ComparisonRunner(
        hallucination=HallucinationEval(
            cases=[
                HallucinationCase(
                    id="test",
                    question="What?",
                    context="Context",
                    ground_truth="Answer",
                    allowed_variants=[],
                    hallucination_patterns=[],
                )
            ]
        ),
    )

    provider = MockProvider(default_response="Answer")
    report = runner.run_all([provider])

    md = report.to_markdown()
    assert "# LLM Eval Results" in md
    assert "Hallucination Tests" in md
    assert "|" in md  # Has table formatting


def test_comparison_report_summary_structure() -> None:
    """Test summary structure."""
    runner = ComparisonRunner.with_defaults()

    provider1 = MockProvider(name="provider1", model="model1", default_response="Response")
    provider2 = MockProvider(name="provider2", model="model2", default_response="Response")

    report = runner.run_all([provider1, provider2])

    assert "hallucination" in report.summary
    assert "brittleness" in report.summary
    assert "structured" in report.summary

    # Check we have results for both providers
    assert len(report.runs) == 2
