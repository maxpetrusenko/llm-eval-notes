"""Tests for cost tracking."""

import pytest
from llm_eval.evals.cost_tracking import (
    CostRecord,
    CostReport,
    MODEL_PRICING,
)


def test_cost_record_model():
    """Test CostRecord creation."""
    record = CostRecord(
        model="gpt-4o",
        input_tokens=1000,
        output_tokens=500,
    )
    assert record.model == "gpt-4o"
    assert record.input_tokens == 1000
    assert record.output_tokens == 500


def test_cost_record_calculation():
    """Test automatic cost calculation."""
    # gpt-4o: $2.50/1M input, $10.00/1M output
    record = CostRecord(
        model="gpt-4o",
        input_tokens=1_000_000,  # 1M input
        output_tokens=1_000_000,  # 1M output
    )
    assert record.cost_usd == 12.50  # $2.50 + $10.00


def test_cost_record_with_cached_tokens():
    """Test cost calculation with cached tokens."""
    record = CostRecord(
        model="gpt-4o",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_tokens=500_000,  # 50% cached
    )
    # Input cost: $2.50, cached discount: $2.50 * 0.9 * 0.5 = $1.125
    # Expected: $2.50 - $1.125 = $1.375
    assert abs(record.cost_usd - 1.375) < 0.01


def test_cost_report_aggregation():
    """Test CostReport aggregates records correctly."""
    report = CostReport()
    report.add(CostRecord(model="gpt-4o", input_tokens=100, output_tokens=50))
    report.add(CostRecord(model="gpt-4o", input_tokens=200, output_tokens=100))
    report.add(CostRecord(model="gpt-4o-mini", input_tokens=500, output_tokens=250))

    assert report.total_input_tokens == 800
    assert report.total_output_tokens == 400
    assert report.total_tokens == 1200
    assert len(report.records) == 3


def test_cost_report_by_model():
    """Test breakdown by model."""
    report = CostReport()
    report.add(CostRecord(model="gpt-4o", input_tokens=100, output_tokens=50))
    report.add(CostRecord(model="gpt-4o-mini", input_tokens=200, output_tokens=100))

    by_model = report.by_model()
    assert "gpt-4o" in by_model
    assert "gpt-4o-mini" in by_model
    assert by_model["gpt-4o"]["input_tokens"] == 100
    assert by_model["gpt-4o-mini"]["input_tokens"] == 200


def test_cost_report_by_eval_type():
    """Test breakdown by evaluation type."""
    report = CostReport()
    report.add(CostRecord(
        model="gpt-4o", input_tokens=100, output_tokens=50, eval_type="hallucination"
    ))
    report.add(CostRecord(
        model="gpt-4o", input_tokens=200, output_tokens=100, eval_type="safety"
    ))

    by_type = report.by_eval_type()
    assert "hallucination" in by_type
    assert "safety" in by_type
    assert by_type["hallucination"]["calls"] == 1
    assert by_type["safety"]["calls"] == 1


def test_cost_report_to_markdown():
    """Test markdown formatting."""
    report = CostReport()
    report.add(CostRecord(model="gpt-4o", input_tokens=100, output_tokens=50))

    md = report.to_markdown()
    assert "# Cost Report" in md
    assert "gpt-4o" in md
    assert "$" in md


def test_cost_report_to_dict():
    """Test dictionary export."""
    report = CostReport()
    report.add(CostRecord(model="gpt-4o", input_tokens=100, output_tokens=50))

    d = report.to_dict()
    assert d["total_input_tokens"] == 100
    assert d["total_output_tokens"] == 50
    assert "by_model" in d
    assert "by_eval_type" in d
    assert "records" in d


def test_model_pricing_exists():
    """Test that key models have pricing."""
    assert "gpt-4o" in MODEL_PRICING
    assert "gpt-4o-mini" in MODEL_PRICING
    assert "claude-3-5-sonnet-20241022" in MODEL_PRICING
    assert "mock-model" in MODEL_PRICING


def test_mock_model_zero_cost():
    """Test that mock model has zero cost."""
    record = CostRecord(model="mock-model", input_tokens=1_000_000, output_tokens=1_000_000)
    assert record.cost_usd == 0.0


def test_unknown_model_zero_cost():
    """Test that unknown models default to zero cost."""
    record = CostRecord(model="unknown-model-xyz", input_tokens=1000, output_tokens=500)
    assert record.cost_usd == 0.0
