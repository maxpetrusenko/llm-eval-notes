"""Tests for streaming validation."""

import pytest
import asyncio
from llm_eval.evals.streaming import (
    StreamingCase,
    StreamingResult,
    MockStreamingProvider,
    StreamingEval,
)


def test_streaming_case_model():
    """Test StreamingCase creation."""
    case = StreamingCase(
        prompt="Test prompt",
        expected_content_contains=["test", "response"],
    )
    assert case.prompt == "Test prompt"
    assert len(case.expected_content_contains) == 2


def test_streaming_result_model():
    """Test StreamingResult creation."""
    result = StreamingResult(
        case_id="test-1",
        prompt="Test",
        chunks_received=5,
        full_content="Full response",
        matches_non_streaming=True,
        error_during_stream=False,
        error_message=None,
        latency_ms=100.0,
        total_ms=500.0,
    )
    assert result.chunks_received == 5
    assert result.matches_non_streaming is True


def test_mock_streaming_provider():
    """Test MockStreamingProvider basic functionality."""
    provider = MockStreamingProvider(content="Hello world", chunks=2)

    # Test non-streaming
    result = provider.complete([{"role": "user", "content": "Hi"}])
    assert result.content == "Hello world"


@pytest.mark.asyncio
async def test_mock_streaming_provider_stream():
    """Test streaming from MockStreamingProvider."""
    provider = MockStreamingProvider(content="Hello world", chunks=2, chunk_delay_ms=10)

    chunks = []
    async for chunk in provider.stream("Hi"):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert "".join(chunks) == "Hello world"


@pytest.mark.asyncio
async def test_streaming_eval_basic():
    """Test basic streaming evaluation."""
    provider = MockStreamingProvider(
        content="AI is transforming the world of computing.",
        chunks=3,
        chunk_delay_ms=10,
    )

    cases = [
        StreamingCase(
            prompt="What is AI?",
            expected_content_contains=["AI"],
            case_id="test-ai",
        )
    ]

    eval_runner = StreamingEval(cases)
    results = await eval_runner.run(provider)

    assert len(results) == 1
    assert results[0].chunks_received == 3
    assert results[0].matches_non_streaming is True
    assert results[0].error_during_stream is False


@pytest.mark.asyncio
async def test_streaming_eval_error_recovery():
    """Test detection of stream errors."""
    provider = MockStreamingProvider(
        content="This should not complete",
        chunks=5,
        error_at_chunk=2,
        error_message="Connection lost",
        chunk_delay_ms=10,
    )

    cases = [
        StreamingCase(
            prompt="Test",
            expected_content_contains=[],
            case_id="error-test",
        )
    ]

    eval_runner = StreamingEval(cases)
    results = await eval_runner.run(provider)

    assert results[0].error_during_stream is True
    assert "Connection lost" in results[0].error_message
    assert results[0].chunks_received == 2  # Error after 2 chunks


@pytest.mark.asyncio
async def test_streaming_eval_metrics():
    """Test metrics calculation."""
    provider = MockStreamingProvider(content="Test response", chunks=2, chunk_delay_ms=10)

    cases = [
        StreamingCase(prompt="Test 1", expected_content_contains=[], case_id="1"),
        StreamingCase(prompt="Test 2", expected_content_contains=[], case_id="2"),
    ]

    eval_runner = StreamingEval(cases)
    results = await eval_runner.run(provider)

    metrics = eval_runner.calculate_metrics(results)

    assert "match_rate" in metrics
    assert "error_rate" in metrics
    assert metrics["match_rate"] == 1.0  # All matched
    assert metrics["error_rate"] == 0.0  # No errors


def test_default_streaming_cases_exist():
    """Test that default cases are available."""
    cases = StreamingEval.default_cases()
    assert len(cases) >= 3
    assert all(isinstance(c, StreamingCase) for c in cases)


@pytest.mark.asyncio
async def test_streaming_latency_measured():
    """Test that streaming latency is measured."""
    provider = MockStreamingProvider(
        content="Test response here",
        chunks=3,
        chunk_delay_ms=20,
    )

    cases = [StreamingCase(prompt="Test", expected_content_contains=[], case_id="latency-test")]

    eval_runner = StreamingEval(cases)
    results = await eval_runner.run(provider)

    # Should have some latency > 0
    assert results[0].latency_ms > 0
    assert results[0].total_ms > results[0].latency_ms
