"""Streaming response validation - test streaming behavior and error recovery."""

from dataclasses import dataclass, field
from typing import Any, AsyncIterator
import asyncio

from llm_eval.providers.base import LLMProvider, CompletionResult


@dataclass
class StreamingCase:
    """A streaming validation test case."""

    prompt: str
    expected_content_contains: list[str]  # Phrases that should appear in response
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingResult:
    """Result of streaming validation."""

    case_id: str
    prompt: str
    chunks_received: int
    full_content: str
    matches_non_streaming: bool  # Content matches non-streaming response
    error_during_stream: bool
    error_message: str | None
    latency_ms: float  # Time to first chunk
    total_ms: float  # Total streaming time


class MockStreamingProvider:
    """Mock provider that supports streaming simulation."""

    def __init__(
        self,
        *,
        content: str = "This is a streaming response.",
        chunks: int = 5,
        error_at_chunk: int | None = None,
        error_message: str = "Stream error",
        chunk_delay_ms: float = 50.0,
    ):
        self.content = content
        self.chunks = chunks
        self.error_at_chunk = error_at_chunk
        self.error_message = error_message
        self.chunk_delay_ms = chunk_delay_ms

    async def stream(
        self,
        prompt: str,
    ) -> AsyncIterator[str]:
        """Simulate streaming response."""
        chunk_size = len(self.content) // self.chunks
        for i in range(self.chunks):
            if self.error_at_chunk is not None and i == self.error_at_chunk:
                raise RuntimeError(self.error_message)

            start = i * chunk_size
            end = start + chunk_size if i < self.chunks - 1 else len(self.content)
            chunk = self.content[start:end]

            await asyncio.sleep(self.chunk_delay_ms / 1000.0)
            yield chunk

    def complete(
        self,
        messages: list,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> CompletionResult:
        """Non-streaming completion."""
        return CompletionResult(
            content=self.content,
            model="mock-streaming",
            tokens_used=len(self.content.split()),
        )


class StreamingEval:
    """Evaluate streaming response behavior."""

    def __init__(self, cases: list[StreamingCase]):
        self.cases = cases

    async def run(self, provider: MockStreamingProvider) -> list[StreamingResult]:
        """Run streaming validation on all cases."""
        results = []
        for case in self.cases:
            result = await self._evaluate_case(provider, case)
            results.append(result)
        return results

    async def _evaluate_case(
        self,
        provider: MockStreamingProvider,
        case: StreamingCase,
    ) -> StreamingResult:
        """Evaluate a single streaming case."""
        import time

        # Collect streaming response
        chunks_received = 0
        full_content = ""
        error_during_stream = False
        error_message = None
        first_chunk_time = None
        start_time = time.time()

        try:
            async for chunk in provider.stream(case.prompt):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                chunks_received += 1
                full_content += chunk
        except RuntimeError as e:
            error_during_stream = True
            error_message = str(e)

        end_time = time.time()

        # Get non-streaming response for comparison
        from llm_eval.providers.base import Message
        non_streaming = provider.complete([Message(role="user", content=case.prompt)])

        # Check if content matches
        matches_non_streaming = full_content.strip() == non_streaming.content.strip()

        # Calculate latencies
        latency_ms = (first_chunk_time - start_time) * 1000 if first_chunk_time else 0
        total_ms = (end_time - start_time) * 1000

        return StreamingResult(
            case_id=case.case_id or case.prompt[:50],
            prompt=case.prompt,
            chunks_received=chunks_received,
            full_content=full_content,
            matches_non_streaming=matches_non_streaming,
            error_during_stream=error_during_stream,
            error_message=error_message,
            latency_ms=latency_ms,
            total_ms=total_ms,
        )

    def calculate_metrics(self, results: list[StreamingResult]) -> dict[str, Any]:
        """Calculate aggregate streaming metrics."""
        if not results:
            return {
                "match_rate": 0.0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
                "avg_chunks": 0.0,
            }

        n = len(results)
        return {
            "match_rate": sum(1 for r in results if r.matches_non_streaming) / n,
            "error_rate": sum(1 for r in results if r.error_during_stream) / n,
            "avg_latency_ms": sum(r.latency_ms for r in results) / n,
            "avg_total_ms": sum(r.total_ms for r in results) / n,
            "avg_chunks": sum(r.chunks_received for r in results) / n,
            "total_cases": n,
        }

    @staticmethod
    def default_cases() -> list[StreamingCase]:
        """Default streaming test cases."""
        return [
            StreamingCase(
                prompt="Write a short poem about AI.",
                expected_content_contains=["AI", "intelligence"],
                case_id="stream-poem",
            ),
            StreamingCase(
                prompt="Explain quantum computing in one paragraph.",
                expected_content_contains=["quantum", "computing"],
                case_id="stream-quantum",
            ),
            StreamingCase(
                prompt="List 3 benefits of exercise.",
                expected_content_contains=["exercise", "benefit"],
                case_id="stream-exercise",
            ),
        ]
