"""Reasoning chain evaluation - validates step-by-step reasoning quality."""

from dataclasses import dataclass, field
from typing import Any
import re

from llm_eval.providers.base import LLMProvider


@dataclass
class ReasoningStep:
    """A single reasoning step."""

    content: str
    step_number: int


@dataclass
class ReasoningCase:
    """A reasoning evaluation test case."""

    prompt: str
    expected_steps: list[str]  # Keywords/phrases expected in reasoning
    expected_conclusion: str  # Expected final answer/conclusion
    case_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result of a reasoning evaluation."""

    case_id: str
    prompt: str
    response: str
    steps_found: int
    steps_expected: int
    missing_steps: list[str]
    has_logical_error: bool
    step_coherence_rate: float  # Fraction of expected steps found
    logical_error_rate: float  # 1.0 if error, 0.0 if not


class ReasoningEval:
    """Evaluate reasoning chain quality."""

    def __init__(self, cases: list[ReasoningCase]):
        self.cases = cases

    def run(self, provider: LLMProvider) -> list[ReasoningResult]:
        """Run evaluation on all cases."""
        results = []
        for case in self.cases:
            result = self._evaluate_case(provider, case)
            results.append(result)
        return results

    def _evaluate_case(self, provider: LLMProvider, case: ReasoningCase) -> ReasoningResult:
        """Evaluate a single reasoning case."""
        from llm_eval.providers.base import Message

        response = provider.complete(
            [Message(role="user", content=case.prompt)],
            temperature=0.0,
        ).content

        # Extract steps from response
        steps_found = self._extract_steps(response)
        steps_expected = len(case.expected_steps)

        # Check which expected steps are present
        missing_steps = []
        found_count = 0
        for expected_step in case.expected_steps:
            if self._step_present(expected_step, response):
                found_count += 1
            else:
                missing_steps.append(expected_step)

        # Check for logical errors (conclusion mismatch)
        has_logical_error = not self._conclusion_matches(
            response, case.expected_conclusion
        )

        step_coherence_rate = found_count / steps_expected if steps_expected > 0 else 0.0
        logical_error_rate = 1.0 if has_logical_error else 0.0

        return ReasoningResult(
            case_id=case.case_id or case.prompt[:50],
            prompt=case.prompt,
            response=response,
            steps_found=len(steps_found),
            steps_expected=steps_expected,
            missing_steps=missing_steps,
            has_logical_error=has_logical_error,
            step_coherence_rate=step_coherence_rate,
            logical_error_rate=logical_error_rate,
        )

    def _extract_steps(self, response: str) -> list[ReasoningStep]:
        """Extract reasoning steps from response."""
        steps = []
        # Match numbered steps like "1. ", "Step 1:", etc.
        patterns = [
            r"(?:step\s*)?(\d+)[\.\):]\s*(.+?)(?=(?:step\s*)?\d+[\.\):]|$)",
            r"(\d+)\.\s*(.+?)(?=\d+\.|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                step_num = int(match[0])
                content = match[1].strip()
                if content:
                    steps.append(ReasoningStep(content=content, step_number=step_num))

        return sorted(steps, key=lambda s: s.step_number)

    def _step_present(self, expected: str, response: str) -> bool:
        """Check if expected step keyword/phrase is in response."""
        return expected.lower() in response.lower()

    def _conclusion_matches(self, response: str, expected: str) -> bool:
        """Check if the conclusion matches expected."""
        # Simple check: expected value in response
        return expected.lower() in response.lower()

    def calculate_metrics(self, results: list[ReasoningResult]) -> dict[str, Any]:
        """Calculate aggregate metrics."""
        if not results:
            return {
                "avg_step_coherence_rate": 0.0,
                "avg_logical_error_rate": 0.0,
                "total_cases": 0,
            }

        total_coherence = sum(r.step_coherence_rate for r in results)
        total_errors = sum(r.logical_error_rate for r in results)
        n = len(results)

        return {
            "avg_step_coherence_rate": total_coherence / n,
            "avg_logical_error_rate": total_errors / n,
            "total_cases": n,
            "perfect_reasoning_rate": sum(
                1 for r in results if r.step_coherence_rate == 1.0 and not r.has_logical_error
            ) / n,
        }

    @staticmethod
    def default_cases() -> list[ReasoningCase]:
        """Default reasoning test cases."""
        return [
            ReasoningCase(
                prompt="Solve: 15 + 27. Show your reasoning.",
                expected_steps=["add", "calculate", "result"],
                expected_conclusion="42",
                case_id="addition-1",
            ),
            ReasoningCase(
                prompt="If a train travels 60 mph for 2.5 hours, how far does it go? Show steps.",
                expected_steps=["multiply", "distance", "time"],
                expected_conclusion="150",
                case_id="distance-1",
            ),
            ReasoningCase(
                prompt="Is 97 a prime number? Explain your reasoning.",
                expected_steps=["divisible", "check", "factor"],
                expected_conclusion="yes",
                case_id="prime-1",
            ),
            ReasoningCase(
                prompt="A rectangle has width 5 and length 8. What is the area? Show your work.",
                expected_steps=["multiply", "area", "5", "8"],
                expected_conclusion="40",
                case_id="area-1",
            ),
            ReasoningCase(
                prompt="If x + 7 = 15, what is x? Show your steps.",
                expected_steps=["subtract", "isolate", "7", "15"],
                expected_conclusion="8",
                case_id="equation-1",
            ),
        ]
