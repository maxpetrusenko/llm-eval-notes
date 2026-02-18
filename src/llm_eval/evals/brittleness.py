"""Prompt brittleness evaluation tests."""

import pydantic

from llm_eval.providers.base import LLMProvider, Message


class PromptVariation(pydantic.BaseModel):
    """A single prompt variation."""

    text: str
    expected_answer: str | None = None


class BrittlenessScenario(pydantic.BaseModel):
    """A brittleness test scenario with multiple prompt variations."""

    id: str
    name: str
    variations: list[PromptVariation]
    tolerance: str = "fuzzy"  # "exact" or "fuzzy"

    # For fuzzy matching
    keywords: list[str] = []


class BrittlenessResult(pydantic.BaseModel):
    """Result of a brittleness test."""

    scenario_id: str
    responses: list[str]
    consistent: bool
    consistency_rate: float
    refused_count: int
    unique_answers: int
    keywords_present_rate: float


class BrittlenessEval(pydantic.BaseModel):
    """Prompt brittleness evaluation runner."""

    scenarios: list[BrittlenessScenario]

    def run(self, provider: LLMProvider) -> list[BrittlenessResult]:
        """Run all brittleness tests."""
        results = []
        for scenario in self.scenarios:
            result = self._run_scenario(scenario, provider)
            results.append(result)
        return results

    def _run_scenario(self, scenario: BrittlenessScenario, provider: LLMProvider) -> BrittlenessResult:
        """Run a single scenario."""
        responses = []
        refused_count = 0

        for variation in scenario.variations:
            response = provider.complete(
                [Message(role="user", content=variation.text)],
                temperature=0.0,
            )
            content = response.content.strip()
            responses.append(content)

            # Check for refusal
            if self._is_refusal(content):
                refused_count += 1

        # Normalize for comparison
        normalized = [self._normalize(r) for r in responses]

        # Count unique answers
        unique_answers = len(set(normalized))

        # Check consistency
        if scenario.tolerance == "exact":
            consistent = unique_answers == 1
        else:
            # Fuzzy: check if all contain expected keywords
            consistent = all(
                any(k.lower() in r.lower() for k in scenario.keywords)
                for r in responses
            )

        # Calculate consistency rate
        consistency_rate = 1.0 if consistent else (len(responses) - unique_answers + 1) / len(responses)

        # Keywords present rate
        if scenario.keywords:
            keywords_present = sum(
                sum(1 for k in scenario.keywords if k.lower() in r.lower())
                for r in responses
            )
            keywords_present_rate = keywords_present / (len(responses) * len(scenario.keywords))
        else:
            keywords_present_rate = 0.0

        return BrittlenessResult(
            scenario_id=scenario.id,
            responses=responses,
            consistent=consistent,
            consistency_rate=consistency_rate,
            refused_count=refused_count,
            unique_answers=unique_answers,
            keywords_present_rate=keywords_present_rate,
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

    def _is_refusal(self, text: str) -> bool:
        """Check if response is a refusal."""
        refusal_phrases = ["i don't know", "not sure", "cannot answer", "unable"]
        return any(phrase in text.lower() for phrase in refusal_phrases)

    def calculate_metrics(self, results: list[BrittlenessResult]) -> dict[str, float]:
        """Calculate aggregate metrics."""
        if not results:
            return {}

        total = len(results)
        return {
            "total_scenarios": total,
            "consistent_scenarios": sum(r.consistent for r in results) / total,
            "avg_consistency_rate": sum(r.consistency_rate for r in results) / total,
            "avg_unique_answers": sum(r.unique_answers for r in results) / total,
            "total_refusals": sum(r.refused_count for r in results),
        }


_default_scenarios = [
    BrittlenessScenario(
        id="capital-of-france",
        name="Capital of France",
        variations=[
            PromptVariation(text="What is the capital of France?"),
            PromptVariation(text="France's capital is...?"),
            PromptVariation(text="Name the capital city of France."),
            PromptVariation(text="Tell me: what's the capital of France?"),
            PromptVariation(text="The capital of France is what?"),
        ],
        tolerance="fuzzy",
        keywords=["paris"],
    ),
    BrittlenessScenario(
        id="simple-math",
        name="Simple Math",
        variations=[
            PromptVariation(text="What is 7 + 5?"),
            PromptVariation(text="7 plus 5 equals?"),
            PromptVariation(text="Calculate: 7 + 5"),
            PromptVariation(text="Seven plus five is...?"),
        ],
        tolerance="exact",
        keywords=["12"],
    ),
    BrittlenessScenario(
        id="word-meaning",
        name="Word Meaning",
        variations=[
            PromptVariation(text="What does 'ephemeral' mean?"),
            PromptVariation(text="Define: ephemeral"),
            PromptVariation(text="The definition of ephemeral is?"),
            PromptVariation(text="Explain the word 'ephemeral'."),
        ],
        tolerance="fuzzy",
        keywords=["transient", "fleeting", "short-lived", "temporary"],
    ),
    BrittlenessScenario(
        id="one-word-change",
        name="One Word Change Test",
        variations=[
            PromptVariation(text="The cat sat on the mat. What animal sat?", expected_answer="cat"),
            PromptVariation(text="The cat sat on the mat. What did the cat sit on?", expected_answer="mat"),
            PromptVariation(text="The dog sat on the mat. What animal sat?", expected_answer="dog"),
            PromptVariation(text="The cat sat on the rug. What did the cat sit on?", expected_answer="rug"),
        ],
        tolerance="fuzzy",
        keywords=["cat", "dog", "mat", "rug"],
    ),
    BrittlenessScenario(
        id="temperature-test",
        name="Temperature Phrasing",
        variations=[
            PromptVariation(text="Is 100 degrees Celsius hot or cold?"),
            PromptVariation(text="100°C: hot or cold?"),
            PromptVariation(text="Would you describe 100 degrees Celsius as hot or cold?"),
        ],
        tolerance="fuzzy",
        keywords=["hot", "boiling"],
    ),
]

DEFAULT_SCENARIOS = _default_scenarios
