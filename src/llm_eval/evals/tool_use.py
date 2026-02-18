"""Tool-use evaluation tests."""

import json
from typing import Any

import pydantic

from llm_eval.providers.base import LLMProvider, Message


class Tool(pydantic.BaseModel):
    """A tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolUseCase(pydantic.BaseModel):
    """A tool-use test case."""

    id: str
    task: str
    expected_tool: str
    expected_params: dict[str, Any]
    available_tools: list[Tool]
    tolerance: str = "exact"  # "exact" or "fuzzy"


class ToolUseResult(pydantic.BaseModel):
    """Result of a tool-use test."""

    case_id: str
    raw_response: str
    tool_selected: str | None
    parameters_extracted: dict[str, Any] | None
    correct_tool: bool
    correct_params: bool
    parse_error: str | None = None


class ToolUseEval(pydantic.BaseModel):
    """Tool-use evaluation runner."""

    cases: list[ToolUseCase]

    def run(self, provider: LLMProvider) -> list[ToolUseResult]:
        """Run all tool-use tests."""
        results = []
        for case in self.cases:
            result = self._run_case(case, provider)
            results.append(result)
        return results

    def _run_case(self, case: ToolUseCase, provider: LLMProvider) -> ToolUseResult:
        """Run a single test case."""
        tools_schema = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in case.available_tools
        ]

        system_prompt = f"""You are a helpful assistant with access to tools.
Available tools:
{json.dumps(tools_schema, indent=2)}

When you need to use a tool, respond with a JSON object:
{{"tool": "tool_name", "parameters": {{...}}}}

If no tool is needed, respond with plain text."""

        response = provider.complete(
            [
                Message(role="system", content=system_prompt),
                Message(role="user", content=case.task),
            ],
            temperature=0.0,
            json_mode=False,
        )

        # Try to parse tool use
        try:
            # Look for JSON in response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.removeprefix("```json").removesuffix("```").strip()
            elif content.startswith("```"):
                content = content.removeprefix("```").removesuffix("```").strip()

            tool_call = json.loads(content)
            tool_selected = tool_call.get("tool")
            params = tool_call.get("parameters", {})

            correct_tool = tool_selected == case.expected_tool
            correct_params = params == case.expected_params

            return ToolUseResult(
                case_id=case.id,
                raw_response=response.content,
                tool_selected=tool_selected,
                parameters_extracted=params,
                correct_tool=correct_tool,
                correct_params=correct_params,
            )
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            return ToolUseResult(
                case_id=case.id,
                raw_response=response.content,
                tool_selected=None,
                parameters_extracted=None,
                correct_tool=False,
                correct_params=False,
                parse_error=str(e),
            )

    def calculate_metrics(self, results: list[ToolUseResult]) -> dict[str, Any]:
        """Calculate aggregate metrics."""
        if not results:
            return {}

        total = len(results)
        valid_tool_calls = sum(1 for r in results if r.tool_selected is not None)

        return {
            "total_cases": total,
            "tool_selection_accuracy": sum(r.correct_tool for r in results) / total,
            "parameter_accuracy": sum(r.correct_params for r in results) / total,
            "parse_success_rate": valid_tool_calls / total,
            "both_correct": sum(r.correct_tool and r.correct_params for r in results) / total,
        }


# Test tools
WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
        },
        "required": ["location"],
    },
)

SEARCH_TOOL = Tool(
    name="search",
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
)

CALCULATOR_TOOL = Tool(
    name="calculate",
    description="Perform a calculation",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression"},
        },
        "required": ["expression"],
    },
)

DEFAULT_CASES = [
    ToolUseCase(
        id="weather-simple",
        task="What's the weather in Tokyo?",
        expected_tool="get_weather",
        expected_params={"location": "Tokyo"},
        available_tools=[WEATHER_TOOL, SEARCH_TOOL],
    ),
    ToolUseCase(
        id="weather-with-unit",
        task="Check the weather in Paris in fahrenheit",
        expected_tool="get_weather",
        expected_params={"location": "Paris", "unit": "fahrenheit"},
        available_tools=[WEATHER_TOOL, SEARCH_TOOL],
    ),
    ToolUseCase(
        id="search-not-weather",
        task="Search for information about quantum computing",
        expected_tool="search",
        expected_params={"query": "quantum computing"},
        available_tools=[WEATHER_TOOL, SEARCH_TOOL],
    ),
    ToolUseCase(
        id="calculator",
        task="What is 257 times 843?",
        expected_tool="calculate",
        expected_params={"expression": "257 * 843"},
        available_tools=[CALCULATOR_TOOL],
    ),
    ToolUseCase(
        id="ambiguous-query",
        task="Tell me about Paris",
        expected_tool="search",  # Could be weather, but search is more general
        expected_params={"query": "Paris"},
        available_tools=[WEATHER_TOOL, SEARCH_TOOL],
    ),
]
