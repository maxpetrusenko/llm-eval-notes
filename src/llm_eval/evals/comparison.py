"""Cross-model comparison framework."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pydantic

from llm_eval.evals.brittleness import DEFAULT_SCENARIOS, BrittlenessEval
from llm_eval.evals.hallucination import DEFAULT_CASES, HallucinationEval
from llm_eval.evals.structured import DEFAULT_CASES as STRUCTURED_DEFAULT_CASES
from llm_eval.evals.structured import StructuredOutputEval
from llm_eval.providers.base import LLMProvider


class ComparisonReport(pydantic.BaseModel):
    """Aggregated comparison report."""

    timestamp: str
    runs: dict[str, dict[str, Any]]
    summary: dict[str, Any]

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# LLM Eval Results",
            f"**Generated:** {self.timestamp}\n",
            "## Summary\n",
        ]

        # Hallucination table
        if "hallucination" in self.summary:
            lines.append("### Hallucination Tests")
            lines.append("| Model | Exact Match | Safe Rate | Hallucination Rate | Refusal Rate |")
            lines.append("|-------|-------------|-----------|-------------------|--------------|")
            for model, metrics in self.summary["hallucination"].items():
                lines.append(
                    f"| {model} | {metrics['exact_match_rate']:.1%} | "
                    f"{metrics['safe_rate']:.1%} | {metrics['hallucination_rate']:.1%} | "
                    f"{metrics['refusal_rate']:.1%} |"
                )
            lines.append("")

        # Brittleness table
        if "brittleness" in self.summary:
            lines.append("### Prompt Brittleness")
            lines.append("| Model | Consistency Rate | Avg Unique Answers | Refusals |")
            lines.append("|-------|-----------------|---------------------|----------|")
            for model, metrics in self.summary["brittleness"].items():
                lines.append(
                    f"| {model} | {metrics['avg_consistency_rate']:.1%} | "
                    f"{metrics['avg_unique_answers']:.1f} | {metrics['total_refusals']} |"
                )
            lines.append("")

        # Structured output table
        if "structured" in self.summary:
            lines.append("### Structured Output")
            lines.append("| Model | Valid JSON | Schema Valid | Retry Success |")
            lines.append("|-------|------------|--------------|---------------|")
            for model, metrics in self.summary["structured"].items():
                lines.append(
                    f"| {model} | {metrics['valid_json_rate']:.1%} | "
                    f"{metrics['schema_valid_rate']:.1%} | {metrics['retry_success_rate']:.1%} |"
                )
            lines.append("")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

        # Save markdown
        md_path = path.with_suffix(".md")
        with open(md_path, "w") as f:
            f.write(self.to_markdown())


class ComparisonRunner(pydantic.BaseModel):
    """Run and compare evals across models."""

    hallucination: HallucinationEval | None = None
    brittleness: BrittlenessEval | None = None
    structured: StructuredOutputEval | None = None

    def run_all(self, providers: list[LLMProvider]) -> ComparisonReport:
        """Run all evals across all providers."""
        runs: dict[str, dict[str, Any]] = {}
        summary: dict[str, dict[str, dict[str, Any]]] = {
            "hallucination": {},
            "brittleness": {},
            "structured": {},
        }

        for provider in providers:
            model_key = f"{provider.name}/{provider.model}"
            runs[model_key] = {}

            # Hallucination
            if self.hallucination:
                h_results = self.hallucination.run(provider)
                h_metrics = self.hallucination.calculate_metrics(h_results)
                runs[model_key]["hallucination"] = {
                    "results": [r.model_dump(mode="json") for r in h_results],
                    "metrics": h_metrics,
                }
                summary["hallucination"][model_key] = h_metrics

            # Brittleness
            if self.brittleness:
                b_results = self.brittleness.run(provider)
                b_metrics = self.brittleness.calculate_metrics(b_results)
                runs[model_key]["brittleness"] = {
                    "results": [r.model_dump(mode="json") for r in b_results],
                    "metrics": b_metrics,
                }
                summary["brittleness"][model_key] = b_metrics

            # Structured
            if self.structured:
                s_results = self.structured.run(provider)
                s_metrics = self.structured.calculate_metrics(s_results)
                runs[model_key]["structured"] = {
                    "results": [r.model_dump(mode="json") for r in s_results],
                    "metrics": s_metrics,
                }
                summary["structured"][model_key] = s_metrics

        return ComparisonReport(
            timestamp=datetime.now().isoformat(),
            runs=runs,
            summary=summary,
        )

    @classmethod
    def with_defaults(cls) -> "ComparisonRunner":
        """Create runner with default test cases."""
        return cls(
            hallucination=HallucinationEval(cases=DEFAULT_CASES),
            brittleness=BrittlenessEval(scenarios=DEFAULT_SCENARIOS),
            structured=StructuredOutputEval(cases=STRUCTURED_DEFAULT_CASES),
        )
