"""Cost tracking for LLM evaluations."""

from dataclasses import dataclass, field
from typing import Any


# Pricing per 1M tokens (as of 2025)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Mock
    "mock-model": {"input": 0.00, "output": 0.00},
}


@dataclass
class CostRecord:
    """Record of tokens and cost for a single API call."""

    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cost_usd: float = 0.0
    eval_type: str = "unknown"
    case_id: str | None = None

    def __post_init__(self):
        if self.cost_usd == 0.0:
            self.cost_usd = self.calculate_cost()

    def calculate_cost(self) -> float:
        """Calculate cost based on model pricing."""
        pricing = MODEL_PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        # Cached tokens are typically 90% cheaper
        cached_discount = (self.cached_tokens / 1_000_000) * pricing["input"] * 0.9
        return input_cost + output_cost - cached_discount


@dataclass
class CostReport:
    """Aggregate cost report for evaluation runs."""

    records: list[CostRecord] = field(default_factory=list)

    def add(self, record: CostRecord) -> None:
        """Add a cost record."""
        self.records.append(record)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_cached_tokens(self) -> int:
        return sum(r.cached_tokens for r in self.records)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def by_model(self) -> dict[str, dict[str, Any]]:
        """Get cost breakdown by model."""
        breakdown = {}
        for record in self.records:
            if record.model not in breakdown:
                breakdown[record.model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "cost_usd": 0.0,
                    "calls": 0,
                }
            breakdown[record.model]["input_tokens"] += record.input_tokens
            breakdown[record.model]["output_tokens"] += record.output_tokens
            breakdown[record.model]["cached_tokens"] += record.cached_tokens
            breakdown[record.model]["cost_usd"] += record.cost_usd
            breakdown[record.model]["calls"] += 1
        return breakdown

    def by_eval_type(self) -> dict[str, dict[str, Any]]:
        """Get cost breakdown by evaluation type."""
        breakdown = {}
        for record in self.records:
            eval_type = record.eval_type
            if eval_type not in breakdown:
                breakdown[eval_type] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "calls": 0,
                }
            breakdown[eval_type]["input_tokens"] += record.input_tokens
            breakdown[eval_type]["output_tokens"] += record.output_tokens
            breakdown[eval_type]["cost_usd"] += record.cost_usd
            breakdown[eval_type]["calls"] += 1
        return breakdown

    def to_markdown(self) -> str:
        """Format report as markdown."""
        lines = ["# Cost Report\n"]
        lines.append("## Summary\n")
        lines.append(f"- **Total Cost:** ${self.total_cost_usd:.4f}")
        lines.append(f"- **Total Tokens:** {self.total_tokens:,}")
        lines.append(f"- **Input Tokens:** {self.total_input_tokens:,}")
        lines.append(f"- **Output Tokens:** {self.total_output_tokens:,}")
        lines.append(f"- **Cached Tokens:** {self.total_cached_tokens:,}")
        lines.append(f"- **API Calls:** {len(self.records)}\n")

        lines.append("## By Model\n")
        lines.append("| Model | Calls | Input | Output | Cost |")
        lines.append("|-------|-------|-------|--------|------|")
        for model, stats in self.by_model().items():
            lines.append(
                f"| {model} | {stats['calls']} | {stats['input_tokens']:,} | "
                f"{stats['output_tokens']:,} | ${stats['cost_usd']:.4f} |"
            )

        lines.append("\n## By Eval Type\n")
        lines.append("| Eval Type | Calls | Input | Output | Cost |")
        lines.append("|-----------|-------|-------|--------|------|")
        for eval_type, stats in self.by_eval_type().items():
            lines.append(
                f"| {eval_type} | {stats['calls']} | {stats['input_tokens']:,} | "
                f"{stats['output_tokens']:,} | ${stats['cost_usd']:.4f} |"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for JSON serialization."""
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_calls": len(self.records),
            "by_model": self.by_model(),
            "by_eval_type": self.by_eval_type(),
            "records": [
                {
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cached_tokens": r.cached_tokens,
                    "cost_usd": r.cost_usd,
                    "eval_type": r.eval_type,
                    "case_id": r.case_id,
                }
                for r in self.records
            ],
        }
