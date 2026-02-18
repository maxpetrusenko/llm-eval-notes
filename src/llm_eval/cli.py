"""CLI for running evals."""

import os
from typing import Any

import typer

from llm_eval.evals.brittleness import BrittlenessResult
from llm_eval.evals.comparison import ComparisonRunner
from llm_eval.evals.hallucination import HallucinationResult
from llm_eval.evals.structured import StructuredResult
from llm_eval.evals.tool_use import ToolUseResult
from llm_eval.providers.anthropic import AnthropicProvider
from llm_eval.providers.base import LLMProvider, MockProvider
from llm_eval.providers.openai import OpenAIProvider

app = typer.Typer(help="LLM evaluation artifacts")


@app.command()
def run(
    eval_name: str = typer.Argument(..., help="Eval type: hallucination, brittleness, structured, tool-use, all"),
    model: str = typer.Option("gpt-4o", help="Model identifier"),
    provider: str = typer.Option("openai", help="Provider: openai, anthropic, mock"),
    output: str | None = typer.Option(None, help="Output file path"),
) -> None:
    """Run a specific evaluation."""
    # Get provider
    llm = _get_provider(provider, model)

    # Run eval
    runner = ComparisonRunner.with_defaults()

    if eval_name == "all":
        report = runner.run_all([llm])
        if output:
            report.save(output)
            typer.echo(f"Results saved to {output}")
        else:
            typer.echo(report.to_markdown())
    elif eval_name == "hallucination":
        if not runner.hallucination:
            typer.echo("Hallucination eval not configured", err=True)
            raise typer.Exit(1)
        h_results: list[HallucinationResult] = runner.hallucination.run(llm)
        metrics = runner.hallucination.calculate_metrics(h_results)
        _print_metrics("Hallucination", metrics)
    elif eval_name == "brittleness":
        if not runner.brittleness:
            typer.echo("Brittleness eval not configured", err=True)
            raise typer.Exit(1)
        b_results: list[BrittlenessResult] = runner.brittleness.run(llm)
        metrics = runner.brittleness.calculate_metrics(b_results)
        _print_metrics("Prompt Brittleness", metrics)
    elif eval_name == "structured":
        if not runner.structured:
            typer.echo("Structured output eval not configured", err=True)
            raise typer.Exit(1)
        s_results: list[StructuredResult] = runner.structured.run(llm)
        metrics = runner.structured.calculate_metrics(s_results)
        _print_metrics("Structured Output", metrics)
    elif eval_name == "tool-use":
        if not runner.tool_use:
            typer.echo("Tool-use eval not configured", err=True)
            raise typer.Exit(1)
        t_results: list[ToolUseResult] = runner.tool_use.run(llm)
        metrics = runner.tool_use.calculate_metrics(t_results)
        _print_metrics("Tool Use", metrics)
    else:
        typer.echo(f"Unknown eval: {eval_name}", err=True)
        raise typer.Exit(1)


@app.command()
def compare(
    models: str = typer.Option("gpt-4o,claude-3-5-sonnet-20241022", help="Comma-separated models"),
    output: str = typer.Option("experiments/results/comparison", help="Output path"),
) -> None:
    """Compare across multiple models."""
    model_list = [m.strip() for m in models.split(",")]

    providers: list[LLMProvider] = []
    for model in model_list:
        if model.startswith("gpt"):
            providers.append(OpenAIProvider(model=model))
        elif model.startswith("claude"):
            providers.append(AnthropicProvider(model=model))
        else:
            typer.echo(f"Unknown model prefix: {model}", err=True)
            raise typer.Exit(1)

    runner = ComparisonRunner.with_defaults()
    results = runner.run_all(providers)

    results.save(output)
    typer.echo(f"Comparison results saved to {output}.md")
    typer.echo("\n" + results.to_markdown())


@app.command()
def list_evals() -> None:
    """List available evals."""
    typer.echo("Available evals:")
    typer.echo("  hallucination  - Ground truth comparison, hallucination detection")
    typer.echo("  brittleness    - Prompt variation consistency tests")
    typer.echo("  structured     - JSON schema validation tests")
    typer.echo("  tool-use       - Tool selection and argument extraction")
    typer.echo("  all            - Run all evals")


def _get_provider(provider: str, model: str) -> LLMProvider:
    """Get provider instance."""
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            typer.echo("OPENAI_API_KEY not set", err=True)
            raise typer.Exit(1)
        return OpenAIProvider(model=model)
    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            typer.echo("ANTHROPIC_API_KEY not set", err=True)
            raise typer.Exit(1)
        return AnthropicProvider(model=model)
    elif provider == "mock":
        return MockProvider()
    else:
        typer.echo(f"Unknown provider: {provider}", err=True)
        raise typer.Exit(1)


def _print_metrics(name: str, metrics: dict[str, Any]) -> None:
    """Print metrics table."""
    typer.echo(f"\n{name} Results:")
    typer.echo("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            typer.echo(f"  {key}: {value:.2%}" if value < 1 else f"  {key}: {value:.2f}")
        else:
            typer.echo(f"  {key}: {value}")


if __name__ == "__main__":
    app()
