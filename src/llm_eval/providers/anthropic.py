"""Anthropic provider implementation."""

import os
from typing import Any

import anthropic

from llm_eval.providers.base import CompletionResult, Message


class AnthropicProvider:
    """Anthropic API provider."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str | None = None) -> None:
        """Initialize Anthropic provider."""
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> CompletionResult:
        """Complete using Anthropic API."""
        # Anthropic requires at least one user message
        if not messages or messages[0].role != "user":
            messages = [Message(role="user", content="\n".join(m.content for m in messages))]

        system_prompt = ""
        filtered_messages = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                filtered_messages.append({"role": m.role, "content": m.content})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        if json_mode:
            # Anthropic doesn't have native JSON mode; prepend instruction
            if filtered_messages:
                filtered_messages[0]["content"] = (
                    "Respond with valid JSON only. " + filtered_messages[0]["content"]
                )

        response = self._client.messages.create(**kwargs)

        return CompletionResult(
            content=response.content[0].text,
            model=response.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model
