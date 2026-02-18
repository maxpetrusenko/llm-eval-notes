"""OpenAI provider implementation."""

import os
from typing import Any

import openai

from llm_eval.providers.base import CompletionResult, Message


class OpenAIProvider:
    """OpenAI API provider."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        """Initialize OpenAI provider."""
        self._model = model
        self._client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> CompletionResult:
        """Complete using OpenAI API."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        return CompletionResult(
            content=response.choices[0].message.content or "",
            model=response.model,
            tokens_used=response.usage.total_tokens if response.usage else None,
            cached_tokens=getattr(response.usage, "prompt_cache_hit_tokens", None) if response.usage else None,
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model
