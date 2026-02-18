"""Base provider protocol and types."""

from typing import Protocol

import pydantic


class Message(pydantic.BaseModel):
    """A chat message."""

    role: str
    content: str


class CompletionResult(pydantic.BaseModel):
    """Result of an LLM completion."""

    content: str
    model: str
    tokens_used: int | None = None
    cached_tokens: int | None = None


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> CompletionResult:
        """Complete a chat conversation."""
        ...

    @property
    def name(self) -> str:
        """Provider name."""
        ...

    @property
    def model(self) -> str:
        """Model identifier."""
        ...


class MockProvider:
    """Mock provider for testing."""

    def __init__(
        self,
        *,
        name: str = "mock",
        model: str = "mock-model",
        responses: dict[str, str] | None = None,
        default_response: str = "Mock response",
    ) -> None:
        """Initialize mock provider."""
        self._name = name
        self._model = model
        self._responses = responses or {}
        self._default_response = default_response

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> CompletionResult:
        """Return mock response."""
        prompt = "\n".join(m.content for m in messages)
        if prompt in self._responses:
            content = self._responses[prompt]
        else:
            content = self._default_response

        return CompletionResult(content=content, model=self._model, tokens_used=10)

    @property
    def name(self) -> str:
        """Provider name."""
        return self._name

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model
