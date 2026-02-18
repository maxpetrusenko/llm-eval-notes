"""Tests for provider implementations."""


from llm_eval.providers.anthropic import AnthropicProvider
from llm_eval.providers.base import CompletionResult, Message, MockProvider
from llm_eval.providers.openai import OpenAIProvider


def test_mock_provider() -> None:
    """Test mock provider."""
    provider = MockProvider()
    result = provider.complete([Message(role="user", content="test")])

    assert result.content == "Mock response"
    assert result.model == "mock-model"
    assert result.tokens_used == 10


def test_mock_provider_with_custom_responses() -> None:
    """Test mock provider with custom responses."""
    provider = MockProvider(
        responses={"hello": "custom response"},
        default_response="fallback",
    )

    result1 = provider.complete([Message(role="user", content="hello")])
    assert result1.content == "custom response"

    result2 = provider.complete([Message(role="user", content="unknown")])
    assert result2.content == "fallback"


def test_provider_protocol() -> None:
    """Test that providers satisfy the protocol."""
    mock = MockProvider()

    # Check they have required attributes
    assert hasattr(mock, "complete")
    assert hasattr(mock, "name")
    assert hasattr(mock, "model")

    assert mock.name == "mock"


def test_message_model() -> None:
    """Test Message model."""
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_completion_result_model() -> None:
    """Test CompletionResult model."""
    result = CompletionResult(
        content="test response",
        model="gpt-4o",
        tokens_used=100,
        cached_tokens=10,
    )

    assert result.content == "test response"
    assert result.model == "gpt-4o"
    assert result.tokens_used == 100
    assert result.cached_tokens == 10


def test_anthropic_provider_init() -> None:
    """Test AnthropicProvider initialization."""
    provider = AnthropicProvider(model="claude-3-5-sonnet-20241022", api_key="fake-key")
    assert provider.model == "claude-3-5-sonnet-20241022"
    assert provider.name == "anthropic"


def test_openai_provider_init() -> None:
    """Test OpenAIProvider initialization."""
    provider = OpenAIProvider(model="gpt-4o", api_key="fake-key")
    assert provider.model == "gpt-4o"
    assert provider.name == "openai"
