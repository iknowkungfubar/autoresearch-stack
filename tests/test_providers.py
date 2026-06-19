"""Tests for LLM providers with mocked HTTP/API clients."""

from unittest.mock import MagicMock, patch

import pytest

# ── Helper fixtures ──────────────────────────────────────────────


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client response."""
    mock = MagicMock()
    mock.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Hello from Claude")],
        usage=MagicMock(input_tokens=10, output_tokens=20),
        stop_reason="end_turn",
    )
    return mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI chat completion response."""
    mock = MagicMock()
    choice = MagicMock()
    choice.message.content = "Hello from GPT"
    choice.finish_reason = "stop"
    mock.chat.completions.create.return_value = MagicMock(
        choices=[choice],
        usage=MagicMock(prompt_tokens=10, completion_tokens=20),
    )
    return mock


@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"},
    ]


# ── AnthropicProvider ────────────────────────────────────────────


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_complete(self, mock_anthropic_client, sample_messages):
        from autoresearch.llm.providers import AnthropicProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test123"}):
            provider = AnthropicProvider()

        with patch.object(provider, "_get_client", return_value=mock_anthropic_client):
            response = provider.complete(
                sample_messages, model="claude-3-5-sonnet-20241022"
            )

        assert response.content == "Hello from Claude"
        assert response.provider == "anthropic"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.usage["input_tokens"] == 10
        assert response.usage["output_tokens"] == 20
        assert response.latency_ms > 0

    def test_chat(self, mock_anthropic_client, sample_messages):
        from autoresearch.llm.providers import AnthropicProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test123"}):
            provider = AnthropicProvider()

        with patch.object(provider, "_get_client", return_value=mock_anthropic_client):
            response = provider.chat(
                sample_messages, model="claude-3-5-sonnet-20241022"
            )

        assert response.content == "Hello from Claude"

    def test_missing_api_key(self):
        from autoresearch.llm.providers import AnthropicProvider

        with patch.dict("os.environ", clear=True):
            provider = AnthropicProvider()
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                provider._get_client()

    def test_system_message_extraction(self, mock_anthropic_client):
        """Verify system message is extracted and not passed as a regular message."""
        from autoresearch.llm.providers import AnthropicProvider

        messages = [
            {"role": "system", "content": "You are Claude."},
            {"role": "user", "content": "Hi"},
        ]

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test123"}):
            provider = AnthropicProvider()

        with patch.object(provider, "_get_client", return_value=mock_anthropic_client):
            provider.complete(messages, model="claude-3-5-sonnet-20241022")

        # Verify API was called correctly
        mock_anthropic_client.messages.create.assert_called_once()
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are Claude."
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    def test_api_error(self):
        from autoresearch.llm.providers import AnthropicProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test123"}):
            provider = AnthropicProvider()

        mock = MagicMock()
        mock.messages.create.side_effect = Exception("API timeout")
        with patch.object(provider, "_get_client", return_value=mock):
            with pytest.raises(RuntimeError, match="Anthropic API error"):
                provider.complete(
                    [{"role": "user", "content": "Hi"}],
                    model="claude-3-5-sonnet-20241022",
                )

    def test_get_model_info(self):
        from autoresearch.llm.providers import AnthropicProvider

        provider = AnthropicProvider()
        info = provider.get_model_info("claude-3-5-sonnet-20241022")
        assert info is not None
        assert info.provider == "anthropic"
        assert info.context_length == 200000

        # Unknown model returns None
        assert provider.get_model_info("nonexistent-model") is None


# ── OpenAIProvider ───────────────────────────────────────────────


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_complete(self, mock_openai_client, sample_messages):
        from autoresearch.llm.providers import OpenAIProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            provider = OpenAIProvider()

        with patch.object(provider, "_get_client", return_value=mock_openai_client):
            response = provider.complete(sample_messages, model="gpt-4o")

        assert response.content == "Hello from GPT"
        assert response.provider == "openai"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20

    def test_chat(self, mock_openai_client, sample_messages):
        from autoresearch.llm.providers import OpenAIProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            provider = OpenAIProvider()

        with patch.object(provider, "_get_client", return_value=mock_openai_client):
            response = provider.chat(sample_messages, model="gpt-4o")

        assert response.content == "Hello from GPT"

    def test_missing_api_key(self):
        from autoresearch.llm.providers import OpenAIProvider

        with patch.dict("os.environ", clear=True):
            provider = OpenAIProvider()
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                provider._get_client()

    def test_api_error(self):
        from autoresearch.llm.providers import OpenAIProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            provider = OpenAIProvider()

        mock = MagicMock()
        mock.chat.completions.create.side_effect = Exception("Rate limited")
        with patch.object(provider, "_get_client", return_value=mock):
            with pytest.raises(RuntimeError, match="OpenAI API error"):
                provider.complete([{"role": "user", "content": "Hi"}], model="gpt-4o")

    def test_custom_base_url(self):
        from autoresearch.llm.providers import OpenAIProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            provider = OpenAIProvider(base_url="https://custom.example.com/v1")
            assert provider.base_url == "https://custom.example.com/v1"

    def test_null_content(self):
        """Test that null content is handled as empty string."""
        from autoresearch.llm.providers import OpenAIProvider

        mock = MagicMock()
        choice = MagicMock()
        choice.message.content = None
        choice.finish_reason = "stop"
        mock.chat.completions.create.return_value = MagicMock(
            choices=[choice],
            usage=MagicMock(prompt_tokens=0, completion_tokens=0),
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            provider = OpenAIProvider()

        with patch.object(provider, "_get_client", return_value=mock):
            response = provider.complete(
                [{"role": "user", "content": "Hi"}], model="gpt-4o"
            )

        assert response.content == ""


# ── OpenRouterProvider ───────────────────────────────────────────


class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""

    def test_complete(self, mock_openai_client, sample_messages):
        from autoresearch.llm.providers import OpenRouterProvider

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            provider = OpenRouterProvider()

        with patch.object(provider, "_get_client", return_value=mock_openai_client):
            response = provider.complete(sample_messages, model="gpt-4o")

        assert response.provider == "openrouter"
        assert response.content == "Hello from GPT"

    def test_missing_api_key(self):
        from autoresearch.llm.providers import OpenRouterProvider

        with patch.dict("os.environ", clear=True):
            provider = OpenRouterProvider()
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                provider._get_client()

    def test_model_mapping(self, mock_openai_client):
        """Verify model name translation for OpenRouter."""
        from autoresearch.llm.providers import OpenRouterProvider

        messages = [{"role": "user", "content": "Hi"}]
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            provider = OpenRouterProvider()

        with patch.object(provider, "_get_client", return_value=mock_openai_client):
            provider.complete(messages, model="claude-3.5-sonnet")

        call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-3.5-sonnet"
        assert "HTTP-Referer" in call_kwargs["extra_headers"]
        assert "X-Title" in call_kwargs["extra_headers"]

    def test_api_error(self):
        from autoresearch.llm.providers import OpenRouterProvider

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}):
            provider = OpenRouterProvider()

        mock = MagicMock()
        mock.chat.completions.create.side_effect = Exception("Quota exceeded")
        with patch.object(provider, "_get_client", return_value=mock):
            with pytest.raises(RuntimeError, match="OpenRouter API error"):
                provider.complete([{"role": "user", "content": "Hi"}], model="gpt-4o")


# ── LLMProviderFactory ───────────────────────────────────────────


class TestProviderFactory:
    """Tests for LLMProviderFactory."""

    def test_create_string(self):
        from autoresearch.llm.providers import LLMProviderFactory

        provider = LLMProviderFactory.create("openai")
        assert provider.__class__.__name__ == "OpenAIProvider"

    def test_create_from_enum(self):
        from autoresearch.llm.providers import LLMProviderFactory, ProviderType

        provider = LLMProviderFactory.create(ProviderType.ANTHROPIC)
        assert provider.__class__.__name__ == "AnthropicProvider"

    def test_create_invalid(self):
        from autoresearch.llm.providers import LLMProviderFactory

        with pytest.raises((ValueError, KeyError)):
            LLMProviderFactory.create("nonexistent_provider")

    def test_from_config(self):
        from autoresearch.llm.providers import LLMProviderFactory

        config = {
            "provider": "ollama",
            "api_key": None,
            "base_url": "http://localhost:11434",
        }
        provider = LLMProviderFactory.from_config(config)
        assert provider.__class__.__name__ == "OllamaProvider"
        assert provider.base_url == "http://localhost:11434"

    def test_from_config_defaults(self):
        """Test factory with empty config uses openai default."""
        from autoresearch.llm.providers import LLMProviderFactory

        provider = LLMProviderFactory.from_config({})
        assert provider.__class__.__name__ == "OpenAIProvider"


# ── LLMClient ────────────────────────────────────────────────────


class TestLLMClient:
    """Tests for LLMClient."""

    def test_no_provider(self):
        from autoresearch.llm.providers import LLMClient

        client = LLMClient()
        with pytest.raises(RuntimeError, match="No provider configured"):
            client.complete([{"role": "user", "content": "test"}])

    def test_with_provider(self, mock_openai_client, sample_messages):
        from autoresearch.llm.providers import LLMClient, OpenAIProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            provider = OpenAIProvider()
            client = LLMClient(provider=provider)

        with patch.object(provider, "_get_client", return_value=mock_openai_client):
            response = client.complete(sample_messages, model="gpt-4o")

        assert response.content == "Hello from GPT"

    def test_set_provider(self):
        from autoresearch.llm.providers import LLMClient, OpenAIProvider

        client = LLMClient()
        assert client.provider is None
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test123"}):
            client = LLMClient(provider=OpenAIProvider())
        assert client.provider is not None


# ── Local providers ──────────────────────────────────────────────


class TestLocalProviders:
    """Tests for local inference providers."""

    def test_ollama_init(self):
        from autoresearch.llm.providers import OllamaProvider

        provider = OllamaProvider(base_url="http://custom:11434")
        assert provider.base_url == "http://custom:11434"

    def test_ollama_list_models(self):
        """Test list_models with mocked HTTP response."""
        from autoresearch.llm.providers import OllamaProvider

        provider = OllamaProvider()
        with patch("providers.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "models": [{"name": "llama3"}, {"name": "mistral"}]
            }
            models = provider.list_models()
        assert len(models) == 2
        assert "llama3" in models

    def test_ollama_list_models_failure(self):
        """Test list_models handles errors gracefully."""
        from autoresearch.llm.providers import OllamaProvider

        provider = OllamaProvider()
        with patch("providers.requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")
            models = provider.list_models()
        assert models == []

    def test_lmstudio_init(self):
        from autoresearch.llm.providers import LMStudioProvider

        provider = LMStudioProvider(base_url="http://localhost:1234/v1")
        assert provider.base_url == "http://localhost:1234/v1"


# ── MODEL REGISTRY ────────────────────────────────────────────────


class TestModelRegistry:
    """Tests for MODEL_REGISTRY."""

    def test_registry_complete(self):
        from autoresearch.llm.providers import MODEL_REGISTRY

        assert len(MODEL_REGISTRY) >= 10
        assert "claude-3-5-sonnet-20241022" in MODEL_REGISTRY
        assert "gpt-4o" in MODEL_REGISTRY
        assert "gemini-1.5-pro" in MODEL_REGISTRY

    def test_model_info_structure(self):
        from autoresearch.llm.providers import MODEL_REGISTRY, ModelInfo

        for model_id, info in MODEL_REGISTRY.items():
            assert isinstance(info, ModelInfo)
            # Some model IDs differ from their dict key (aliases)
            assert isinstance(info.id, str)
            assert len(info.id) > 0
            assert isinstance(info.name, str)
            assert isinstance(info.provider, str)
            assert info.context_length > 0
            assert info.max_output_tokens > 0


# ── LLMResponse ──────────────────────────────────────────────────


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_construction(self):
        from autoresearch.llm.providers import LLMResponse

        response = LLMResponse(
            content="test",
            model="gpt-4o",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            FinishReason="stop",
            raw_response={"id": "123"},
            latency_ms=150.0,
        )
        assert response.content == "test"
        assert response.usage["completion_tokens"] == 20
        assert response.raw_response == {"id": "123"}

    def test_defaults(self):
        from autoresearch.llm.providers import LLMResponse

        response = LLMResponse(content="", model="test", provider="test")
        assert response.usage == {}
        assert response.FinishReason == "stop"
        assert response.raw_response is None
        assert response.latency_ms == 0.0
