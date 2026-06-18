"""Tests for remaining LLM providers (Zen, Cohere, vLLM, llama.cpp, etc.)."""

from unittest.mock import MagicMock, patch

import pytest


class TestZenProvider:
    """Tests for Zen AI provider."""

    def test_init(self):
        from providers import ZenProvider

        with patch.dict("os.environ", {"ZEN_API_KEY": "test-key"}):
            provider = ZenProvider()
            assert provider.base_url == "https://api.zen-ai.com/v1"

    def test_model_mapping(self):
        from providers import ZenProvider

        provider = ZenProvider(api_key="test-zen-key-12345")  # gitleaks:allow

        with patch("providers.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Zen response"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            response = provider.complete(
                [{"role": "user", "content": "Hi"}],
                model="gpt-4o",
            )
            assert response.content == "Zen response"
            assert response.provider == "zen"

    def test_missing_api_key(self):
        from providers import ZenProvider

        with patch.dict("os.environ", clear=True):
            provider = ZenProvider()
            with pytest.raises(ValueError, match="ZEN_API_KEY"):
                provider.complete([{"role": "user", "content": "Hi"}], model="gpt-4o")


class TestMistralProvider:
    """Tests for Mistral AI provider."""

    def test_init(self):
        from providers import MistralProvider

        provider = MistralProvider()
        assert provider.base_url == "https://api.mistral.ai/v1"

    def test_missing_api_key(self):
        from providers import MistralProvider

        with patch.dict("os.environ", clear=True):
            provider = MistralProvider()
            with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                provider._get_client()


class TestOllamaProvider:
    """Tests for Ollama local provider."""

    def test_custom_base_url(self):
        from providers import OllamaProvider

        provider = OllamaProvider(base_url="http://custom:11434")
        assert provider.base_url == "http://custom:11434"

    def test_list_models(self):
        from providers import OllamaProvider

        provider = OllamaProvider()
        with patch("providers.requests.get") as mock_get:
            mock_get.return_value.json.return_value = {
                "models": [{"name": "llama3"}, {"name": "mistral"}]
            }
            models = provider.list_models()
        assert "llama3" in models
        assert len(models) == 2

    def test_list_models_error(self):
        from providers import OllamaProvider

        provider = OllamaProvider()
        with patch("providers.requests.get") as mock_get:
            mock_get.side_effect = Exception("refused")
            models = provider.list_models()
        assert models == []

    def test_complete(self):
        from providers import OllamaProvider

        provider = OllamaProvider()
        with patch("providers.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "message": {"content": "ollama reply"},
                "done": True,
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            response = provider.complete(
                [{"role": "user", "content": "Hi"}],
                model="llama3",
            )
            assert response.content == "ollama reply"
            assert response.provider == "ollama"


class TestLMStudioProvider:
    """Tests for LM Studio provider."""

    def test_custom_url(self):
        from providers import LMStudioProvider

        provider = LMStudioProvider(base_url="http://localhost:1234/v1")
        assert provider.base_url == "http://localhost:1234/v1"


class TestAzureOpenAIProvider:
    """Tests for Azure OpenAI provider."""

    def test_init(self):
        from providers import AzureOpenAIProvider

        provider = AzureOpenAIProvider()
        assert provider is not None

    def test_missing_api_key(self):
        from providers import AzureOpenAIProvider

        with patch.dict("os.environ", clear=True):
            provider = AzureOpenAIProvider()
            with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
                provider._get_client()


class TestGoogleVertexProvider:
    """Tests for Google Vertex AI provider."""

    def test_missing_project(self):
        from providers import GoogleVertexProvider

        with patch.dict("os.environ", clear=True):
            provider = GoogleVertexProvider()
            with pytest.raises((ValueError, ImportError)):
                provider.complete(
                    [{"role": "user", "content": "Hi"}],
                    model="gemini-1.5-pro",
                )


class TestAnthropicConverse:
    """Tests for Anthropic provider edge cases."""

    def test_empty_response_content(self):
        from providers import AnthropicProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider()

        mock = MagicMock()
        mock.messages.create.return_value = MagicMock(
            content=[MagicMock(text="")],
            usage=MagicMock(input_tokens=0, output_tokens=0),
            stop_reason="stop",
        )
        with patch.object(provider, "_get_client", return_value=mock):
            response = provider.complete(
                [{"role": "user", "content": ""}],
                model="claude-3-5-sonnet-20241022",
            )
        assert response.content == ""

    def test_longer_context_model(self):
        from providers import AnthropicProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            provider = AnthropicProvider()

        info = provider.get_model_info("claude-3-5-sonnet-20241022")
        assert info.context_length == 200000
