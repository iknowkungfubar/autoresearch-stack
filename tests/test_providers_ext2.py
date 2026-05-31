"""Tests for remaining LLM providers: VLLM, LiteLLM, LlamaCpp, TextGenWebUI."""
import pytest
from unittest.mock import patch, MagicMock

pytest.importorskip = getattr(pytest, "importorskip", lambda x: None)


class TestVLLMProvider:
    """Tests for vLLM provider (OpenAI-compatible API)."""

    def test_init(self):
        from providers import VLLMProvider
        p = VLLMProvider(api_key="sk-test")
        assert p is not None

    def test_provider_type(self):
        from providers import VLLMProvider
        p = VLLMProvider(api_key="sk-test")
        assert p.__class__.__name__ == "VLLMProvider"


class TestLiteLLMProvider:
    """Tests for LiteLLM provider."""

    def test_init(self):
        from providers import LiteLLMProvider
        p = LiteLLMProvider(api_key="sk-test")
        assert p is not None

    def test_provider_type(self):
        from providers import LiteLLMProvider
        p = LiteLLMProvider(api_key="sk-test")
        assert p.__class__.__name__ == "LiteLLMProvider"


class TestLlamaCppProvider:
    """Tests for llama.cpp provider."""

    def test_init(self):
        from providers import LlamaCppProvider
        p = LlamaCppProvider()
        assert p is not None

    def test_provider_type(self):
        from providers import LlamaCppProvider
        p = LlamaCppProvider()
        assert p.__class__.__name__ == "LlamaCppProvider"


class TestTextGenWebUIProvider:
    """Tests for Text Generation WebUI provider."""

    def test_init(self):
        from providers import TextGenWebUIProvider
        p = TextGenWebUIProvider(base_url="http://localhost:7860")
        assert p.base_url == "http://localhost:7860"

    def test_complete(self):
        from providers import TextGenWebUIProvider
        p = TextGenWebUIProvider()
        with patch("providers.requests.post") as mock_post:
            mock = MagicMock()
            mock.json.return_value = {
                "choices": [{"text": "webui reply"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            mock.raise_for_status.return_value = None
            mock_post.return_value = mock
            resp = p.complete([{"role": "user", "content": "Hi"}], model="llama")
        assert resp.content == "webui reply"
        assert resp.provider == "textgen_webui"


class TestProviderFactoryFull:
    """Tests for LLMProviderFactory covering all provider types."""

    def test_all_registered_providers(self):
        from providers import LLMProviderFactory, ProviderType

        registered = [
            ProviderType.OLLAMA,
            ProviderType.LMSTUDIO,
            ProviderType.VLLM,
            ProviderType.LITELLM,
            ProviderType.TEXTGEN_WEBUI,
            ProviderType.LLAMA_CPP,
        ]
        for pt in registered:
            p = LLMProviderFactory.create(pt)
            assert p is not None

    def test_factory_create_vllm(self):
        from providers import LLMProviderFactory, ProviderType
        p = LLMProviderFactory.create(ProviderType.VLLM)
        assert p.__class__.__name__ == "VLLMProvider"

    def test_factory_create_litellm(self):
        from providers import LLMProviderFactory, ProviderType
        p = LLMProviderFactory.create(ProviderType.LITELLM)
        assert p.__class__.__name__ == "LiteLLMProvider"

    def test_factory_create_llamacpp(self):
        from providers import LLMProviderFactory, ProviderType
        p = LLMProviderFactory.create(ProviderType.LLAMA_CPP)
        assert p.__class__.__name__ == "LlamaCppProvider"

    def test_factory_create_textgen(self):
        from providers import LLMProviderFactory, ProviderType
        p = LLMProviderFactory.create(ProviderType.TEXTGEN_WEBUI)
        assert p.__class__.__name__ == "TextGenWebUIProvider"
