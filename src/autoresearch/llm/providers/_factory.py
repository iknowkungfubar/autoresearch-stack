"""Provider factory and unified client."""

import logging
import os
from typing import Any, Dict, Optional, Union

from ._cloud import (
    AnthropicProvider,
    AzureOpenAIProvider,
    GoogleVertexProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    ZenProvider,
)
from ._local import (
    LiteLLMProvider,
    LlamaCppProvider,
    LMStudioProvider,
    OllamaProvider,
    TextGenWebUIProvider,
    VLLMProvider,
)
from ._types import BaseLLMProvider, LLMResponse, ProviderType


logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    PROVIDERS = {
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.OPENROUTER: OpenRouterProvider,
        ProviderType.ZEN: ZenProvider,
        ProviderType.AZURE_OPENAI: AzureOpenAIProvider,
        ProviderType.GOOGLE_VERTEX: GoogleVertexProvider,
        ProviderType.MISTRAL: MistralProvider,
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.LMSTUDIO: LMStudioProvider,
        ProviderType.VLLM: VLLMProvider,
        ProviderType.LITELLM: LiteLLMProvider,
        ProviderType.LLAMA_CPP: LlamaCppProvider,
        ProviderType.TEXTGEN_WEBUI: TextGenWebUIProvider,
    }

    @classmethod
    def create(
        cls,
        provider: Union[ProviderType, str],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Create a provider instance."""
        if isinstance(provider, str):
            provider = ProviderType(provider)

        provider_class = cls.PROVIDERS.get(provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider}")

        return provider_class(api_key=api_key, base_url=base_url, **kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseLLMProvider:
        """Create provider from config dict."""
        return cls.create(
            provider=config.get("provider", "openai"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            **config.get("extra_params", {}),
        )


class LLMClient:
    """Unified client for all providers."""

    def __init__(
        self, provider: Optional[BaseLLMProvider] = None, default_model: str = "gpt-4o"
    ):
        self.provider = provider
        self.default_model = default_model
        self.logger = logging.getLogger("LLMClient")

    def complete(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        if not self.provider:
            raise RuntimeError("No provider configured")
        return self.provider.complete(
            messages,
            model or self.default_model,
            temperature,
            max_tokens,
            **kwargs,
        )

    def chat(self, messages: list[dict[str, str]], **kwargs) -> LLMResponse:
        """Chat completion (alias)."""
        return self.complete(messages, **kwargs)

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Create client from environment variables."""
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = LLMProviderFactory.create(ProviderType.ANTHROPIC)
            return cls(provider, default_model="claude-3-5-sonnet-20241022")
        elif os.getenv("OPENAI_API_KEY"):
            provider = LLMProviderFactory.create(ProviderType.OPENAI)
            return cls(provider, default_model="gpt-4o")
        elif os.getenv("OPENROUTER_API_KEY"):
            provider = LLMProviderFactory.create(ProviderType.OPENROUTER)
            return cls(provider, default_model="gpt-4o")
        else:
            raise ValueError("No API key found in environment")
