"""
Unified LLM Provider Interface - Multi-provider support.

Supports:
- Cloud providers: Anthropic, OpenAI, OpenRouter, Zen, Google Vertex, Azure, Cohere
- Local inference: Ollama, vLLM, LMStudio, llama.cpp, LiteLLM, Text Generation WebUI
- Agentic harnesses: OpenCode, OpenCrew, AgentForge, CrewAI, AutoGen

Phase 7.1: Multi-Provider Support.
"""

from ._cloud import (
    AnthropicProvider,
    AzureOpenAIProvider,
    GoogleVertexProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    ZenProvider,
)
from ._factory import LLMClient, LLMProviderFactory
from ._local import (
    LiteLLMProvider,
    LlamaCppProvider,
    LMStudioProvider,
    OllamaProvider,
    TextGenWebUIProvider,
    VLLMProvider,
)
from ._registry import MODEL_REGISTRY
from ._types import BaseLLMProvider, LLMResponse, ModelInfo, ProviderType

__all__ = [
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "BaseLLMProvider",
    "GoogleVertexProvider",
    "LiteLLMProvider",
    "LlamaCppProvider",
    "LLMClient",
    "LLMProviderFactory",
    "LLMResponse",
    "LMStudioProvider",
    "MistralProvider",
    "ModelInfo",
    "MODEL_REGISTRY",
    "OllamaProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderType",
    "TextGenWebUIProvider",
    "VLLMProvider",
    "ZenProvider",
]
