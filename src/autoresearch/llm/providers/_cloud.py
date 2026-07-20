"""
Cloud LLM provider implementations.

This module is now a backward-compatibility shim. The provider classes were
extracted into the ``cloud/`` sub-package (see ``cloud/__init__.py``) to
improve module depth and testability. All public symbols are re-exported here
so ``from autoresearch.llm.providers._cloud import X`` continues to work.
"""

from .cloud import (
    AnthropicProvider,
    AzureOpenAIProvider,
    GoogleVertexProvider,
    MistralProvider,
    OpenAIProvider,
    OpenRouterProvider,
    ZenProvider,
)

__all__ = [
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "GoogleVertexProvider",
    "MistralProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ZenProvider",
]
