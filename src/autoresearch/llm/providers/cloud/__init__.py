"""
Cloud LLM providers - Package Scaffold

This package was extracted from the monolithic _cloud.py (556 lines) to improve
module depth and testability. All public provider classes are re-exported here
for backward compatibility so
``from autoresearch.llm.providers.cloud import X`` continues to work.
"""

from .anthropic import AnthropicProvider
from .mistral import MistralProvider
from .openai_compat import (
    AzureOpenAIProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from .vertex import GoogleVertexProvider
from .zen import ZenProvider

__all__ = [
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "GoogleVertexProvider",
    "MistralProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ZenProvider",
]
