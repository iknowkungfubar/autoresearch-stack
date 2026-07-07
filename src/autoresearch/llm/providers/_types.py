"""Common types and base class for LLM providers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProviderType(Enum):
    """LLM provider types."""

    # Cloud providers
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ZEN = "zen"
    GOOGLE_VERTEX = "google_vertex"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    MISTRAL = "mistral"
    ANTHROPIC_CONVERSE = "anthropic_converse"

    # Local inference
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    LMSTUDIO = "lmstudio"
    LITELLM = "litellm"
    TEXTGEN_WEBUI = "textgen_webui"
    KOBOLD_CPP = "koboldcpp"
    LOCALAI = "localai"

    # Agentic harnesses (orchestrators)
    OPENCODE = "opencode"
    OPENCREW = "opencrew"
    AGENTFORGE = "agentforge"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    LANGCHAIN = "langchain"
    LLAMA_INDEX = "llama_index"


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    name: str
    provider: str
    context_length: int = 128000
    max_output_tokens: int = 8192
    supports_vision: bool = False
    supports_function_calling: bool = False
    pricing_input: float = 0.0
    pricing_output: float = 0.0
    is_local: bool = False


@dataclass
class LLMResponse:
    """Response from LLM."""

    content: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    FinishReason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = None
    latency_ms: float = 0.0


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.extra_params = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion (alias for complete)."""
        pass

    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get model information."""
        from ._registry import MODEL_REGISTRY

        return MODEL_REGISTRY.get(model)
