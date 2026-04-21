"""
Unified LLM Provider Interface - Multi-provider support.

Supports:
- Cloud providers: Anthropic, OpenAI, OpenRouter, Zen, Google Vertex, Azure, Cohere
- Local inference: Ollama, vLLM, LMStudio, llama.cpp, LiteLLM, Text Generation WebUI
- Agentic harnesses: OpenCode, OpenCrew, AgentForge, CrewAI, AutoGen

Phase 7.1: Multi-Provider Support.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


# Provider type enum
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


# Common model registry
MODEL_REGISTRY = {
    # Anthropic models
    "claude-3-5-sonnet-20241022": ModelInfo(
        id="claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        context_length=200000,
        max_output_tokens=8192,
        pricing_input=0.003,
        pricing_output=0.015,
    ),
    "claude-3-opus-20240229": ModelInfo(
        id="claude-3-opus-20240229",
        name="Claude 3 Opus",
        provider="anthropic",
        context_length=200000,
        max_output_tokens=4096,
        pricing_input=0.015,
        pricing_output=0.075,
    ),
    "claude-3-haiku-20240307": ModelInfo(
        id="claude-3-haiku-20240307",
        name="Claude 3 Haiku",
        provider="anthropic",
        context_length=128000,
        max_output_tokens=4096,
        pricing_input=0.00025,
        pricing_output=0.00125,
    ),
    # OpenAI models
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        context_length=128000,
        max_output_tokens=16384,
        supports_vision=True,
        pricing_input=0.005,
        pricing_output=0.015,
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        context_length=128000,
        max_output_tokens=16384,
        pricing_input=0.00015,
        pricing_output=0.0006,
    ),
    "gpt-4-turbo": ModelInfo(
        id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider="openai",
        context_length=128000,
        max_output_tokens=4096,
        pricing_input=0.01,
        pricing_output=0.03,
    ),
    "gpt-3.5-turbo": ModelInfo(
        id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider="openai",
        context_length=16385,
        max_output_tokens=4096,
        pricing_input=0.0005,
        pricing_output=0.0015,
    ),
    # OpenRouter models
    "anthropic/claude-3.5-sonnet": ModelInfo(
        id="anthropic/claude-3.5-sonnet",
        name="Claude 3.5 Sonnet (OR)",
        provider="openrouter",
        context_length=200000,
        max_output_tokens=8192,
        pricing_input=0.003,
        pricing_output=0.015,
    ),
    # Google models
    "gemini-1.5-pro": ModelInfo(
        id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        provider="google_vertex",
        context_length=2000000,
        max_output_tokens=8192,
        supports_vision=True,
        pricing_input=0.00125,
        pricing_output=0.005,
    ),
    "gemini-1.5-flash": ModelInfo(
        id="gemini-1.5-flash",
        name="Gemini 1.5 Flash",
        provider="google_vertex",
        context_length=1000000,
        max_output_tokens=8192,
        supports_vision=True,
        pricing_input=0.0,
        pricing_output=0.0,
    ),
    # Mistral models
    "mistral-large-latest": ModelInfo(
        id="mistral-large-latest",
        name="Mistral Large",
        provider="mistral",
        context_length=128000,
        max_output_tokens=8192,
        pricing_input=0.002,
        pricing_output=0.006,
    ),
    "mistral-small-latest": ModelInfo(
        id="mistral-small-latest",
        name="Mistral Small",
        provider="mistral",
        context_length=128000,
        max_output_tokens=4096,
        pricing_input=0.0002,
        pricing_output=0.0006,
    ),
    # Local models (representative)
    "llama-3.1-70b": ModelInfo(
        id="llama-3.1-70b",
        name="Llama 3.1 70B",
        provider="ollama",
        context_length=128000,
        max_output_tokens=4096,
        is_local=True,
    ),
    "mistral-7b": ModelInfo(
        id="mistral-7b",
        name="Mistral 7B",
        provider="ollama",
        context_length=8192,
        max_output_tokens=4096,
        is_local=True,
    ),
    "phi-3": ModelInfo(
        id="phi-3-mini-4k",
        name="Phi-3 Mini",
        provider="ollama",
        context_length=4096,
        max_output_tokens=4096,
        is_local=True,
    ),
}


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
        return MODEL_REGISTRY.get(model)


# Cloud Provider Implementations
class AnthropicProvider(BaseLLMProvider):
    """Anthropic (Claude) provider."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self._client = None

    def _get_client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY required")
            try:
                from anthropic import Anthropic

                self._client = Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        start = time.time()
        client = self._get_client()

        # Convert messages to Anthropic format
        system = None
        anthropic_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                anthropic_messages.append(msg)

        max_tokens = max_tokens or 4096

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=anthropic_messages,
            )
            content = response.content[0].text
            return LLMResponse(
                content=content,
                model=model,
                provider="anthropic",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                FinishReason=response.stop_reason,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs
    ):
        super().__init__(api_key, base_url, **kwargs)
        self._client = None

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required")
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=api_key,
                    base_url=self.base_url or "https://api.openai.com/v1",
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        start = time.time()
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
            )
            content = response.choices[0].message.content
            return LLMResponse(
                content=content or "",
                model=model,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                FinishReason=response.choices[0].finish_reason,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter.ai provider."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            api_key,
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )
        self._client = None

    def _get_client(self):
        """Lazy load OpenRouter client."""
        if self._client is None:
            api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY required")
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        start = time.time()
        client = self._get_client()

        # Translate model names for OpenRouter
        model_map = {
            "gpt-4o": "openai/gpt-4o",
            "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
        }
        or_model = model_map.get(model, model)

        try:
            response = client.chat.completions.create(
                model=or_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                extra_headers={
                    "HTTP-Referer": "https://github.com/iknowkungfubar/autoresearch-stack",
                    "X-Title": "Autonomous Research Stack",
                },
            )
            content = response.choices[0].message.content
            return LLMResponse(
                content=content or "",
                model=model,
                provider="openrouter",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                FinishReason=response.choices[0].finish_reason,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class ZenProvider(BaseLLMProvider):
    """Zen AI provider (zen-ai.com)."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            api_key,
            base_url="https://api.zen-ai.com/v1",
            **kwargs,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        import requests

        start = time.time()
        api_key = self.api_key or os.getenv("ZEN_API_KEY")
        if not api_key:
            raise ValueError("ZEN_API_KEY required")

        # Model mapping for Zen
        model_map = {
            "gpt-4o": "gpt-4o",
            "claude-3.5-sonnet": "claude-sonnet-3-5-20241022",
            "gemini": "gemini-pro",
        }
        zen_model = model_map.get(model, model)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": zen_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens or 4096,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return LLMResponse(
                content=content,
                model=model,
                provider="zen",
                usage=data.get("usage", {}),
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"Zen API error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: str = "2024-02-01",
        **kwargs,
    ):
        super().__init__(api_key, base_url, api_version=api_version, **kwargs)
        self._client = None

    def _get_client(self):
        """Lazy load Azure client."""
        if self._client is None:
            api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY required")

            base_url = self.base_url or os.getenv("AZURE_OPENAI_ENDPOINT")
            if not base_url:
                raise ValueError("AZURE_OPENAI_ENDPOINT required")

            try:
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    api_key=api_key,
                    azure_endpoint=base_url,
                    api_version=self.extra_params.get("api_version", "2024-02-01"),
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        start = time.time()
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=model,  # Azure deployment name
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
            )
            content = response.choices[0].message.content
            return LLMResponse(
                content=content or "",
                model=model,
                provider="azure_openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                FinishReason=response.choices[0].finish_reason,
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class GoogleVertexProvider(BaseLLMProvider):
    """Google Vertex AI provider."""

    def __init__(
        self, api_key: Optional[str] = None, project_id: Optional[str] = None, **kwargs
    ):
        super().__init__(api_key, project_id=project_id, **kwargs)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        start = time.time()

        try:
            from google.auth import default  # noqa: F401
            from google.auth import load_credentials_from_file  # noqa: F401
            import vertexai
            from vertexai.generative_models import GenerativeModel

            project_id = self.extra_params.get("project_id") or os.getenv(
                "GOOGLE_CLOUD_PROJECT"
            )
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT required")

            vertexai.init(project=project_id)

            # Model mapping
            model_map = {
                "gemini-1.5-pro": "gemini-1.5-pro",
                "gemini-1.5-flash": "gemini-1.5-flash",
            }
            vertex_model = model_map.get(model, model)

            gen_model = GenerativeModel(vertex_model)

            # Convert messages to content
            contents = []
            for msg in messages:
                if msg.get("role") == "user":
                    contents.append(msg.get("content", ""))
                elif msg.get("role") == "model":
                    contents.append(f"Response: {msg.get('content', '')}")

            response = gen_model.generate_content(
                contents,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens or 8192,
                },
            )
            content = response.text

            return LLMResponse(
                content=content,
                model=model,
                provider="google_vertex",
                latency_ms=(time.time() - start) * 1000,
            )
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform required: pip install google-cloud-aiplatform"
            )
        except Exception as e:
            raise RuntimeError(f"Google Vertex error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class MistralProvider(BaseLLMProvider):
    """Mistral AI provider."""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(
            api_key,
            base_url="https://api.mistral.ai/v1",
            **kwargs,
        )
        self._client = None

    def _get_client(self):
        """Lazy load Mistral client."""
        if self._client is None:
            api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY required")
            try:
                from mistralai import Mistral

                self._client = Mistral(api_key=api_key)
            except ImportError:
                raise ImportError("mistralai package required: pip install mistralai")
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        start = time.time()
        client = self._get_client()

        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 8192,
            )
            content = response.choices[0].message.content

            return LLMResponse(
                content=content,
                model=model,
                provider="mistral",
                usage=response.usage.model_dump() if response.usage else {},
                FinishReason=response.choices[0].finish_reason,
                latency_ms=(time.time() - start) * 1000,
            )
        except ImportError:
            raise ImportError("mistralai package required: pip install mistralai")
        except Exception as e:
            raise RuntimeError(f"Mistral API error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


# Local Inference Providers
class OllamaProvider(BaseLLMProvider):
    """Ollama local inference provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        super().__init__(api_key, base_url=base_url, **kwargs)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        import requests

        start = time.time()
        base_url = self.base_url or "http://localhost:11434"

        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "num_predict": max_tokens or 4096,
                },
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=model,
                provider="ollama",
                usage=data.get("done", False),
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)

    def list_models(self) -> List[str]:
        """List available models."""
        import requests

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


class LMStudioProvider(BaseLLMProvider):
    """LM Studio local inference provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:1234/v1",
        **kwargs,
    ):
        super().__init__(api_key, base_url=base_url, **kwargs)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        import requests

        start = time.time()
        base_url = self.base_url or "http://localhost:1234/v1"

        # LM Studio uses OpenAI-compatible API
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens or 4096,
                },
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return LLMResponse(
                content=content,
                model=model,
                provider="lmstudio",
                usage=data.get("usage", {}),
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"LM Studio error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class VLLMProvider(BaseLLMProvider):
    """vLLM inference provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
        **kwargs,
    ):
        super().__init__(api_key, base_url=base_url, **kwargs)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        import requests

        start = time.time()
        base_url = self.base_url or "http://localhost:8000/v1"

        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens or 4096,
                },
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            return LLMResponse(
                content=content,
                model=model,
                provider="vllm",
                usage=data.get("usage", {}),
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"vLLM error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class LiteLLMProvider(BaseLLMProvider):
    """LiteLLM proxy provider (supports multiple backends)."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs
    ):
        super().__init__(api_key, base_url, **kwargs)
        self._client = None

    def _get_client(self):
        """Lazy load LiteLLM client."""
        if self._client is None:
            _api_key = self.api_key or os.getenv("LITELLM_API_KEY")
            try:
                from litellm import completion

                self._client = completion
            except ImportError:
                raise ImportError("litellm package required: pip install litellm")
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        start = time.time()
        client = self._get_client()

        # LiteLLM model format: provider/model
        try:
            response = client(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or 4096,
                api_base=self.base_url,
                api_key=self.api_key,
            )
            content = response.choices[0].message.content

            return LLMResponse(
                content=content,
                model=model,
                provider="litellm",
                latency_ms=(time.time() - start) * 1000,
            )
        except ImportError:
            raise ImportError("litellm package required: pip install litellm")
        except Exception as e:
            raise RuntimeError(f"LiteLLM error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class LlamaCppProvider(BaseLLMProvider):
    """llama.cpp binding provider."""

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        self._client = None

    def _get_client(self):
        """Lazy load llama.cpp."""
        if self._client is None:
            try:
                from llama_cpp import Llama

                model_path = self.extra_params.get("model_path")
                if not model_path:
                    raise ValueError("model_path required")
                self._client = Llama(model_path=model_path, n_ctx=4096, n_threads=4)
            except ImportError:
                raise ImportError(
                    "llama-cpp-python required: pip install llama-cpp-python"
                )
        return self._client

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        start = time.time()
        client = self._get_client()

        # Convert messages to single prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt += "\nassistant: "

        try:
            response = client(
                prompt=prompt,
                max_tokens=max_tokens or 4096,
                temperature=temperature,
                stop=["user:", "system:"],
            )
            content = response["choices"][0]["text"]

            return LLMResponse(
                content=content,
                model=model,
                provider="llama_cpp",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"llama.cpp error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


class TextGenWebUIProvider(BaseLLMProvider):
    """Text Generation WebUI provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:5000",
        **kwargs,
    ):
        super().__init__(api_key, base_url=base_url, **kwargs)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion."""
        import requests

        start = time.time()
        base_url = self.base_url or "http://localhost:5000"

        # Convert to prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt += "\nassistant: "

        try:
            response = requests.post(
                f"{base_url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_new_tokens": max_tokens or 4096,
                    "temperature": temperature,
                    "stop": ["user:", "system:"],
                },
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()

            content = data.get("choices", [{}])[0].get("text", "")

            return LLMResponse(
                content=content,
                model=model,
                provider="textgen_webui",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            raise RuntimeError(f"TextGen WebUI error: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Chat completion."""
        return self.complete(messages, model, temperature, max_tokens, **kwargs)


# Provider factory
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


# Unified client
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
        messages: List[Dict[str, str]],
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

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat completion (alias)."""
        return self.complete(messages, **kwargs)

    @classmethod
    def from_env(cls) -> "LLMClient":
        """Create client from environment variables."""
        # Auto-detect provider from API keys
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


if __name__ == "__main__":
    # Test providers
    print("=== Testing LLM Provider Factory ===\n")

    # Test from environment
    try:
        client = LLMClient.from_env()
        print(f"Auto-detected provider: {client.provider.__class__.__name__}")
    except ValueError as e:
        print(f"Environment not configured: {e}\n")

    # List supported models
    print("Supported models:")
    for model_id in list(MODEL_REGISTRY.keys())[:10]:
        info = MODEL_REGISTRY[model_id]
        print(f"  {model_id}: {info.name} ({info.provider})")
