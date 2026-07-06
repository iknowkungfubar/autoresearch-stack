"""Cloud LLM provider implementations.

Anthropic, OpenAI, OpenRouter, Zen, Azure OpenAI, Google Vertex, Mistral.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from ._types import BaseLLMProvider, LLMResponse


logger = logging.getLogger(__name__)


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
        start = time.time()
        api_key = self.api_key or os.getenv("ZEN_API_KEY")
        if not api_key:
            raise ValueError("ZEN_API_KEY required")

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
                model=model,
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
            import vertexai
            from google.auth import (
                default,
                load_credentials_from_file,
            )
            from vertexai.generative_models import GenerativeModel

            project_id = self.extra_params.get("project_id") or os.getenv(
                "GOOGLE_CLOUD_PROJECT"
            )
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT required")

            vertexai.init(project=project_id)

            model_map = {
                "gemini-1.5-pro": "gemini-1.5-pro",
                "gemini-1.5-flash": "gemini-1.5-flash",
            }
            vertex_model = model_map.get(model, model)

            gen_model = GenerativeModel(vertex_model)

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
