"""OpenAI-compatible cloud providers.

OpenAI, OpenRouter, and Azure OpenAI all use the OpenAI Python SDK.
"""

import logging
import os
import time
from typing import Dict, List, Optional

from .._types import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


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
