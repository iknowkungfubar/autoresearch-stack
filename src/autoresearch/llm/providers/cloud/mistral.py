"""Mistral AI cloud provider."""

import logging
import os
import time
from typing import Dict, List, Optional

from .._types import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


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
