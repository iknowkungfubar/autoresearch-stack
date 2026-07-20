"""Zen AI (zen-ai.com) cloud provider."""

import logging
import os
import time
from typing import Dict, List, Optional

import requests

from .._types import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


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
