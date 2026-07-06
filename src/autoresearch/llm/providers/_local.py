"""Local LLM provider implementations.

Ollama, LM Studio, vLLM, LiteLLM, llama.cpp, TextGen WebUI.
"""

import os
import time
from typing import Any, Dict, List, Optional

import requests

from ._types import BaseLLMProvider, LLMResponse


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
        start = time.time()
        base_url = self.base_url or "http://localhost:1234/v1"

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
        start = time.time()
        base_url = self.base_url or "http://localhost:5000"

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
