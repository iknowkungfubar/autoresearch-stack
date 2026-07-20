"""Google Vertex AI cloud provider."""

import logging
import os
import time
from typing import Dict, List, Optional

from .._types import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)


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
