"""Tests for local LLM provider complete() paths (LM Studio, vLLM, TextGen WebUI).

These exercise real request construction and response parsing for the local
inference providers in autoresearch/llm/providers/_local.py that were previously
untested.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestLMStudioComplete:
    """LM Studio uses an OpenAI-compatible /chat/completions endpoint."""

    def test_complete_parses_choices(self):
        from autoresearch.llm.providers import LMStudioProvider

        provider = LMStudioProvider(base_url="http://localhost:1234/v1")
        with patch("autoresearch.llm.providers._local.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "lmstudio reply"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            response = provider.complete(
                [{"role": "user", "content": "Hi"}],
                model="llama3",
                temperature=0.3,
                max_tokens=128,
            )

        assert response.content == "lmstudio reply"
        assert response.provider == "lmstudio"
        assert response.model == "llama3"
        # The provider must POST to the chat completions endpoint.
        called_url = mock_post.call_args[0][0]
        assert called_url.endswith("/chat/completions")
        sent = mock_post.call_args.kwargs["json"]
        assert sent["max_tokens"] == 128
        assert sent["temperature"] == 0.3

    def test_complete_default_max_tokens(self):
        from autoresearch.llm.providers import LMStudioProvider

        provider = LMStudioProvider()
        with patch("autoresearch.llm.providers._local.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "x"}}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            provider.complete([{"role": "user", "content": "Hi"}], model="m")

        sent = mock_post.call_args.kwargs["json"]
        assert sent["max_tokens"] == 4096

    def test_complete_raises_on_error(self):
        from autoresearch.llm.providers import LMStudioProvider

        provider = LMStudioProvider()
        with patch(
            "autoresearch.llm.providers._local.requests.post",
            side_effect=Exception("boom"),
        ):
            with pytest.raises(RuntimeError, match="LM Studio error"):
                provider.complete(
                    [{"role": "user", "content": "Hi"}], model="m"
                )

    def test_chat_aliases_complete(self):
        from autoresearch.llm.providers import LMStudioProvider

        provider = LMStudioProvider()
        with patch(
            "autoresearch.llm.providers._local.LMStudioProvider.complete",
            return_value="marker",
        ) as mock_complete:
            result = provider.chat(
                [{"role": "user", "content": "Hi"}], model="m"
            )
        assert result == "marker"
        mock_complete.assert_called_once()


class TestVLLMComplete:
    """vLLM also uses an OpenAI-compatible /chat/completions endpoint."""

    def test_complete_parses_choices(self):
        from autoresearch.llm.providers import VLLMProvider

        provider = VLLMProvider(base_url="http://localhost:8000/v1")
        with patch("autoresearch.llm.providers._local.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "vllm reply"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2},
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            response = provider.complete(
                [{"role": "user", "content": "Hi"}], model="qwen"
            )

        assert response.content == "vllm reply"
        assert response.provider == "vllm"
        called_url = mock_post.call_args[0][0]
        assert called_url.endswith("/chat/completions")

    def test_complete_raises_on_error(self):
        from autoresearch.llm.providers import VLLMProvider

        provider = VLLMProvider()
        with patch(
            "autoresearch.llm.providers._local.requests.post",
            side_effect=Exception("down"),
        ):
            with pytest.raises(RuntimeError, match="vLLM error"):
                provider.complete(
                    [{"role": "user", "content": "Hi"}], model="q"
                )


class TestTextGenWebUIComplete:
    """TextGen WebUI uses a /v1/completions endpoint with a formatted prompt."""

    def test_complete_builds_prompt_and_parses_text(self):
        from autoresearch.llm.providers import TextGenWebUIProvider

        provider = TextGenWebUIProvider(base_url="http://localhost:5000")
        with patch("autoresearch.llm.providers._local.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "choices": [{"text": "textgen reply"}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            response = provider.complete(
                [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "Hi"},
                ],
                model="mistral",
                temperature=0.9,
                max_tokens=64,
            )

        assert response.content == "textgen reply"
        assert response.provider == "textgen_webui"
        called_url = mock_post.call_args[0][0]
        assert called_url.endswith("/v1/completions")
        sent = mock_post.call_args.kwargs["json"]
        # Prompt must concatenate role/content and append the assistant marker.
        assert "system: be brief" in sent["prompt"]
        assert "user: Hi" in sent["prompt"]
        assert sent["prompt"].endswith("assistant: ")
        assert sent["max_new_tokens"] == 64
        assert sent["temperature"] == 0.9

    def test_complete_missing_text_defaults_empty(self):
        from autoresearch.llm.providers import TextGenWebUIProvider

        provider = TextGenWebUIProvider()
        with patch("autoresearch.llm.providers._local.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"choices": []}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            response = provider.complete(
                [{"role": "user", "content": "Hi"}], model="m"
            )

        assert response.content == ""

    def test_complete_raises_on_error(self):
        from autoresearch.llm.providers import TextGenWebUIProvider

        provider = TextGenWebUIProvider()
        with patch(
            "autoresearch.llm.providers._local.requests.post",
            side_effect=Exception("refused"),
        ):
            with pytest.raises(RuntimeError, match="TextGen WebUI error"):
                provider.complete(
                    [{"role": "user", "content": "Hi"}], model="m"
                )
