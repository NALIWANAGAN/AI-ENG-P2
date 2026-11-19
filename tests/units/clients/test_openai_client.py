import importlib
import json
from unittest.mock import MagicMock

import pytest
from openai import OpenAI, OpenAIError
from pydantic import SecretStr

import clients.openai_client as openai_module
from core.settings import settings


# =======================================================
# Fixture-Backed OpenAI Client Tests
# =======================================================


class TestMockedOpenAIClient:
    def test_happy_fixture_replaces_module_client(self, mock_openai_client: MagicMock) -> None:
        assert openai_module.client is mock_openai_client

    def test_happy_default_text_completion(self, mock_openai_client: MagicMock) -> None:
        messages = [{"role": "user", "content": "Hello"}]

        response = openai_module.client.chat.completions.create(
            model=settings.OLLAMA_MODEL_NAME,
            messages=messages,
        )

        message = response.choices[0].message
        assert message.content == "Say Hello."
        assert message.tool_calls is None
        mock_openai_client.chat.completions.create.assert_called_once_with(
            model=settings.OLLAMA_MODEL_NAME,
            messages=messages,
        )

    def test_happy_tool_call_response(self, mock_openai_client: MagicMock) -> None:
        tool_call = MagicMock()
        tool_call.id = "call_001"
        tool_call.type = "function"
        tool_call.function.name = "read_file_slice"
        tool_call.function.arguments = json.dumps({"path": "src/schema.sql", "line_start": 1, "line_end": 50})

        mock_message = mock_openai_client.chat.completions.create.return_value.choices[0].message
        mock_message.content = None
        mock_message.tool_calls = [tool_call]

        response = openai_module.client.chat.completions.create(
            model=settings.OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": "Inspect schema.sql"}],
        )

        returned_msg = response.choices[0].message
        assert returned_msg.content is None
        assert len(returned_msg.tool_calls) == 1
        assert returned_msg.tool_calls[0].function.name == "read_file_slice"
        assert json.loads(returned_msg.tool_calls[0].function.arguments)["path"] == "src/schema.sql"

    def test_sad_completion_raises_openai_error(self, mock_openai_client: MagicMock) -> None:
        mock_openai_client.chat.completions.create.side_effect = OpenAIError("Provider unavailable")

        with pytest.raises(OpenAIError, match="Provider unavailable"):
            openai_module.client.chat.completions.create(
                model=settings.OLLAMA_MODEL_NAME,
                messages=[{"role": "user", "content": "Trigger failure"}],
            )


# =======================================================
# Client Initialization & Settings Tests
# =======================================================


class TestOpenAIClientConfig:
    def test_happy_initializes_with_configured_url_and_secret_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "OLLAMA_PROVIDER_URL", "https://example.test/v1")
        monkeypatch.setattr(settings, "OLLAMA_API_KEY", SecretStr("test-secret-key"))

        try:
            reloaded = importlib.reload(openai_module)
            assert isinstance(reloaded.client, OpenAI)
            assert str(reloaded.client.base_url).rstrip("/") == "https://example.test/v1"
            assert reloaded.client.api_key == "test-secret-key"
        finally:
            monkeypatch.undo()
            importlib.reload(openai_module)

    def test_edge_falls_back_to_ollama_key_when_api_key_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "OLLAMA_API_KEY", None)

        try:
            reloaded = importlib.reload(openai_module)
            assert isinstance(reloaded.client, OpenAI)
            assert reloaded.client.api_key == "ollama"
        finally:
            monkeypatch.undo()
            importlib.reload(openai_module)
