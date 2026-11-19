from unittest.mock import AsyncMock, MagicMock
import clients.openai_client as openai_module
import pytest


@pytest.fixture
def anyio_backend():
    """Run @pytest.mark.anyio tests on asyncio only (trio is not installed)."""
    return "asyncio"


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Unit test fixture: replaces the module-level AsyncOpenAI client with a mock."""
    mock_client = MagicMock()

    # Default text completion response
    mock_message = MagicMock()
    mock_message.content = "Say Hello."
    mock_message.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    monkeypatch.setattr(openai_module, "client", mock_client)
    return mock_client
