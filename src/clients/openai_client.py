from core.settings import settings
from openai import AsyncOpenAI

# Initialize the client pointing to Ollama Cloud
client = AsyncOpenAI(
    base_url=settings.OLLAMA_PROVIDER_URL,
    api_key=(
        settings.OLLAMA_API_KEY.get_secret_value()
        if settings.OLLAMA_API_KEY is not None
        else "ollama"
    ),
)
