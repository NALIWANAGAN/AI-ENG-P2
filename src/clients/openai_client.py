from core.settings import settings
from openai import OpenAI

# Initialize the client pointing to Ollama Cloud
client = OpenAI(
    base_url=settings.OLLAMA_PROVIDER_URL,
    api_key=(
        settings.OLLAMA_API_KEY.get_secret_value()
        if settings.OLLAMA_API_KEY is not None
        else "ollama"
    ),
)
