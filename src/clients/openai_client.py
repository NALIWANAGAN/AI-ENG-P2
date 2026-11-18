import os
from core.settings import settings
from openai import OpenAI

# Initialize the client pointing to Ollama Cloud
client = OpenAI(
    base_url=os.getenv(settings.OLLAMA_PROVIDER_URL),
    api_key=os.getenv(settings.OLLAMA_API_KEY)
)
