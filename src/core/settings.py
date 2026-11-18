from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "codebase-and-schema-auditor"
    DEBUG: bool = True
    ENVIRONMENT: Literal["dev", "prod", "test"] = "dev"
    LOG_LEVEL: str = "INFO"

    # Ollama Model
    OLLAMA_PROVIDER_URL: str = "https://ollama.com/v1"
    OLLAMA_API_KEY: SecretStr | None = None
    OLLAMA_MODEL_NAME: str = "gpt-oss:20b"

    # SQLGlot
    SQLGLOT_DB_DIALECT: str = "postgres"


settings = Settings()
