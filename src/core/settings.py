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

    # File & Directory Read Guardrails
    DEFAULT_LINES_PER_BATCH: int = 200
    DEFAULT_ITEMS_PER_BATCH: int = 100
    DEFAULT_SEARCH_MATCHES_PER_BATCH: int = 50
    DEFAULT_MAX_MATCHES_PER_FILE: int = 20
    MAX_READ_TOKENS: int = 4000
    MAX_AUDIT_REPORT_BYTES: int = 200_000
    TOKENIZER_ENCODING: str = "o200k_base"

    # LLM Rate-Limit, Decoding & Retry Policy (RPM / TPM pacing)
    LLM_TEMPERATURE: float = 0.0
    LLM_PARALLEL_TOOL_CALLS: bool = True
    LLM_WRITE_CHECKPOINT_TURNS: int = 3
    LLM_COMPACT_MIN_CHARS: int = 300
    LLM_MAX_RETRIES: int = 5
    LLM_TURN_DELAY_SECONDS: float = 4.0
    LLM_RETRY_BASE_DELAY_SECONDS: float = 10.0
    LLM_RETRY_MAX_DELAY_SECONDS: float = 65.0


settings = Settings()
