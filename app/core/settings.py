from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Google Cloud ---
    SERVICE_ACCOUNT_FILE_PATH: str | None = None

    # --- Gemini---
    GEMINI_API_KEY: str | None = None
    LANGUAGE_MODEL_NAME: str = "gemini-3.5-flash"
    EMBEDDING_MODEL_NAME: str = "gemini-embedding-001"

    # --- Qdrant ---
    QDRANT_API_KEY: str | None = None
    QDRANT_URL: str | None = None
    QDRANT_COLLECTION_NAME: str = "wikipedia"


# Export an unique instance of configuration
settings = Settings()