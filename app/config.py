from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # DeepSeek LLM
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_service_type: str = "local"  # "local" or "aragemma"
    embedding_api_url: str = ""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # JWT
    jwt_secret_key: str = "change-me-to-a-random-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Database
    database_url: str = "sqlite:///./rag_app.db"

    # Logfire
    logfire_token: str = ""

    # Rate Limiting
    rate_limit_default: str = "60/minute"

    # Document Processing
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Tesseract
    tesseract_cmd: str = "tesseract"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
