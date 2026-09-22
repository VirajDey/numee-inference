from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the inference service.

    Validated at import time so a misconfigured deployment fails on startup
    rather than on the first request that needs the missing value.
    """

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Shared secret the calling service presents as X-Internal-Token.
    inference_service_token: str = Field(..., min_length=16)

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str | None = None
    openai_vector_size: int = 1536
    sentence_transformer_vector_size: int = 384

    # Embeddings
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token: str | None = None

    # Embedding and document parsing are CPU bound, so they run in a bounded
    # thread pool. Size this to the replica's CPU allocation, not to traffic.
    inference_max_concurrency: int = 4

    log_level: str = "INFO"


settings = Settings()
