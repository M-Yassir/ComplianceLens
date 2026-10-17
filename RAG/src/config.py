from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

MODULE_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = MODULE_BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"

    # Database
    CHROMA_HOST: str = "127.0.0.1"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "ccpa_child_chunks"

    # Embedding Provider Setup
    EMBEDDING_PROVIDER: str = "gemini"
    EMBEDDING_MODEL: str = "models/text-embedding-001"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_DIMENSION: int = 768

    # LLM Provider Setup
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_API_KEY: str = ""

    # Langfuse Observability & Tracing
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_ENABLED: bool = True

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    model_config = SettingsConfigDict(
        env_file=str(MODULE_BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()