from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory pointing to project root
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ChromaDB Configuration
    CHROMA_HOST: str = "127.0.0.1"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "ccpa_child_chunks"

    # Environment
    ENV: str = "development"

    # Paths
    DATA_DIR: Path = BASE_DIR / "data"

    @property
    def PROCESSED_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "processed"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()