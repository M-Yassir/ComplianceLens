from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from src.config import settings


def get_embedding_function() -> EmbeddingFunction:
    """Factory method returning configured Chroma-compatible embedding function."""
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "gemini":
        if not settings.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY is required for Gemini provider.")
        return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=settings.EMBEDDING_API_KEY,
            model_name=settings.EMBEDDING_MODEL,
        )

    elif provider == "openai":
        if not settings.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY is required for OpenAI provider.")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.EMBEDDING_API_KEY,
            model_name=settings.EMBEDDING_MODEL,
        )

    elif provider == "huggingface":
        return embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=settings.EMBEDDING_API_KEY or None,
            model_name=settings.EMBEDDING_MODEL,
        )

    elif provider == "default":
        return embedding_functions.DefaultEmbeddingFunction()

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: '{settings.EMBEDDING_PROVIDER}'")