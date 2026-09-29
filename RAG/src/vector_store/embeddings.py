from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from src.config import settings


class LangChainEmbeddingAdapter(EmbeddingFunction):
    """Adapts any LangChain Embeddings instance to Chroma's EmbeddingFunction interface."""

    def __init__(self, lc_embeddings):
        self.lc_embeddings = lc_embeddings

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.lc_embeddings.embed_documents(input)


def get_embedding_function() -> EmbeddingFunction:
    """Factory method returning configured Chroma-compatible embedding function."""
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        if not settings.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY required for OpenAI provider.")
        lc_emb = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY,
            dimensions=settings.EMBEDDING_DIMENSION,
        )
        return LangChainEmbeddingAdapter(lc_emb)

    elif provider == "gemini":
        if not settings.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY required for Gemini provider.")
        lc_emb = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.EMBEDDING_API_KEY,
            output_dimensionality=settings.EMBEDDING_DIMENSION,
        )
        return LangChainEmbeddingAdapter(lc_emb)

    elif provider == "huggingface":
        lc_emb = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        return LangChainEmbeddingAdapter(lc_emb)

    elif provider == "ollama":
        lc_emb = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL or "http://localhost:11434",
        )
        return LangChainEmbeddingAdapter(lc_emb)

    elif provider == "default":
        return embedding_functions.DefaultEmbeddingFunction()

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: '{settings.EMBEDDING_PROVIDER}'")