import time
from typing import List, Any, Optional
import numpy as np
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings

from src.config import settings


class LangChainEmbeddingAdapter(EmbeddingFunction, Embeddings):
    """Dual interface adapter for ChromaDB and LangChain.
    
    - `__call__`: Speaks ChromaDB's protocol, guaranteeing a 2D np.ndarray output shape (N, D).
    - `embed_documents` / `embed_query`: Speaks LangChain's Embeddings interface by returning pure Python float lists.
    """

    def __init__(self, lc_embeddings: Any, provider_name: str = "langchain_adapter"):
        self.lc_embeddings = lc_embeddings
        self._provider_name = provider_name

    def name(self) -> str:
        """Required by ChromaDB to validate collection embedding identity."""
        model_attr = getattr(self.lc_embeddings, "model", "custom")
        return f"{self._provider_name}_{model_attr}"

    def _flatten_texts(self, input_data: Any) -> List[str]:
        """Recursively flattens strings, lists, or nested lists into a 1D List[str]."""
        if input_data is None:
            return []
        if isinstance(input_data, str):
            return [input_data]

        flat: List[str] = []
        if isinstance(input_data, (list, tuple)):
            for item in input_data:
                flat.extend(self._flatten_texts(item))
        else:
            flat.append(str(input_data))
        return flat

    def _clean_embeddings(self, raw_embeddings: Any) -> List[List[float]]:
        """Converts raw embeddings to pure Python 2D list[list[float]]."""
        if not raw_embeddings:
            return []

        # Handle edge-case where raw_embeddings is returned as a 1D float array/list
        if isinstance(raw_embeddings[0], (float, int, np.floating)):
            return [[float(x) for x in raw_embeddings]]

        cleaned: List[List[float]] = []
        for vec in raw_embeddings:
            if isinstance(vec, np.ndarray):
                cleaned.append(vec.astype(float).tolist())
            elif isinstance(vec, list):
                cleaned.append([float(x) for x in vec])
            else:
                cleaned.append([float(x) for x in list(vec)])
        return cleaned

    def embed_documents(
        self, texts: Optional[Any] = None, input: Optional[Any] = None, **kwargs
    ) -> List[List[float]]:
        """LangChain Embeddings interface returning List[List[float]]."""
        raw_input = input if input is not None else texts
        if raw_input is None:
            raw_input = kwargs.get("queries", kwargs.get("query", []))

        docs = self._flatten_texts(raw_input)
        if not docs:
            return []

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                try:
                    raw = self.lc_embeddings.embed_documents(docs)
                except AttributeError as ae:
                    if "tolist" in str(ae):
                        # Fallback for langchain_google_genai 1D list return bug
                        raw = [self.lc_embeddings.embed_query(d) for d in docs]
                    else:
                        raise ae
                return self._clean_embeddings(raw)
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries:
                    wait_time = 45
                    print(
                        f"\n Rate limit hit (429). Waiting {wait_time}s to reset quota (Attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e
        return []

    def embed_query(
        self, text: Optional[Any] = None, input: Optional[Any] = None, **kwargs
    ) -> List[float]:
        """LangChain Embeddings interface for single queries returning List[float]."""
        raw_input = input if input is not None else text
        if raw_input is None:
            raw_input = kwargs.get("query", "")

        docs = self._flatten_texts(raw_input)
        if not docs:
            return []

        query_str = docs[0]

        if hasattr(self.lc_embeddings, "embed_query"):
            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    res = self.lc_embeddings.embed_query(query_str)
                    if isinstance(res, np.ndarray):
                        return res.astype(float).tolist()
                    return [float(x) for x in res]
                except Exception as e:
                    err_str = str(e)
                    if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries:
                        wait_time = 45
                        print(
                            f"\n Rate limit hit (429). Waiting {wait_time}s to reset quota (Attempt {attempt}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                    else:
                        break

        res = self.embed_documents([query_str])
        return res[0] if res else []

    def __call__(self, input: Any) -> Any:
        """ChromaDB EmbeddingFunction interface returning a 2D np.ndarray."""
        embeddings = self.embed_documents(input=input)
        arr = np.array(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        print(f"[DEBUG] shape={arr.shape}, ndim={arr.ndim}, dtype={arr.dtype}")
        return arr


def get_embedding_function() -> LangChainEmbeddingAdapter:
    """Factory method returning configured dual-compatible embedding provider."""
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        if not settings.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY required for OpenAI provider.")
        lc_emb = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY,
            dimensions=settings.EMBEDDING_DIMENSION,
        )
        return LangChainEmbeddingAdapter(lc_emb, provider_name="openai")

    elif provider == "gemini":
        if not settings.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY required for Gemini provider.")
        lc_emb = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.EMBEDDING_API_KEY,
            output_dimensionality=settings.EMBEDDING_DIMENSION,
        )
        return LangChainEmbeddingAdapter(lc_emb, provider_name="gemini")

    elif provider == "huggingface":
        lc_emb = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        return LangChainEmbeddingAdapter(lc_emb, provider_name="huggingface")

    elif provider == "ollama":
        lc_emb = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL or "http://localhost:11434",
        )
        return LangChainEmbeddingAdapter(lc_emb, provider_name="ollama")

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: '{settings.EMBEDDING_PROVIDER}'")