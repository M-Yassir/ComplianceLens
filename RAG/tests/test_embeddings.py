import pytest
from typing import List
from langchain_core.embeddings import Embeddings

from src.vector_store.embeddings import LangChainEmbeddingAdapter


class MockLangChainEmbeddings(Embeddings):
    """Mock LangChain embeddings provider for unit testing."""

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.model = "mock-model-v1"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(i + 1) for i in range(self.dimension)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.5 for _ in range(self.dimension)]


@pytest.fixture
def mock_adapter():
    mock_lc = MockLangChainEmbeddings(dimension=4)
    return LangChainEmbeddingAdapter(mock_lc, provider_name="test_provider")


# --- Name & Utility Tests ---

def test_adapter_name(mock_adapter):
    """Ensure ChromaDB gets a properly formatted adapter name."""
    assert mock_adapter.name() == "test_provider_mock-model-v1"


def test_flatten_texts(mock_adapter):
    """Ensure recursive text flattening handles strings, lists, and nested structures."""
    assert mock_adapter._flatten_texts("single string") == ["single string"]
    assert mock_adapter._flatten_texts(["a", "b"]) == ["a", "b"]
    assert mock_adapter._flatten_texts([["nested1", "nested2"], "flat"]) == ["nested1", "nested2", "flat"]
    assert mock_adapter._flatten_texts(None) == []


# --- LangChain Interface Tests ---

def test_embed_documents_type_and_values(mock_adapter):
    """LangChain interface must return pure Python List[List[float]]."""
    res = mock_adapter.embed_documents(["doc1", "doc2"])
    assert isinstance(res, list)
    assert len(res) == 2
    assert isinstance(res[0], list)
    assert isinstance(res[0][0], float)
    assert res[0] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_embed_query_type_and_values(mock_adapter):
    """LangChain query interface must return a pure Python List[float]."""
    res = mock_adapter.embed_query("query text")
    assert isinstance(res, list)
    assert len(res) == 4
    assert isinstance(res[0], float)
    assert res == pytest.approx([0.5, 0.5, 0.5, 0.5])


# --- ChromaDB Interface (__call__) Tests ---

def test_chroma_call_single_text(mock_adapter):
    """ChromaDB __call__ with a single text returns validated embedding list."""
    res = mock_adapter("single query")
    assert isinstance(res, list)
    assert len(res) == 1
    assert res[0] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_chroma_call_multiple_texts(mock_adapter):
    """ChromaDB __call__ with multiple inputs returns embeddings for each document."""
    res = mock_adapter(["text1", "text2", "text3"])
    assert isinstance(res, list)
    assert len(res) == 3
    assert res[0] == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_chroma_call_empty_input_raises_value_error(mock_adapter):
    """ChromaDB's EmbeddingFunction wrapper raises ValueError on empty inputs."""
    with pytest.raises(ValueError, match="Expected Embeddings to be non-empty list or numpy array"):
        mock_adapter([])