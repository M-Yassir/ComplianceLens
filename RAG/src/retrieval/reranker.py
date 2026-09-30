from typing import List, Optional
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

from src.config import settings
from src.retrieval.hybrid import HybridRetriever


class CrossEncoderReranker:
    """Reranks retrieved candidate Documents using FlashRank cross-encoder model."""

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-6-v2",
        cache_dir: Optional[str] = None,
    ):
        # FlashRank uses lightweight ONNX execution on CPU
        self.ranker = Ranker(
            model_name=model_name,
            cache_dir=cache_dir or str(settings.BASE_DIR / ".cache"),
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 4,
    ) -> List[Document]:
        """Reranks candidate documents based on cross-attention with the query."""
        if not documents:
            return []

        # Convert LangChain Document objects to FlashRank passage dicts
        passages = [
            {
                "id": doc.metadata.get("parent_id", str(idx)),
                "text": f"§ {doc.metadata.get('section', '')} {doc.metadata.get('section_title', '')}\n{doc.page_content}",
                "doc_ref": doc,
            }
            for idx, doc in enumerate(documents)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_req)

        # Reconstruct ordered Document objects with cross-encoder relevance scores
        reranked_docs: List[Document] = []
        for item in results[:top_k]:
            original_doc = item["doc_ref"]
            original_doc.metadata["rerank_score"] = float(item["score"])
            reranked_docs.append(original_doc)

        return reranked_docs


if __name__ == "__main__":
    hybrid = HybridRetriever()
    reranker = CrossEncoderReranker()

    query = "What are the requirements under § 7002 for opt-out links?"

    print(f"\n[Reranker Test] Query: '{query}'")
    print("=" * 85)

    # 1. Retrieve 10 candidates from Hybrid Retriever
    candidates = hybrid.get_relevant_documents(query, top_k=10)
    print(f"Retrieved {len(candidates)} candidates from Hybrid Search.")

    # 2. Rerank candidates down to top 4
    reranked = reranker.rerank(query=query, documents=candidates, top_k=4)

    for idx, doc in enumerate(reranked, 1):
        print(f"\n[{idx}] Parent ID    : {doc.metadata['parent_id']}")
        print(f"    Section      : § {doc.metadata['section']}")
        print(f"    Score        : {doc.metadata.get('rerank_score', 0.0):.4f}")
        print(f"    Title        : {doc.metadata['section_title']}")
        print(f"    Snippet      :\n    {doc.page_content[:200]}...\n")