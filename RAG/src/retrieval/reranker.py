from typing import List, Optional
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

from src.config import settings
from src.retrieval.hybrid import HybridRetriever


class CrossEncoderReranker:
    """Reranks retrieved candidate Documents using FlashRank cross-encoder model."""

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        cache_dir: Optional[str] = None,
    ):
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

    def rerank_with_diversity(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        max_per_section: int = 2,
        fetch_k: int = 15,
    ) -> List[Document]:
        """
        Reranks a wider candidate pool first, then filters candidates to ensure
        that no single regulatory section monopolizes the context window.
        """
        if not documents:
            return []

        # 1. Rerank a wider initial window of candidates
        raw_reranked = self.rerank(
            query=query,
            documents=documents,
            top_k=fetch_k,
        )

        # 2. Filter for section diversity
        selected_docs: List[Document] = []
        section_counts: dict = {}

        for doc in raw_reranked:
            sec = doc.metadata.get("section", "unknown")
            count = section_counts.get(sec, 0)

            if count < max_per_section:
                selected_docs.append(doc)
                section_counts[sec] = count + 1

            if len(selected_docs) == top_k:
                break

        return selected_docs


if __name__ == "__main__":
    hybrid = HybridRetriever()
    reranker = CrossEncoderReranker()

    query = "What must a contractor do if a consumer submits a CCPA deletion request directly to the contractor?"

    print(f"\n[Reranker Test] Query: '{query}'")
    print("=" * 85)

    candidates = hybrid.get_relevant_documents(query, top_k=20)
    print(f"Retrieved {len(candidates)} candidates from Hybrid Search.")

    # Test Section Diversity Reranking
    reranked = reranker.rerank_with_diversity(
        query=query,
        documents=candidates,
        top_k=5,
        max_per_section=2,
        fetch_k=15,
    )

    for idx, doc in enumerate(reranked, 1):
        print(f"\n[{idx}] Parent ID    : {doc.metadata['parent_id']}")
        print(f"    Section      : § {doc.metadata['section']}")
        print(f"    Score        : {doc.metadata.get('rerank_score', 0.0):.4f}")
        print(f"    Title        : {doc.metadata['section_title']}")
        print(f"    Snippet      :\n    {doc.page_content[:150]}...\n")