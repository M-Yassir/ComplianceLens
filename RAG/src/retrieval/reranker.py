from typing import List, Optional
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

from src.config import settings
from src.retrieval.hybrid import HybridRetriever


# ms-marco-MiniLM-L-12-v2 (and FlashRank's cross-encoders generally) tokenize
# `[query, passage]` as a PAIR and truncate the pair to 512 tokens total.
# FlashRank's Ranker constructs its tokenizer via
# `tokenizer.enable_truncation(max_length=512)` with no strategy/direction
# override, so it inherits the `tokenizers` library's defaults:
# strategy="longest_first" (truncates whichever of query/passage is longer -
# in practice always the passage) and direction="right" (keeps the first
# ~480 tokens, discards the rest). A parent chunk running ~1000-1200 tokens
# therefore gets roughly its back half silently dropped BEFORE the
# cross-encoder ever scores it - which can easily be exactly the part that
# would have shown the passage was relevant. This is a model/tokenizer
# limit, not something `max_length=` can raise past (the underlying
# position embeddings are capped there too), so the fix is on our side:
# don't hand it more passage than it can actually see.
_RERANK_PASSAGE_CHAR_BUDGET = 1600  # ~400 tokens at ~4 chars/token - safety
# margin under the ~480-token budget left after the query + special tokens,
# used only as a fallback when no shorter matched excerpt is available.


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

    def _build_passage_text(self, doc: Document) -> str:
        """Text actually handed to the cross-encoder for scoring - kept
        short enough that FlashRank's own truncation can't silently drop
        the substantive part of it (see module docstring above).

        Prefers `matched_excerpt` - the specific ~250-token child chunk
        that caused this parent to be retrieved in the first place
        (populated by ParentChildRetriever / HybridRetriever's BM25 path) -
        over the full parent `page_content`, since it's both short enough
        to fit and already the query-relevant slice rather than the
        section's full, mostly-boilerplate body. Falls back to a bounded
        prefix of the parent text if no excerpt was carried through (e.g.
        documents built some other way).
        """
        header = f"§ {doc.metadata.get('section', '')} {doc.metadata.get('section_title', '')}"
        excerpt = doc.metadata.get("matched_excerpt") or ""
        if not excerpt.strip():
            excerpt = doc.page_content[:_RERANK_PASSAGE_CHAR_BUDGET]
        return f"{header}\n{excerpt}"

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
                "text": self._build_passage_text(doc),
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
    query3 = "If a business never audits or checks if its vendor is following privacy rules, can the business defend itself by claiming it had no reason to believe the vendor was violating the law?"
    query4 = "Is a third-party analytics vendor allowed to take customer data obtained from one client and combine it with data collected from its own website visitors to update user profiles?"
    query5 = "How frequently is a business allowed to conduct technical or operational testing on a contractor's systems to ensure compliance?"
    query6 = "An e-commerce platform verifies a mother's identity to grant consent for her 10-year-old child by checking her driver's license against a state database. The company retains a copy of her driver's license in its secure legal archives for auditing purposes. Is this practice compliant?"
    query7 = "A consumer who participates in a retail rewards program—earning coupons based on cumulative spending—submits a request to delete all personal information, but states they want to stay in the rewards program. Can the retailer refuse to delete their email address and purchase history?"
    query8 = "If a large digital enterprise processing data for 12 million individuals compiles its annual privacy metrics using global request totals rather than isolating California consumer requests, does this satisfy state reporting requirements?"
    query9 = "A distributed denial-of-service (DDoS) attack takes down a business's customer portal for 48 hours, preventing users from accessing their account details, but no data is exfiltrated or modified. Does this incident fall within the required scope of the cybersecurity audit assessment?"
    query10 = "By what calendar date must high-volume consumer data platforms publish their annual metric disclosures, and what specific automated decision-making technology (ADMT) metrics must be included?"
    
    query = query10
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