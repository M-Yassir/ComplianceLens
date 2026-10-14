import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.schemas.chunks import ParentChunk
from src.vector_store.embeddings import get_embedding_function


class ParentChildRetriever:
    """Retrieves full parent legal sections based on child chunk vector similarity."""

    def __init__(
        self,
        parent_file_path: Optional[Path] = None,
        top_k_children: int = 5,
    ):
        self.parent_file_path = parent_file_path or (
            settings.PROCESSED_DATA_DIR / "parents_2026.json"
        )
        self.top_k_children = top_k_children

        # Fast lookup in memory for parent documents {parent_id: dict}
        self.parents_db: Dict[str, Dict[str, Any]] = self._load_parents()

        # Persistent ChromaDB connection via vector adapter
        chroma_ef = get_embedding_function()
        self.vector_store = Chroma(
            client_settings=None,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=chroma_ef,
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )

    def _load_parents(self) -> Dict[str, Dict[str, Any]]:
        """Loads processed parent sections into memory for immediate resolution."""
        if not self.parent_file_path.exists():
            raise FileNotFoundError(
                f"Parent documents payload not found at: {self.parent_file_path}"
            )

        raw_data = json.loads(self.parent_file_path.read_text(encoding="utf-8"))
        return {doc["parent_id"]: doc for doc in raw_data}

    def get_relevant_documents(
        self,
        query: str,
        version_filter: Optional[str] = None,
        max_retries: int = 3,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Queries child vectors in ChromaDB and returns full parent section Documents.

        `k` overrides how many child chunks are searched for this call
        (falls back to `self.top_k_children` when omitted). This exists
        so callers that want a wider dense candidate pool - notably
        HybridRetriever, which otherwise ends up fusing a ~5-chunk dense
        result against an 80-chunk BM25 result for the same query - can
        ask for one without constructing a second ParentChildRetriever
        instance just to change this number.
        """
        effective_k = k if k is not None else self.top_k_children

        search_kwargs: Dict[str, Any] = {}
        if version_filter:
            search_kwargs["filter"] = {"version": version_filter}

        # 1. Similarity search on fine-grained child chunks with 429 resilience
        child_docs = []
        for attempt in range(1, max_retries + 1):
            try:
                child_docs = self.vector_store.similarity_search(
                    query=query,
                    k=effective_k,
                    **search_kwargs,
                )
                break
            except Exception as e:
                err_msg = str(e)
                if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries:
                    wait_time = 45
                    print(f"\n Rate limit hit (429). Waiting {wait_time}s for Google API quota reset (Attempt {attempt}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e

        if not child_docs:
            return []

        # 2. Collect unique parent IDs while maintaining relevance order, and
        # keep each parent's BEST-matching child chunk text alongside it.
        # This is carried through to the reranker (see reranker.py's
        # `_build_passage_text`): the cross-encoder it uses truncates each
        # (query, passage) pair to 512 tokens total, truncating whichever
        # side is longer - almost always the ~1000-1200 token parent text,
        # keeping only its opening ~480 tokens. A full parent passed to
        # reranking risks having its actually-relevant content silently cut
        # off before scoring; this ~250-token child excerpt (the specific
        # text that matched the query) stays well under that limit and is
        # already the relevant slice, so it's a strictly better rerank
        # input than the full parent body. The full parent text is still
        # what's returned as `page_content` - this only adds an extra,
        # shorter field alongside it.
        parent_ids: List[str] = []
        best_child_text_by_parent: Dict[str, str] = {}
        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if not pid:
                continue
            if pid not in parent_ids:
                parent_ids.append(pid)
            if pid not in best_child_text_by_parent:
                best_child_text_by_parent[pid] = doc.page_content

        # 3. Resolve parent chunks using standard BaseLegalChunk schema keys
        resolved_parents: List[Document] = []
        for pid in parent_ids:
            parent_data = self.parents_db.get(pid)
            if parent_data:
                resolved_parents.append(
                    Document(
                        page_content=parent_data["text"],
                        metadata={
                            "parent_id": parent_data["parent_id"],
                            "section": parent_data.get("section", "N/A"),
                            "section_title": parent_data.get("section_title", ""),
                            "sub_item": parent_data.get("sub_item"),
                            "version": parent_data.get("version", "2026"),
                            "provisional_lineage_id": parent_data.get("provisional_lineage_id", ""),
                            "canonical_lineage_id": parent_data.get("canonical_lineage_id", ""),
                            "matched_excerpt": best_child_text_by_parent.get(pid, ""),
                        },
                    )
                )

        return resolved_parents


if __name__ == "__main__":
    retriever = ParentChildRetriever(top_k_children=4)
    query = "What are the requirements for consumer opt-out links?"

    print(f"\n[Retriever Test] Query: '{query}'")
    print("=" * 85)

    results = retriever.get_relevant_documents(query)
    for idx, doc in enumerate(results, 1):
        print(f"\n[{idx}] Parent ID    : {doc.metadata['parent_id']}")
        print(f"    Section      : § {doc.metadata['section']}")
        print(f"    Sub-item     : {doc.metadata['sub_item'] or 'N/A'}")
        print(f"    Title        : {doc.metadata['section_title']}")
        print(f"    Lineage ID   : {doc.metadata['provisional_lineage_id']}")
        print(f"    Snippet      :\n    {doc.page_content[:200]}...\n")