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
        self, query: str, version_filter: Optional[str] = None, max_retries: int = 3
    ) -> List[Document]:
        """Queries child vectors in ChromaDB and returns full parent section Documents."""
        search_kwargs: Dict[str, Any] = {}
        if version_filter:
            search_kwargs["filter"] = {"version": version_filter}

        # 1. Similarity search on fine-grained child chunks with 429 resilience
        child_docs = []
        for attempt in range(1, max_retries + 1):
            try:
                child_docs = self.vector_store.similarity_search(
                    query=query,
                    k=self.top_k_children,
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

        # 2. Collect unique parent IDs while maintaining relevance order
        parent_ids: List[str] = []
        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in parent_ids:
                parent_ids.append(pid)

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