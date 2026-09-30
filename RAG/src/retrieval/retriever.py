import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.vector_store.embeddings import get_embedding_function


class ParentChildRetriever:
    """Retrieves full parent legal sections based on child chunk vector similarity across regulatory versions."""

    def __init__(
        self,
        processed_data_dir: Optional[Path] = None,
        top_k_children: int = 5,
    ):
        self.processed_data_dir = processed_data_dir or settings.PROCESSED_DATA_DIR
        self.top_k_children = top_k_children

        # Fast in-memory lookup for parent documents across ALL version files {parent_id: dict}
        self.parents_db: Dict[str, Dict[str, Any]] = self._load_all_parents()

        # Connect to ChromaDB via HttpClient
        chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        chroma_ef = get_embedding_function()

        self.vector_store = Chroma(
            client=chroma_client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=chroma_ef,
        )

    def _load_all_parents(self) -> Dict[str, Dict[str, Any]]:
        """Loads all processed parent files (parents_*.json) in data/processed/ into memory."""
        parents_map: Dict[str, Dict[str, Any]] = {}
        parent_files = list(self.processed_data_dir.glob("parents_*.json"))

        if not parent_files:
            print(f"⚠️ Warning: No parent files found matching 'parents_*.json' in {self.processed_data_dir}")
            return parents_map

        for p_file in parent_files:
            try:
                raw_data = json.loads(p_file.read_text(encoding="utf-8"))
                for doc in raw_data:
                    parents_map[doc["parent_id"]] = doc
            except Exception as e:
                print(f"⚠️ Error loading parent file {p_file.name}: {e}")

        return parents_map

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
                    wait_time = 10 * attempt
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

        # 3. Resolve parent chunks using standard schema keys
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
                            "version": parent_data.get("version", "N/A"),
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
        print(f"    Version      : {doc.metadata['version']}")
        print(f"    Section      : § {doc.metadata['section']}")
        print(f"    Sub-item     : {doc.metadata['sub_item'] or 'N/A'}")
        print(f"    Title        : {doc.metadata['section_title']}")
        print(f"    Lineage ID   : {doc.metadata['canonical_lineage_id'] or doc.metadata['provisional_lineage_id']}")
        print(f"    Snippet      :\n    {doc.page_content[:200]}...\n")