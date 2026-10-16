import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.vector_store.embeddings import get_embedding_function


class ParentChildRetriever:
    """Retrieves full parent legal sections and glossary entries based on child chunk vector similarity."""

    def __init__(
        self,
        parent_file_path: Optional[Path] = None,
        top_k_children: int = 5,
    ):
        self.parent_file_path = parent_file_path or (
            settings.PROCESSED_DATA_DIR / "parents_2026.json"
        )
        self.top_k_children = top_k_children

        # Fast lookup in memory for parent documents & § 7001 glossary entries {parent_id: dict}
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
        """Loads processed parent sections and maps § 7001 glossary definitions into memory."""
        parents_db: Dict[str, Dict[str, Any]] = {}

        # 1. Load standard parent sections (e.g., parents_2026.json)
        if self.parent_file_path.exists():
            try:
                raw_data = json.loads(self.parent_file_path.read_text(encoding="utf-8"))
                for doc in raw_data:
                    parents_db[doc["parent_id"]] = doc
            except Exception as e:
                print(f"⚠️ Error loading parent file {self.parent_file_path.name}: {e}")

        # Load parents_2023.json if present
        parents_2023_path = self.parent_file_path.parent / "parents_2023.json"
        if parents_2023_path.exists() and parents_2023_path != self.parent_file_path:
            try:
                raw_2023 = json.loads(parents_2023_path.read_text(encoding="utf-8"))
                for doc in raw_2023:
                    if doc["parent_id"] not in parents_db:
                        parents_db[doc["parent_id"]] = doc
            except Exception as e:
                pass

        # 2. Map Section 7001 Glossary entries (glossary_2026.json & glossary_2023.json)
        for g_file in ["glossary_2026.json", "glossary_2023.json"]:
            g_path = self.parent_file_path.parent / g_file
            if g_path.exists():
                try:
                    g_data = json.loads(g_path.read_text(encoding="utf-8"))
                    for item in g_data:
                        chunk_id = item.get("source_chunk_id")
                        if chunk_id and chunk_id not in parents_db:
                            version = item.get("version", "2026")
                            term_title = item.get("term", "").title()
                            parents_db[chunk_id] = {
                                "parent_id": chunk_id,
                                "section": "7001",
                                "section_title": f"Definitions ({term_title})" if term_title else "Definitions.",
                                "sub_item": item.get("section_citation"),
                                "version": version,
                                "text": f"§ 7001. Definitions.\n{item.get('definition_text', '')}",
                                "provisional_lineage_id": f"v{version}_sec7001_{item.get('term', '')}",
                                "canonical_lineage_id": None,
                            }
                except Exception as e:
                    print(f"⚠️ Error loading glossary file {g_file}: {e}")

        return parents_db

    def get_relevant_documents(
        self,
        query: str,
        version_filter: Optional[str] = None,
        max_retries: int = 3,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Queries child vectors in ChromaDB and returns resolved parent/glossary Documents."""
        effective_k = k if k is not None else self.top_k_children

        search_kwargs: Dict[str, Any] = {}
        if version_filter:
            search_kwargs["filter"] = {"version": version_filter}

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

        resolved_parents: List[Document] = []
        for pid in parent_ids:
            parent_data = self.parents_db.get(pid)
            if parent_data:
                resolved_parents.append(
                    Document(
                        page_content=parent_data["text"],
                        metadata={
                            "parent_id": parent_data["parent_id"],
                            "section": str(parent_data.get("section", "N/A")),
                            "section_title": parent_data.get("section_title", ""),
                            "sub_item": parent_data.get("sub_item"),
                            "version": parent_data.get("version", "2026"),
                            "provisional_lineage_id": parent_data.get("provisional_lineage_id", ""),
                            "canonical_lineage_id": parent_data.get("canonical_lineage_id", ""),
                            "matched_excerpt": best_child_text_by_parent.get(pid, ""),
                        },
                    )
                )
            elif "sec7001" in pid or "7001" in pid:
                # Direct § 7001 fallback directly from child chunk if unmapped
                excerpt = best_child_text_by_parent.get(pid, "")
                resolved_parents.append(
                    Document(
                        page_content=excerpt or "§ 7001. Definitions.",
                        metadata={
                            "parent_id": pid,
                            "section": "7001",
                            "section_title": "Definitions.",
                            "sub_item": None,
                            "version": "2026" if "v2026" in pid else "2023",
                            "provisional_lineage_id": pid,
                            "canonical_lineage_id": None,
                            "matched_excerpt": excerpt,
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