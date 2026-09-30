import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.config import settings
from src.retrieval.retriever import ParentChildRetriever


def _tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase words/numbers for BM25 indexing."""
    return re.findall(r"\b\w+\b", text.lower())


def _extract_section_citations(query: str) -> List[str]:
    """Extracts explicit section numbers from user queries (e.g. '§ 7002' or 'Section 7002' -> ['7002'])."""
    matches = re.findall(r"(?:§|section)\s*(\d+[a-z]?)", query, re.IGNORECASE)
    return [m.lower() for m in matches]


class HybridRetriever:
    """Hybrid Retriever combining Dense Vector Search and Sparse BM25 via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense_retriever: Optional[ParentChildRetriever] = None,
        processed_data_dir: Optional[Path] = None,
    ):
        self.processed_data_dir = processed_data_dir or settings.PROCESSED_DATA_DIR
        self.dense_retriever = dense_retriever or ParentChildRetriever(
            processed_data_dir=self.processed_data_dir
        )

        # In-memory store of child chunks for BM25 indexing
        self.child_chunks: List[Dict[str, Any]] = self._load_all_child_chunks()

        # Build BM25 index over child chunk text ENRICHED with section number and title
        tokenized_corpus = [
            _tokenize(f"§ {c.get('section', '')} {c.get('section_title', '')} {c.get('text', '')}")
            for c in self.child_chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def _load_all_child_chunks(self) -> List[Dict[str, Any]]:
        """Loads all processed child chunk files (chunks_*.json) into memory for BM25."""
        child_chunks: List[Dict[str, Any]] = []
        chunk_files = list(self.processed_data_dir.glob("chunks_*.json"))

        if not chunk_files:
            print(f"⚠️️ Warning: No child chunk files found matching 'chunks_*.json' in {self.processed_data_dir}")
            return child_chunks

        for c_file in chunk_files:
            try:
                raw_data = json.loads(c_file.read_text(encoding="utf-8"))
                child_chunks.extend(raw_data)
            except Exception as e:
                print(f"⚠️ Error loading child chunk file {c_file.name}: {e}")

        return child_chunks

    def _get_bm25_parent_documents(
        self, query: str, top_k_children: int = 20, version_filter: Optional[str] = None
    ) -> List[Document]:
        """Performs BM25 search over section-enriched child chunks."""
        if not self.bm25 or not self.child_chunks:
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Pair indices with scores
        scored_indices = list(enumerate(scores))

        # Filter by regulatory version if requested
        if version_filter:
            scored_indices = [
                (idx, score)
                for idx, score in scored_indices
                if self.child_chunks[idx].get("version") == version_filter
            ]

        # Sort by score descending
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        # Extract top child candidates and deduplicate parent IDs
        top_child_indices = scored_indices[:top_k_children]
        parent_ids: List[str] = []

        for idx, score in top_child_indices:
            if score <= 0:
                continue
            child = self.child_chunks[idx]
            pid = child.get("parent_id")
            if pid and pid not in parent_ids:
                parent_ids.append(pid)

        # Resolve parent chunks into LangChain Document objects
        resolved_parents: List[Document] = []
        for pid in parent_ids:
            parent_data = self.dense_retriever.parents_db.get(pid)
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

    def _reciprocal_rank_fusion(
        self,
        dense_docs: List[Document],
        bm25_docs: List[Document],
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> List[Document]:
        """Fuses ranked document lists using RRF, with explicit section citation boosting."""
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # 1. Score Dense candidates
        for rank, doc in enumerate(dense_docs, start=1):
            pid = doc.metadata["parent_id"]
            doc_map[pid] = doc
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (rrf_k + rank))

        # 2. Score BM25 candidates
        for rank, doc in enumerate(bm25_docs, start=1):
            pid = doc.metadata["parent_id"]
            if pid not in doc_map:
                doc_map[pid] = doc
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (rrf_k + rank))

        # 3. Citation Boosting: If query explicitly asks for a section (e.g. "§ 7002"), boost matching sections
        target_sections = _extract_section_citations(query)
        if target_sections:
            for pid, doc in doc_map.items():
                doc_section = str(doc.metadata.get("section", "")).lower().strip()
                if any(ts == doc_section or f"§ {ts}" in doc_section for ts in target_sections):
                    # Apply a massive boost for direct section citation matches
                    rrf_scores[pid] += 2.0

        # 4. Sort by combined RRF score descending
        sorted_pids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)

        return [doc_map[pid] for pid in sorted_pids[:top_k]]

    def get_relevant_documents(
        self,
        query: str,
        version_filter: Optional[str] = None,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> List[Document]:
        """Executes dense + BM25 search and returns RRF-fused top_k parent documents."""
        dense_docs = self.dense_retriever.get_relevant_documents(
            query=query, version_filter=version_filter
        )

        bm25_docs = self._get_bm25_parent_documents(
            query=query, top_k_children=top_k * 4, version_filter=version_filter
        )

        return self._reciprocal_rank_fusion(
            dense_docs=dense_docs,
            bm25_docs=bm25_docs,
            query=query,
            top_k=top_k,
            rrf_k=rrf_k,
        )


if __name__ == "__main__":
    hybrid = HybridRetriever()
    query = "What are the requirements under § 7002 for opt-out links?"

    print(f"\n[Hybrid Retriever Test] Query: '{query}'")
    print("=" * 85)

    results = hybrid.get_relevant_documents(query, top_k=4)
    for idx, doc in enumerate(results, 1):
        print(f"\n[{idx}] Parent ID    : {doc.metadata['parent_id']}")
        print(f"    Version      : {doc.metadata['version']}")
        print(f"    Section      : § {doc.metadata['section']}")
        print(f"    Sub-item     : {doc.metadata['sub_item'] or 'N/A'}")
        print(f"    Title        : {doc.metadata['section_title']}")
        print(f"    Snippet      :\n    {doc.page_content[:200]}...\n")