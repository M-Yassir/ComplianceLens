import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel

from src.schemas.chunks import ChildChunk


class VectorIndexManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "ccpa_child_chunks",
    ):
        self.client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _sanitize_metadata(self, chunk: ChildChunk) -> Dict[str, Any]:
        """Flattens ChildChunk fields into primitive types allowed by ChromaDB (str, int, float, bool)."""
        raw_meta = chunk.model_dump(
            exclude={"chunk_id", "text"}  # chunk_id is used as DB ID, text is document payload
        )
        clean_meta = {}
        for k, v in raw_meta.items():
            if v is None:
                continue
            clean_meta[k] = v
        return clean_meta

    def index_chunks(self, chunks: List[ChildChunk]) -> int:
        """Upserts a list of ChildChunk models into ChromaDB."""
        if not chunks:
            print("No chunks provided for indexing.")
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._sanitize_metadata(c) for c in chunks]

        # ChromaDB handles batching internally or via explicit upsert
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def delete_version(self, version_tag: str) -> None:
        """Purges all chunks associated with a specific regulatory version."""
        self.collection.delete(where={"version": version_tag})

    def count(self) -> int:
        """Returns total records indexed in the collection."""
        return self.collection.count()


# =====================================================================
# Pipeline Execution
# =====================================================================
def load_and_index_file(json_file: Path, index_mgr: VectorIndexManager):
    if not json_file.exists():
        print(f"File not found: {json_file}")
        return

    raw_data = json.loads(json_file.read_text(encoding="utf-8"))
    chunks = [ChildChunk(**item) for item in raw_data]

    indexed_count = index_mgr.index_chunks(chunks)
    print(f" Successfully indexed {indexed_count} child chunks from [{json_file.name}] into ChromaDB.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    CHUNKS_FILE = PROCESSED_DIR / "chunks_2026.json"

    manager = VectorIndexManager()
    load_and_index_file(CHUNKS_FILE, manager)
    print(f"Total items in collection '{manager.collection_name}': {manager.count()}")