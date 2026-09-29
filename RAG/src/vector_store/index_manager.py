import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.schemas.chunks import ChildChunk
from src.vector_store.embeddings import get_embedding_function


class VectorIndexManager:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        self.host = host or settings.CHROMA_HOST
        self.port = port or settings.CHROMA_PORT
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        print(f"Connecting to ChromaDB at http://{self.host}:{self.port}...")

        self.client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.embedding_fn = get_embedding_function()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _sanitize_metadata(self, chunk: ChildChunk) -> Dict[str, Any]:
        """Flattens ChildChunk fields into primitive types allowed by ChromaDB."""
        raw_meta = chunk.model_dump(exclude={"chunk_id", "text"})
        return {k: v for k, v in raw_meta.items() if v is not None}

    def index_chunks(
        self,
        chunks: List[ChildChunk],
        batch_size: int = 10,
        delay_between_batches: float = 6.0,
        max_retries: int = 5,
    ) -> int:
        """Upserts ChildChunk models into ChromaDB with rate limit backoff retries."""
        if not chunks:
            print("No chunks provided for indexing.")
            return 0

        total_chunks = len(chunks)
        total_indexed = 0

        print(f"Indexing {total_chunks} chunks (batch size: {batch_size}, delay: {delay_between_batches}s)...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.text for c in batch]
            metadatas = [self._sanitize_metadata(c) for c in batch]

            # Retry loop for rate-limit protection
            for attempt in range(1, max_retries + 1):
                try:
                    self.collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    total_indexed += len(batch)
                    print(f" Progress: [{total_indexed}/{total_chunks}] chunks indexed into ChromaDB.")
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        wait_time = 45
                        print(f" ⚠️ Quota hit (429) on attempt {attempt}/{max_retries}. Pausing for {wait_time}s to reset Google API quota...")
                        time.sleep(wait_time)
                    else:
                        raise e
            else:
                raise RuntimeError(f"Failed to index batch starting at index {i} after {max_retries} attempts.")

            # Sleep between batches to stay under the 100 RPM limit (10 items / 6s = ~100 RPM)
            if i + batch_size < total_chunks and delay_between_batches > 0:
                time.sleep(delay_between_batches)

        return total_indexed

    def delete_version(self, version_tag: str) -> None:
        """Purges all chunks associated with a specific regulatory version."""
        self.collection.delete(where={"version": version_tag})

    def count(self) -> int:
        """Returns total records indexed in the collection."""
        return self.collection.count()


def load_and_index_file(json_file: Path, index_mgr: VectorIndexManager):
    if not json_file.exists():
        print(f"File not found: {json_file}")
        return

    raw_data = json.loads(json_file.read_text(encoding="utf-8"))
    chunks = [ChildChunk(**item) for item in raw_data]

    indexed_count = index_mgr.index_chunks(chunks)
    print(f"\nSuccessfully indexed {indexed_count} child chunks from [{json_file.name}] into ChromaDB.")


if __name__ == "__main__":
    CHUNKS_FILE = settings.PROCESSED_DATA_DIR / "chunks_2026.json"

    manager = VectorIndexManager()
    load_and_index_file(CHUNKS_FILE, manager)
    print(f"Total items in collection '{manager.collection_name}': {manager.count()}")