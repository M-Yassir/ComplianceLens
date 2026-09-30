import argparse
import json
import sys
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError

from src.config import settings
from src.vector_store.embeddings import get_embedding_function


def get_collection():
    client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    try:
        return client.get_collection(name=settings.CHROMA_COLLECTION_NAME)
    except (NotFoundError, ValueError):
        print(f"\n❌ Error: Collection '{settings.CHROMA_COLLECTION_NAME}' does not exist in ChromaDB.")
        print("💡 Run the index manager first to populate the database:")
        print("   python -m src.vector_store.index_manager\n")
        sys.exit(1)


def peek_records(limit: int = 3, full_vectors: bool = False):
    col = get_collection()
    
    data = col.get(
        limit=limit,
        include=["documents", "metadatas", "embeddings"]
    )

    print(f"\n Total Count in '{settings.CHROMA_COLLECTION_NAME}': {col.count()}")
    print("=" * 85)

    for idx in range(len(data["ids"])):
        cid = data["ids"][idx]
        doc = data["documents"][idx]
        meta = data["metadatas"][idx]
        emb = data["embeddings"][idx] if data.get("embeddings") is not None else []

        print(f"\n Record [{idx + 1}/{limit}] ID: {cid}")
        print(f" Metadata: {json.dumps(meta, indent=2)}")
        
        if len(emb) > 0:
            print(f" Embedding Vector ({len(emb)} Dimensions):")
            if full_vectors:
                print(f"   {emb}")
            else:
                preview = [f"{x:.6f}" for x in emb[:5]]
                end_preview = [f"{x:.6f}" for x in emb[-3:]]
                print(f"   [{', '.join(preview)}, ... {', '.join(end_preview)}]")
        else:
            print(" Embedding Vector: [None]")

        print(f" Text Payload:\n{doc}")
        print("-" * 85)


def get_by_id(chunk_id: str):
    col = get_collection()
    data = col.get(ids=[chunk_id], include=["documents", "metadatas", "embeddings"])

    if not data["ids"]:
        print(f"\n Record with ID '{chunk_id}' not found.")
        return

    emb = data["embeddings"][0] if data.get("embeddings") is not None else []

    print(f"\n Record Details for ID: {chunk_id}")
    print("=" * 85)
    print(f"Metadata:\n{json.dumps(data['metadatas'][0], indent=2)}")
    
    if len(emb) > 0:
        print(f"\nEmbedding Vector ({len(emb)} Dimensions):")
        print(f"{emb}")

    print(f"\nText Payload:\n{data['documents'][0]}")


def query_vector_db(query_text: str, top_k: int = 3):
    col = get_collection()
    embedding_fn = get_embedding_function()
    
    # Generate explicit 1D embedding vector using LangChain interface
    query_embedding = embedding_fn.embed_query(query_text)

    # Pass pre-computed embeddings directly to ChromaDB
    res = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\n Similarity Search Results for: '{query_text}'")
    print("=" * 85)

    for idx in range(len(res["ids"][0])):
        cid = res["ids"][0][idx]
        doc = res["documents"][0][idx]
        dist = res["distances"][0][idx]
        meta = res["metadatas"][0][idx]

        print(f"\n[{idx + 1}] Distance (Cosine): {dist:.4f} | ID: {cid}")
        print(f"    Section : § {meta.get('section', 'N/A')}")
        print(f"    Title   : {meta.get('section_title', 'N/A')}")
        print(f"    Parent  : {meta.get('parent_id', 'N/A')}")
        print(f"    Text    : {doc[:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCPA Vector DB Terminal Inspector")
    parser.add_argument("--peek", type=int, nargs="?", const=3, help="Peek first N records (default: 3)")
    parser.add_argument("--full-vector", action="store_true", help="Display full un-truncated vector arrays")
    parser.add_argument("--id", type=str, help="Fetch a single record by its chunk_id")
    parser.add_argument("--query", type=str, help="Run a semantic search query against ChromaDB")

    args = parser.parse_args()

    if args.id:
        get_by_id(args.id)
    elif args.query:
        query_vector_db(args.query)
    else:
        limit = args.peek if args.peek else 3
        peek_records(limit=limit, full_vectors=args.full_vector)