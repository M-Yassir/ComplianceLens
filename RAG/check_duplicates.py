import json
from collections import Counter
from pathlib import Path

# Path to your generated chunks file
chunks_file = Path("data/processed/chunks_2026.json")

if not chunks_file.exists():
    print(f"Error: Could not find {chunks_file}")
else:
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    ids = [c["chunk_id"] for c in chunks]

    counts = Counter(ids)
    duplicates = {chunk_id: count for chunk_id, count in counts.items() if count > 1}

    print("=" * 60)
    print(f"Total chunk records : {len(ids)}")
    print(f"Unique chunk IDs    : {len(counts)}")
    print("=" * 60)

    if duplicates:
        print(f"\n⚠️ FOUND {len(duplicates)} DUPLICATE CHUNK ID(S):\n")
        for chunk_id, count in duplicates.items():
            print(f"  • ID '{chunk_id}' appears {count} times")

        print("\n" + "-" * 60)
        print("INSPECTING DUPLICATE CONTENTS:")
        print("-" * 60)
        for chunk_id in duplicates:
            matches = [c for c in chunks if c["chunk_id"] == chunk_id]
            for idx, item in enumerate(matches, 1):
                preview = item["text"][:100].replace("\n", " ")
                print(f"[{idx}] ID: {chunk_id}")
                print(f"    Citation : § {item['section']}{item.get('sub_item') or ''}")
                print(f"    Text     : {preview}...\n")
    else:
        print("\n All chunk IDs are 100% unique!")