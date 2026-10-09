import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import settings
from src.schemas.versioning import LineageRecord, VerifiedChangelogCard
from src.versioning.aligner import TwoPassAligner
from src.versioning.card_generator import ChangelogCardGenerator


def load_processed_json(file_path: Path) -> List[Dict[str, Any]]:
    """Loads a processed version JSON file from data/processed/."""
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Normalize chunk objects to ensure citation and metadata fields exist
    normalized = []
    for chunk in data:
        section = chunk.get("section")
        sub_item = chunk.get("sub_item")
        
        if not chunk.get("citation"):
            if section and section != "0000":
                chunk["citation"] = f"§ {section}{sub_item}" if sub_item else f"§ {section}"
            else:
                chunk["citation"] = chunk.get("section_title") or "Preamble"
                
        normalized.append(chunk)
        
    return normalized


class VersioningPipeline:
    """
    End-to-end orchestrator for cross-version alignment and changelog card generation.
    """

    def __init__(
        self,
        aligner: Optional[TwoPassAligner] = None,
        card_generator: Optional[ChangelogCardGenerator] = None,
    ):
        self.aligner = aligner or TwoPassAligner()
        self.card_generator = card_generator or ChangelogCardGenerator()

    def run_alignment(
        self,
        source_version: str = "2023",
        target_version: str = "2026",
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Executes offline alignment across source and target processed JSON files.
        """
        out_dir = output_dir or settings.PROCESSED_DATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        source_path = out_dir / f"version_{source_version}.json"
        target_path = out_dir / f"version_{target_version}.json"

        print(f"Loading source version ({source_version}) from: {source_path}")
        source_chunks = load_processed_json(source_path)

        print(f"Loading target version ({target_version}) from: {target_path}")
        target_chunks = load_processed_json(target_path)

        # 1. Run Two-Pass Alignment
        print("\n[Pass 1 & 2] Executing statutory alignment...")
        records: List[LineageRecord] = self.aligner.align(
            source_chunks=source_chunks,
            target_chunks=target_chunks,
            source_version=source_version,
            target_version=target_version,
        )
        print(f"Alignment complete: {len(records)} lineage records generated.")

        # Map chunks by chunk_id for quick lookup during card generation
        source_by_id = {c["chunk_id"]: c for c in source_chunks if "chunk_id" in c}
        target_by_id = {c["chunk_id"]: c for c in target_chunks if "chunk_id" in c}

        # 2. Generate Verified Changelog Cards
        print("\n[Card Generation & Verification] Processing changelog summaries...")
        verified_cards: List[VerifiedChangelogCard] = []

        for idx, rec in enumerate(records, 1):
            src_chunk = source_by_id.get(rec.source_chunk_id) if rec.source_chunk_id else None
            tgt_chunk = target_by_id.get(rec.target_chunk_id) if rec.target_chunk_id else None

            print(
                f"  ({idx}/{len(records)}) [{rec.change_type.value.upper()}] "
                f"Lineage: {rec.canonical_lineage_id} "
                f"({rec.source_citation or 'N/A'} -> {rec.target_citation or 'N/A'})"
            )

            verified_card = self.card_generator.generate_card(
                record=rec,
                source_chunk=src_chunk,
                target_chunk=tgt_chunk,
            )
            verified_cards.append(verified_card)

        # 3. Save JSON Artifacts
        records_output_path = out_dir / f"lineage_records_{source_version}_{target_version}.json"
        cards_output_path = out_dir / f"changelog_cards_{source_version}_{target_version}.json"

        with open(records_output_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in records], f, indent=2)

        with open(cards_output_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in verified_cards], f, indent=2)

        print("\n" + "=" * 80)
        print("Pipeline Execution Complete!")
        print(f"Saved Lineage Records : {records_output_path}")
        print(f"Saved Verified Cards  : {cards_output_path}")
        print("=" * 80)

        return {
            "records_count": len(records),
            "cards_count": len(verified_cards),
            "records_file": str(records_output_path),
            "cards_file": str(cards_output_path),
        }


if __name__ == "__main__":
    pipeline = VersioningPipeline()
    pipeline.run_alignment(source_version="2023", target_version="2026")