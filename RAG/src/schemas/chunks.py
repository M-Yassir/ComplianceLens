from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# Explicit alignment statuses for full structural change tracking
AlignmentStatus = Literal[
    "UNCHANGED", 
    "MODIFIED", 
    "ADDED", 
    "DELETED", 
    "RELETTERED", 
    "SPLIT", 
    "MERGED"
]


class BaseLegalChunk(BaseModel):
    """Base metadata fields shared across all chunk representations."""
    version: str = Field(
        ..., 
        description="Document version identifier or effective date (e.g., '2023-03', '2026-09')"
    )
    section: str = Field(
        ..., 
        description="Statutory section number (e.g., '7001', '7123')"
    )
    sub_item: Optional[str] = Field(
        default=None, 
        description="Clause/sub-item identifier (e.g., '(e)(3)')"
    )
    section_title: str = Field(
        ..., 
        description="Parent section header title (e.g., 'Definitions')"
    )
    provisional_lineage_id: str = Field(
        ..., 
        description="Positional label assigned during initial chunking (e.g., 'v2026_sec7001_sub_e')"
    )


class ChildChunk(BaseLegalChunk):
    """Leaf chunk stored and indexed in Vector DB for similarity retrieval."""
    chunk_id: str = Field(
        ..., 
        description="Unique primary key for vector database storage (e.g., 'v2026_sec7001_sub_e_1')"
    )
    parent_id: str = Field(
        ..., 
        description="Foreign key referencing the associated ParentChunk"
    )
    text: str = Field(
        ..., 
        description="Embedded text body, prepended with parent section header for context"
    )
    token_count: int = Field(
        ..., 
        description="Exact BPE token count of this child chunk"
    )
    is_fallback_split: bool = Field(
        default=False, 
        description="Flags whether recursive character splitting was used due to lack of sub-structures"
    )
    canonical_lineage_id: Optional[str] = Field(
        default=None, 
        description="Resolved semantic identity written back post-alignment for zero-join vector lookups"
    )


class ParentChunk(BaseLegalChunk):
    """Broader context block fetched during retrieval and passed to the LLM prompt."""
    parent_id: str = Field(
        ..., 
        description="Unique primary key for parent payload KV store"
    )
    text: str = Field(
        ..., 
        description="Full multi-clause parent text payload provided to the LLM"
    )
    token_count: int = Field(
        ..., 
        description="Exact BPE token count of parent payload"
    )
    child_ids: List[str] = Field(
        default_factory=list, 
        description="List of child chunk IDs contained within this parent window"
    )


class GlossaryEntry(BaseModel):
    """Term definition extracted from § 7001 used for side-car prompt injection."""
    version: str = Field(
        ..., 
        description="Document version identifier or effective date (e.g., '2023-03', '2026-09')"
    )
    term: str = Field(
        ..., 
        description="Canonical term name in lowercase (e.g., 'automated decisionmaking technology')"
    )
    aliases: List[str] = Field(
        default_factory=list, 
        description="Abbreviations or alternate representations (e.g., ['ADMT'])"
    )
    section_citation: str = Field(
        ..., 
        description="Exact citation source (e.g., '§ 7001(e)')"
    )
    definition_text: str = Field(
        ..., 
        description="Complete, unabridged definition text body"
    )
    is_distinctive: bool = Field(
        ..., 
        description="True for multi-word/rare terms matched via substring; False for generic terms requiring exact quoted matching"
    )
    source_chunk_id: str = Field(
        ..., 
        description="Foreign key referencing underlying ChildChunk ID for automated lineage inheritance"
    )


class LineagePairwiseResolution(BaseModel):
    """Alignment record comparing any two arbitrary versions (e.g., 2023 vs 2026 or 2021 vs 2023)."""
    canonical_lineage_id: str = Field(
        ..., 
        description="Persistent identity mapping the same underlying provision across versions"
    )
    source_version: str = Field(
        ..., 
        description="Baseline version identifier (e.g., '2023-03')"
    )
    source_chunk_id: Optional[str] = Field(
        default=None, 
        description="ChildChunk ID in the source version (None if added in target version)"
    )
    target_version: str = Field(
        ..., 
        description="Compared version identifier (e.g., '2026-09')"
    )
    target_chunk_id: Optional[str] = Field(
        default=None, 
        description="ChildChunk ID in the target version (None if deleted from source version)"
    )
    alignment_status: AlignmentStatus = Field(
        ..., 
        description="Categorical status representing structural/semantic transformation"
    )
    similarity_score: Optional[float] = Field(
        default=None, 
        description="Cosine similarity score produced by aligner.py (None for pure ADDED/DELETED actions)"
    )


class LineageHistoryRecord(BaseModel):
    """Derived, read-only materialized view tracking a single provision's chunk IDs 
    across all historical versions.
    
    INVARIANTS:
    1. Always strictly 1:1 (one chunk_id per version key).
    2. On SPLIT: The pre-split lineage terminates at source_version. New branch 
       lineages are minted for target_version. Provenance is preserved via 
       LineagePairwiseResolution rows with alignment_status='SPLIT'.
    3. On MERGED: The primary lineage continues into target_version; secondary 
       lineages terminate at source_version. Provenance is preserved via 
       LineagePairwiseResolution rows with alignment_status='MERGED'.
    4. Programmatically derived from LineagePairwiseResolution—never edited directly.
    """
    canonical_lineage_id: str = Field(
        ..., 
        description="Global persistent identity for a specific legal provision timeline"
    )
    version_chunk_map: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Strict 1:1 mapping of version tags to chunk IDs (e.g., {'2021': 'c_1', '2023': 'c_2', '2026': 'c_5'})"
    )