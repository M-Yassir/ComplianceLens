from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    RENUMBERED = "renumbered"
    SPLIT = "split"
    MERGED = "merged"


class LineageRecord(BaseModel):
    """Tracks structural mapping between two versions of a regulatory provision."""

    canonical_lineage_id: str = Field(
        ...,
        description="Unique lineage ID for this specific provision branch.",
    )
    parent_lineage_id: Optional[str] = Field(
        None,
        description="Lineage ID of origin provision if this branch was created via a SPLIT.",
    )
    source_version: str = Field(..., description="Version label for source text (e.g. '2023-03').")
    target_version: str = Field(..., description="Version label for target text (e.g. '2026-09').")
    source_chunk_id: Optional[str] = Field(
        None,
        description="Primary key / ID of source vector store chunk. None for ADDED.",
    )
    target_chunk_id: Optional[str] = Field(
        None,
        description="Primary key / ID of target vector store chunk. None for DELETED.",
    )
    source_citation: Optional[str] = Field(
        None,
        description="Human-readable citation in source version (e.g. '§ 7123(c)(1)').",
    )
    target_citation: Optional[str] = Field(
        None,
        description="Human-readable citation in target version (e.g. '§ 7123(c)(1)(A)').",
    )
    change_type: ChangeType = Field(..., description="Classified change category.")
    similarity_score: Optional[float] = Field(
        None,
        description="Composite score (dense + lexical). None for ADDED/DELETED.",
    )
    lexical_overlap_score: Optional[float] = Field(
        None,
        description="Token/Jaccard/BM25 overlap score. None for ADDED/DELETED.",
    )


class ChangelogCard(BaseModel):
    """Unverified LLM-generated changelog summary produced from raw difflib output."""

    card_id: str = Field(..., description="Unique identifier for this card.")
    canonical_lineage_id: str = Field(..., description="Lineage ID tied to this change.")
    source_version: str
    target_version: str
    source_citation: Optional[str] = None
    target_citation: Optional[str] = None
    change_type: ChangeType
    executive_summary: str = Field(
        ...,
        description="1-2 sentence high-level summary of the regulatory change.",
    )
    substantive_changes: List[str] = Field(
        default_factory=list,
        description="Bullet points of specific new, modified, or removed legal obligations.",
    )
    impact_level: str = Field(
        ...,
        description="Qualitative assessment of legal impact: 'HIGH', 'MEDIUM', or 'LOW'.",
    )
    raw_diff_summary: str = Field(
        ...,
        description="Ground-truth text diff produced deterministically by difflib.",
    )


class VerifiedChangelogCard(ChangelogCard):
    """Grounded changelog card that has passed the verification gate and is safe for indexing."""

    is_verified: bool = Field(
        True,
        description="Set to True only when all claims pass grounding verification against raw_diff_summary.",
    )
    verification_notes: Optional[str] = Field(
        None,
        description="Optional audit trail or score from the verifier.",
    )