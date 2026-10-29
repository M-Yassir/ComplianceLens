from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


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