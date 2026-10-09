import difflib
import uuid
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel

from src.generation.llm_service import get_llm
from src.schemas.versioning import (
    LineageRecord,
    ChangelogCard,
    VerifiedChangelogCard,
    ChangeType,
)
from src.versioning.verifier import CardVerificationGate, VerificationError


def build_citation(chunk: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Constructs a human-readable citation from chunk JSON metadata.
    Handles 'section', 'sub_item', and 'section_title' fields.
    """
    if not chunk:
        return None

    if chunk.get("citation"):
        return chunk["citation"]

    section = chunk.get("section")
    sub_item = chunk.get("sub_item")
    section_title = chunk.get("section_title")

    if section and section != "0000":
        return f"§ {section}{sub_item}" if sub_item else f"§ {section}"
    elif section == "0000":
        return section_title or "Preamble"

    return None


def compute_raw_diff(source_text: Optional[str], target_text: Optional[str]) -> str:
    """Computes a human-readable, deterministic unified diff string between source and target text."""
    src_lines = (source_text or "").splitlines(keepends=True)
    tgt_lines = (target_text or "").splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            src_lines,
            tgt_lines,
            fromfile="Version_N-1",
            tofile="Version_N",
            lineterm="",
        )
    )

    if not diff_lines:
        return "No text changes detected."

    return "\n".join(diff_lines)


class _ChangelogLLMOutput(BaseModel):
    """Internal Pydantic schema for LLM structured output generation."""

    executive_summary: str = Field(
        ...,
        description="1-2 sentence high-level summary of the regulatory change grounded strictly in the diff.",
    )
    substantive_changes: List[str] = Field(
        default_factory=list,
        description="Bullet points of specific new, modified, or removed legal obligations.",
    )
    impact_level: str = Field(
        ...,
        description="Qualitative assessment of legal impact: 'HIGH', 'MEDIUM', or 'LOW'.",
    )


class ChangelogCardGenerator:
    """
    Generates structured ChangelogCards from LineageRecords using deterministic diffing
    and LLM narration via llm_service.get_llm(), gated by CardVerificationGate.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        verifier: Optional[CardVerificationGate] = None,
        max_retries: int = 2,
    ):
        # Uses standard get_llm() factory reading settings from config.py / .env
        self.llm = llm or get_llm(temperature=0.1)
        self.verifier = verifier or CardVerificationGate()
        self.max_retries = max_retries

    def generate_card(
        self,
        record: LineageRecord,
        source_chunk: Optional[Dict[str, Any]] = None,
        target_chunk: Optional[Dict[str, Any]] = None,
    ) -> VerifiedChangelogCard:
        """
        Main entry point to compute raw diffs, invoke LLM card generation, and verify grounding.
        """
        source_text = source_chunk.get("text", "") if source_chunk else ""
        target_text = target_chunk.get("text", "") if target_chunk else ""

        source_citation = record.source_citation or build_citation(source_chunk)
        target_citation = record.target_citation or build_citation(target_chunk)

        # 1. Deterministic Ground Truth Diff
        raw_diff = compute_raw_diff(source_text, target_text)

        # Handle UNCHANGED records directly without calling LLM
        if record.change_type == ChangeType.UNCHANGED:
            card = ChangelogCard(
                card_id=f"card_{uuid.uuid4().hex[:8]}",
                canonical_lineage_id=record.canonical_lineage_id,
                source_version=record.source_version,
                target_version=record.target_version,
                source_citation=source_citation,
                target_citation=target_citation,
                change_type=record.change_type,
                executive_summary="Provision remains unchanged between versions.",
                substantive_changes=[],
                impact_level="LOW",
                raw_diff_summary=raw_diff,
            )
            return self.verifier.verify(card)

        # 2. LLM Generation with Grounding Verification Retry Loop
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                raw_card = self._llm_summarize_diff(
                    record=record,
                    raw_diff=raw_diff,
                    source_text=source_text,
                    target_text=target_text,
                    source_citation=source_citation,
                    target_citation=target_citation,
                )

                # 3. Grounded Verification Gate
                verified_card = self.verifier.verify(raw_card)
                return verified_card

            except VerificationError as ve:
                last_error = ve

        raise RuntimeError(
            f"Failed to generate verified ChangelogCard for lineage {record.canonical_lineage_id} "
            f"after {self.max_retries + 1} attempts. Verification Error: {last_error}"
        )

    def _llm_summarize_diff(
        self,
        record: LineageRecord,
        raw_diff: str,
        source_text: str,
        target_text: str,
        source_citation: Optional[str],
        target_citation: Optional[str],
    ) -> ChangelogCard:
        """Invokes get_llm() with structured schema output to narrate raw diff."""
        structured_llm = self.llm.with_structured_output(_ChangelogLLMOutput)

        prompt = f"""You are a regulatory compliance legal analyst. Summarize the regulatory changes between two versions of a California privacy regulation.

CRITICAL GUARDRAIL: You must ONLY summarize changes explicitly present in the RAW TEXT DIFF provided below. Do NOT infer, invent, or extrapolate legal obligations not contained in the raw text diff.

Metadata:
- Change Type: {record.change_type.value}
- Source Citation: {source_citation or 'N/A'}
- Target Citation: {target_citation or 'N/A'}

RAW TEXT DIFF (Ground Truth):
```diff
{raw_diff}
SOURCE TEXT (N-1):
{source_text[:1500]}

TARGET TEXT (N):
{target_text[:1500]}
"""

        # Generate structured output directly via LangChain model wrapper
        llm_out: _ChangelogLLMOutput = structured_llm.invoke(prompt)

        # Assemble the full ChangelogCard Pydantic object
        return ChangelogCard(
            card_id=f"card_{uuid.uuid4().hex[:8]}",
            canonical_lineage_id=record.canonical_lineage_id,
            source_version=record.source_version,
            target_version=record.target_version,
            source_citation=source_citation,
            target_citation=target_citation,
            change_type=record.change_type,
            executive_summary=llm_out.executive_summary,
            substantive_changes=llm_out.substantive_changes,
            impact_level=llm_out.impact_level,
            raw_diff_summary=raw_diff,
        )