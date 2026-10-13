# src/versioning/section_key_rag.py

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# Suppress Google SDK Automatic Function Calling (AFC) deprecation notice
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import settings
from src.generation.llm_service import get_llm, _get_text_content
from src.retrieval.retriever import ParentChildRetriever


class SectionKeyRAGResponse(BaseModel):
    """Structured output payload for section-key comparative RAG queries."""

    query: str
    answer: str
    sections_analyzed: List[str] = Field(default_factory=list)
    abstained: bool = False


class SectionKeyCrossVersionRAG:
    """Runtime comparative RAG engine operating within src/versioning/.
    
    Uses statutory section citation numbers (§ XXXX) as immutable parent bounds
    for O(1) side-by-side alignment across 2023 and 2026 CCPA parent payloads.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        top_k_sections: int = 3,
        retriever: Optional[ParentChildRetriever] = None,
    ):
        self.llm = llm or get_llm(temperature=0.1)
        self.top_k_sections = top_k_sections
        self.retriever = retriever or ParentChildRetriever(top_k_children=10)

        # In-memory O(1) section map: normalized section_number -> List[ParentChunk]
        self.parent_map_2023 = self._build_section_parent_map("parents_2023.json")
        self.parent_map_2026 = self._build_section_parent_map("parents_2026.json")

    def _clean_section_id(self, sec_str: Any) -> str:
        """Normalizes section strings to pure alphanumeric keys (e.g., '§ 7001.' -> '7001')."""
        if not sec_str:
            return ""
        cleaned = re.sub(r"[^\d[a-z]]", "", str(sec_str).lower().strip())
        return cleaned if cleaned not in ("na", "0000", "none") else ""

    def _extract_part_number(self, parent_id: str) -> int:
        """Extracts part number from parent_id (e.g., 'v2026_sec7002_p2' -> 2) for sequential sorting."""
        match = re.search(r"_p(\d+)$", str(parent_id).lower().strip())
        return int(match.group(1)) if match else 0

    def _build_section_parent_map(self, json_filename: str) -> Dict[str, List[Dict[str, Any]]]:
        """Reads parent JSON files and groups all parent chunk parts by section key."""
        path = settings.PROCESSED_DATA_DIR / json_filename
        if not path.exists():
            print(f"Warning: Parent file not found: {path}")
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error parsing {json_filename}: {e}")
            return {}

        section_map: Dict[str, List[Dict[str, Any]]] = {}
        for parent in data:
            sec_num = self._clean_section_id(parent.get("section"))
            if sec_num:
                if sec_num not in section_map:
                    section_map[sec_num] = []
                section_map[sec_num].append(parent)

        # Sort parent chunks within each section by part number (_p1, _p2, _p3) to preserve text sequence
        for sec_num in section_map:
            section_map[sec_num].sort(key=lambda p: self._extract_part_number(p.get("parent_id", "")))

        return section_map

    def _extract_sections_from_query(self, query: str) -> List[str]:
        """Extracts explicit section numbers (e.g. '7001' from 'section 7001' or '§ 7001')."""
        matches = re.findall(r"(?:§|section)\s*(\d+[a-z]?)", query, re.IGNORECASE)
        extracted: List[str] = []
        for m in matches:
            sec = self._clean_section_id(m)
            if sec and sec not in extracted:
                extracted.append(sec)
        return extracted

    def query_section_changes(
        self, user_query: str, extracted_sections: Optional[List[str]] = None
    ) -> SectionKeyRAGResponse:
        """Executes target discovery, multi-part parent joins, guardrails, and comparative synthesis."""
        target_sections: List[str] = []

        # 1A. Explicit Section Extraction (from Router parameter or directly from query text)
        explicit_sections = extracted_sections or self._extract_sections_from_query(user_query)
        for sec in explicit_sections:
            clean_sec = self._clean_section_id(sec)
            if clean_sec and clean_sec not in target_sections:
                target_sections.append(clean_sec)

        # 1B. Vector Discovery ONLY IF no explicit section citation was found in the query
        if not target_sections:
            discovered_docs = self.retriever.get_relevant_documents(user_query)
            for doc in discovered_docs:
                sec = self._clean_section_id(doc.metadata.get("section"))
                if sec and sec not in target_sections:
                    target_sections.append(sec)
                    if len(target_sections) >= self.top_k_sections:
                        break

        # Guardrail Gate 1: Abstain if no target sections could be resolved
        if not target_sections:
            return SectionKeyRAGResponse(
                query=user_query,
                answer=(
                    "Unable to identify relevant statutory sections in the regulatory database "
                    "to perform a cross-version comparison for this query."
                ),
                sections_analyzed=[],
                abstained=True,
            )

        # 2. Deterministic O(1) side-by-side multi-part parent payload pairing
        formatted_blocks: List[str] = []
        valid_sections_count = 0

        for sec in target_sections:
            parents_2023 = self.parent_map_2023.get(sec, [])
            parents_2026 = self.parent_map_2026.get(sec, [])

            if not parents_2023 and not parents_2026:
                continue

            valid_sections_count += 1
            sec_title = (parents_2026 or parents_2023)[0].get("section_title", "General Provisions")
            header = f"=== SECTION § {sec.upper()}: {sec_title} ==="

            # Reconstitute full unabridged section text across all parent parts
            text_2023 = "\n\n".join(p["text"] for p in parents_2023)
            text_2026 = "\n\n".join(p["text"] for p in parents_2026)

            if parents_2023 and parents_2026:
                block = (
                    f"{header}\n"
                    f"--- 2023 BASELINE TEXT (§ {sec}) ---\n{text_2023}\n\n"
                    f"--- 2026 UPDATED TEXT (§ {sec}) ---\n{text_2026}\n"
                )
            elif parents_2026 and not parents_2023:
                block = (
                    f"{header} [NEWLY ENACTED SECTION IN 2026]\n"
                    f"--- 2023 BASELINE TEXT ---\n[Section § {sec} did not exist in the 2023 regulations]\n\n"
                    f"--- 2026 UPDATED TEXT (§ {sec}) ---\n{text_2026}\n"
                )
            else:
                block = (
                    f"{header} [REPEALED / DELETED SECTION IN 2026]\n"
                    f"--- 2023 BASELINE TEXT (§ {sec}) ---\n{text_2023}\n\n"
                    f"--- 2026 UPDATED TEXT ---\n[Section § {sec} was repealed or deleted in the 2026 regulations]\n"
                )

            formatted_blocks.append(block)

        # Guardrail Gate 2: Abstain if zero parent payloads matched in hash map
        if valid_sections_count == 0:
            return SectionKeyRAGResponse(
                query=user_query,
                answer=f"Could not resolve parent section payloads for target section(s): {', '.join(target_sections)}.",
                sections_analyzed=target_sections,
                abstained=True,
            )

        context_str = "\n========================================\n".join(formatted_blocks)

        # 3. LLM Comparative Synthesis Prompt
        prompt = f"""You are an expert regulatory compliance attorney analyzing statutory evolution between the 2023 and 2026 California Consumer Privacy Act (CCPA) regulations.

USER QUESTION:
"{user_query}"

INSTRUCTIONS:
1. Carefully compare the side-by-side parent section texts provided below for Section(s): {', '.join(target_sections)}.
2. Direct your analysis specifically to answering the user's question regarding changes, additions, deletions, reletterings, or structural updates.
3. Explicitly cite specific section numbers (e.g., § {target_sections[0]}) and subsection identifiers (e.g., (a), (b)(3)) in your analysis.
4. If a provision has not changed between 2023 and 2026 regarding the user's query topic, state that explicitly.
5. Base your response STRICTLY on the comparative statutory context provided below. Do not extrapolate outside this context.

STATUTORY COMPARATIVE CONTEXT:
{context_str}
"""

        response = self.llm.invoke(prompt)
        answer = _get_text_content(response.content).strip()

        return SectionKeyRAGResponse(
            query=user_query,
            answer=answer,
            sections_analyzed=target_sections,
            abstained=False,
        )


if __name__ == "__main__":
    diff_rag = SectionKeyCrossVersionRAG(top_k_sections=2)
    test_query = "What changed regarding restrictions on personal information under section 7002?"

    print(f"\n[Testing src/versioning/section_key_rag.py]")
    print(f"Query: '{test_query}'\n")

    res = diff_rag.query_section_changes(test_query)
    print(f"Sections Analyzed : {res.sections_analyzed}")
    print(f"Abstained         : {res.abstained}\n")
    print("=== SYNTHESIZED COMPARATIVE ANSWER ===")
    print(res.answer)