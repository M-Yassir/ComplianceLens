# src/versioning/section_key_rag.py

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from pydantic import BaseModel, Field

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
    terms_analyzed: List[str] = Field(default_factory=list)
    abstained: bool = False


class SectionKeyCrossVersionRAG:
    """Runtime comparative RAG engine operating within src/versioning/.
    
    Handles standard sections via parent maps (§ XXXX) and § 7001 definitions 
    via term-level $O(1)$ glossary maps (glossary_2023/2026.json).
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

        # Standard section maps
        self.parent_map_2023 = self._build_section_parent_map("parents_2023.json")
        self.parent_map_2026 = self._build_section_parent_map("parents_2026.json")

        # Granular term-level glossary maps for § 7001
        self.glossary_2023 = self._build_glossary_map("glossary_2023.json")
        self.glossary_2026 = self._build_glossary_map("glossary_2026.json")

    def _clean_key(self, text: Any) -> str:
        """Normalizes terms and section strings (e.g. 'Automated Decisionmaking Technology' -> 'automated decisionmaking technology')."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", str(text).lower().strip())

    def _clean_section_id(self, sec_str: Any) -> str:
        if not sec_str:
            return ""
        cleaned = re.sub(r"[^\d[a-z]]", "", str(sec_str).lower().strip())
        return cleaned if cleaned not in ("na", "0000", "none") else ""

    def _build_section_parent_map(self, json_filename: str) -> Dict[str, List[Dict[str, Any]]]:
        path = settings.PROCESSED_DATA_DIR / json_filename
        if not path.exists():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        section_map: Dict[str, List[Dict[str, Any]]] = {}
        for parent in data:
            sec_num = self._clean_section_id(parent.get("section"))
            if sec_num:
                section_map.setdefault(sec_num, []).append(parent)

        return section_map

    def _build_glossary_map(self, json_filename: str) -> Dict[str, Dict[str, Any]]:
        """Maps term keys (and aliases) directly to term definition objects."""
        path = settings.PROCESSED_DATA_DIR / json_filename
        if not path.exists():
            print(f"⚠️ Warning: Glossary file not found: {path}")
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"❌ Error parsing {json_filename}: {e}")
            return {}

        glossary_map: Dict[str, Dict[str, Any]] = {}
        for item in data:
            term_key = self._clean_key(item.get("term"))
            if term_key:
                glossary_map[term_key] = item
                # Index aliases if present
                for alias in item.get("aliases", []):
                    alias_key = self._clean_key(alias)
                    if alias_key:
                        glossary_map[alias_key] = item

        return glossary_map

    def _find_matching_terms(self, query: str) -> List[str]:
        """Finds terms from user query that match active defined terms in either glossary."""
        query_clean = self._clean_key(query)
        matched_terms: Set[str] = set()

        all_known_terms = set(self.glossary_2026.keys()).union(set(self.glossary_2023.keys()))

        # Check for direct substring overlap (e.g., query contains 'automated decisionmaking technology')
        for term in all_known_terms:
            if term in query_clean or query_clean in term:
                matched_terms.add(term)

        # Keyword heuristics for common acronyms/terms
        if "admt" in query_clean and "automated decisionmaking technology" in all_known_terms:
            matched_terms.add("automated decisionmaking technology")

        return list(matched_terms)

    def _extract_sections_from_query(self, query: str) -> List[str]:
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
        """Executes targeted comparative analysis across standard sections and/or term definitions."""
        target_sections = extracted_sections or self._extract_sections_from_query(user_query)
        target_terms = self._find_matching_terms(user_query)

        # Vector Discovery if neither explicit sections nor terms were found
        if not target_sections and not target_terms:
            discovered_docs = self.retriever.get_relevant_documents(user_query)
            for doc in discovered_docs:
                sec = self._clean_section_id(doc.metadata.get("section"))
                if sec and sec not in target_sections:
                    target_sections.append(sec)
                    if len(target_sections) >= self.top_k_sections:
                        break

        # Check if Section 7001 is targeted or term definitions were matched
        is_definition_query = "7001" in target_sections or len(target_terms) > 0

        formatted_blocks: List[str] = []
        analyzed_sections: List[str] = []
        analyzed_terms: List[str] = []

        # -------------------------------------------------------------------
        # BRANCH A: Term-Level Definition Alignment (§ 7001 / Glossary)
        # -------------------------------------------------------------------
        if is_definition_query and target_terms:
            for term_key in target_terms:
                term_2023 = self.glossary_2023.get(term_key)
                term_2026 = self.glossary_2026.get(term_key)

                if not term_2023 and not term_2026:
                    continue

                analyzed_terms.append(term_key)
                display_term = (term_2026 or term_2023).get("term", term_key).upper()
                cite_2026 = term_2026.get("section_citation", "§ 7001") if term_2026 else "N/A"
                cite_2023 = term_2023.get("section_citation", "§ 7001") if term_2023 else "N/A"

                header = f"=== DEFINITION: '{display_term}' (2023: {cite_2023} vs 2026: {cite_2026}) ==="

                if term_2023 and term_2026:
                    block = (
                        f"{header}\n"
                        f"--- 2023 DEFINITION ({cite_2023}) ---\n{term_2023['definition_text']}\n\n"
                        f"--- 2026 DEFINITION ({cite_2026}) ---\n{term_2026['definition_text']}\n"
                    )
                elif term_2026 and not term_2023:
                    block = (
                        f"{header} [NEWLY DEFINED TERM IN 2026]\n"
                        f"--- 2023 DEFINITION ---\n[Term '{display_term}' was not defined in 2023 regulations]\n\n"
                        f"--- 2026 DEFINITION ({cite_2026}) ---\n{term_2026['definition_text']}\n"
                    )
                else:
                    block = (
                        f"{header} [DELETED DEFINITION IN 2026]\n"
                        f"--- 2023 DEFINITION ({cite_2023}) ---\n{term_2023['definition_text']}\n\n"
                        f"--- 2026 DEFINITION ---\n[Term '{display_term}' was removed from 2026 regulations]\n"
                    )

                formatted_blocks.append(block)

            if "7001" not in analyzed_sections and analyzed_terms:
                analyzed_sections.append("7001")

        # -------------------------------------------------------------------
        # BRANCH B: Standard Parent Section Alignment (§ 7002, § 7123, etc.)
        # -------------------------------------------------------------------
        for sec in target_sections:
            if sec == "7001":  # Skip whole § 7001 parent dump if we handled terms above
                continue

            parents_2023 = self.parent_map_2023.get(sec, [])
            parents_2026 = self.parent_map_2026.get(sec, [])

            if not parents_2023 and not parents_2026:
                continue

            analyzed_sections.append(sec)
            sec_title = (parents_2026 or parents_2023)[0].get("section_title", "General Provisions")
            header = f"=== SECTION § {sec.upper()}: {sec_title} ==="

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

        # Guardrail Gate: Abstain if no blocks constructed
        if not formatted_blocks:
            return SectionKeyRAGResponse(
                query=user_query,
                answer="Unable to resolve relevant section or term context for cross-version comparison.",
                sections_analyzed=analyzed_sections,
                terms_analyzed=analyzed_terms,
                abstained=True,
            )

        context_str = "\n========================================\n".join(formatted_blocks)

        # Comparative Synthesis Prompt
        prompt = f"""You are an expert regulatory compliance attorney analyzing statutory evolution between the 2023 and 2026 California Consumer Privacy Act (CCPA) regulations.

USER QUESTION:
"{user_query}"

INSTRUCTIONS:
1. Examine the side-by-side comparative context provided below.
2. Direct your response specifically to answering the user's question regarding changes, additions, deletions, reletterings, or definition modifications.
3. Explicitly cite section numbers (e.g., § 7001(a), § 7002) and term names in your answer.
4. If a term or provision has not changed between 2023 and 2026 regarding the user's query topic, state that explicitly.
5. Base your response STRICTLY on the comparative statutory context provided below.

STATUTORY COMPARATIVE CONTEXT:
{context_str}
"""

        response = self.llm.invoke(prompt)
        answer = _get_text_content(response.content).strip()

        return SectionKeyRAGResponse(
            query=user_query,
            answer=answer,
            sections_analyzed=analyzed_sections,
            terms_analyzed=analyzed_terms,
            abstained=False,
        )


if __name__ == "__main__":
    diff_rag = SectionKeyCrossVersionRAG(top_k_sections=2)

    # Test 1: Term definition query
    q1 = "What changed regarding the Requests to Delete?"
    res1 = diff_rag.query_section_changes(q1)
    print("\n" + "=" * 80)
    print(f"Query: {q1}")
    print(f"Sections Analyzed : {res1.sections_analyzed}")
    print(f"Terms Analyzed    : {res1.terms_analyzed}")
    print(f"Answer:\n{res1.answer}\n")