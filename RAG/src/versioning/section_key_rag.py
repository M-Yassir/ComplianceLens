# src/versioning/section_key_rag.py

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import settings
from src.generation.llm_service import get_llm, _get_text_content
from src.retrieval.retriever import ParentChildRetriever

DEFINITIONS_SECTION = "7001"


class SectionKeyRAGResponse(BaseModel):
    """Structured output payload for section-key comparative RAG queries."""

    query: str
    answer: str
    sections_analyzed: List[str] = Field(default_factory=list)
    terms_analyzed: List[str] = Field(default_factory=list)
    abstained: bool = False
    context: str = ""  # the exact side-by-side context the LLM saw (used for citation grounding checks)


class DiffContext(BaseModel):
    """Deterministic, LLM-free result of resolving a query to comparable 2023/2026 text."""

    context_str: str = ""
    sections: List[str] = Field(default_factory=list)
    terms: List[str] = Field(default_factory=list)


class SectionKeyCrossVersionRAG:
    """Runtime comparative RAG engine operating within src/versioning/.

    Handles standard sections via parent maps (§ XXXX) and § 7001 definitions
    via term-level $O(1)$ glossary maps (glossary_2023/2026.json).
    """

    MAX_TERMS = 5              # definition blocks for term-targeted queries
    MAX_OVERVIEW_BLOCKS = 15   # definition blocks when a query targets § 7001 as a whole

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

        # Granular term-level glossary maps for § 7001 (keys include aliases)
        self.glossary_2023 = self._build_glossary_map("glossary_2023.json")
        self.glossary_2026 = self._build_glossary_map("glossary_2026.json")

        # Canonical (one entry per term, no aliases), a union index for query matching,
        # and a chunk-id join from a retrieved 7001 child chunk to its glossary term.
        self._canon_2023 = self._canonical(self.glossary_2023)
        self._canon_2026 = self._canonical(self.glossary_2026)
        self._term_index: Dict[str, Dict[str, Any]] = {}
        for canon_map in (self._canon_2023, self._canon_2026):  # 2026 wins on conflicts
            for key, item in canon_map.items():
                prev = self._term_index.get(key)
                merged = dict(item)
                if prev is not None:
                    merged["is_distinctive"] = bool(prev.get("is_distinctive", True) or item.get("is_distinctive", True))
                self._term_index[key] = merged
        self._by_chunk_2026: Dict[str, Dict[str, Any]] = {
            it["source_chunk_id"]: it for it in self._canon_2026.values() if it.get("source_chunk_id")
        }

    # ------------------------------------------------------------------ utils
    def _clean_key(self, text: Any) -> str:
        """Normalizes terms (e.g. 'Automated Decisionmaking Technology' -> 'automated decisionmaking technology')."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", str(text).lower().strip())

    def _clean_section_id(self, sec_str: Any) -> str:
        """'7051' / '§ 7051(a)' / 'Section 7051' -> '7051'. (The previous regex was a no-op.)"""
        if not sec_str:
            return ""
        s = str(sec_str).lower().strip()
        m = re.search(r"\d+[a-z]?", s)
        cleaned = m.group(0) if m else re.sub(r"[^0-9a-z]", "", s)
        return cleaned if cleaned not in ("na", "0000", "none") else ""

    @staticmethod
    def _norm_def(text: Any) -> str:
        """Definition text with its '(a)' lettering, whitespace, case and quote style removed,
        so a pure re-lettering is not reported as a change."""
        t = str(text or "")
        t = re.sub(r"^\s*\(\s*[a-z0-9]+\s*\)\s*", "", t, flags=re.IGNORECASE)
        t = t.replace("“", '"').replace("”", '"').replace("’", "'")
        return re.sub(r"\s+", " ", t).strip().lower()

    # ------------------------------------------------------------------ loading
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

    def _canonical(self, glossary_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Only the primary-term keys of a glossary map (aliases dropped)."""
        return {k: v for k, v in glossary_map.items() if k == self._clean_key(v.get("term"))}

    # ------------------------------------------------------------------ term matching
    def _find_matching_terms(self, query: str) -> List[str]:
        """Defined terms the query actually talks about (canonical keys, in order of appearance).

        Whole-word matches only (plural tolerated), longest match wins, and terms flagged
        `is_distinctive: false` in the glossary ('agency', 'business', 'consumer'...) count only
        when the query is clearly asking about a definition - otherwise nearly every query would
        match them and be diverted into the definitions branch.
        """
        q = self._clean_key(query)
        definition_intent = bool(re.search(r"\b(defin\w*|meaning|means?)\b", q))

        spans: List[Tuple[int, int, str]] = []
        for key, item in self._term_index.items():
            if not item.get("is_distinctive", True) and not definition_intent:
                continue
            # whole words; each word may be plural ('requests to delete' ~ 'request to delete')
            body = r"\s+".join(re.escape(w) + r"(?:e?s)?" for w in key.split())
            for m in re.finditer(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])", q):
                spans.append((m.start(), m.end(), key))

        if re.search(r"\badmt\b", q) and "automated decisionmaking technology" in self._term_index:
            m = re.search(r"\badmt\b", q)
            spans.append((m.start(), m.end(), "automated decisionmaking technology"))

        kept: List[Tuple[int, int, str]] = []
        for start, end, key in sorted(spans, key=lambda s: -(s[1] - s[0])):  # longest first
            if all(end <= ks or start >= ke for ks, ke, _ in kept):
                kept.append((start, end, key))

        ordered: List[str] = []
        for _, _, key in sorted(kept):
            if key not in ordered:
                ordered.append(key)
        return ordered[: self.MAX_TERMS]

    def _term_for_doc(self, doc: Any) -> Optional[str]:
        """Join a retrieved § 7001 chunk to its glossary term (source_chunk_id, else text containment)."""
        for cid in (doc.metadata.get("parent_id"), doc.metadata.get("chunk_id")):
            item = self._by_chunk_2026.get(cid) if cid else None
            if item:
                return self._clean_key(item.get("term"))
        text = self._norm_def(getattr(doc, "page_content", ""))
        if text and len(text) < 2500:
            for key, item in self._canon_2026.items():
                d = self._norm_def(item.get("definition_text"))
                if d and d in text:
                    return key
        return None

    def _extract_sections_from_query(self, query: str) -> List[str]:
        matches = re.findall(r"(?:§|sections?)\s*(\d+[a-z]?)", query, re.IGNORECASE)
        extracted: List[str] = []
        for m in matches:
            sec = self._clean_section_id(m)
            if sec and sec not in extracted:
                extracted.append(sec)
        return extracted

    # ------------------------------------------------------------------ discovery
    def _discover(self, query: str, docs: Optional[List[Any]] = None) -> Tuple[List[str], List[str]]:
        """Vector discovery (2026 index) -> (section keys, glossary term keys), relevance order."""
        if docs is None:
            docs = self.retriever.get_relevant_documents(query)
        sections: List[str] = []
        terms: List[str] = []
        for doc in docs:
            sec = self._clean_section_id(doc.metadata.get("section"))
            if not sec:
                continue
            if sec == DEFINITIONS_SECTION and (self.glossary_2023 or self.glossary_2026):
                key = self._term_for_doc(doc)
                if key and key not in terms and len(terms) < self.MAX_TERMS:
                    terms.append(key)
                elif not key and DEFINITIONS_SECTION not in sections:
                    sections.append(DEFINITIONS_SECTION)  # unresolved -> whole-§7001 overview
                continue
            if sec not in sections and len(sections) < self.top_k_sections:
                sections.append(sec)
        return sections, terms

    # ------------------------------------------------------------------ block builders
    def _definition_block(self, term_key: str) -> Optional[str]:
        term_2023 = self._canon_2023.get(term_key) or self.glossary_2023.get(term_key)
        term_2026 = self._canon_2026.get(term_key) or self.glossary_2026.get(term_key)
        if not term_2023 and not term_2026:
            return None

        display_term = (term_2026 or term_2023).get("term", term_key).upper()
        cite_2026 = term_2026.get("section_citation", "§ 7001") if term_2026 else "N/A"
        cite_2023 = term_2023.get("section_citation", "§ 7001") if term_2023 else "N/A"
        header = f"=== DEFINITION: '{display_term}' (2023: {cite_2023} vs 2026: {cite_2026}) ==="

        if term_2023 and term_2026:
            same = self._norm_def(term_2023["definition_text"]) == self._norm_def(term_2026["definition_text"])
            tag = " [WORDING UNCHANGED" + (" - CITATION CHANGED]" if cite_2023 != cite_2026 else "]") if same else " [WORDING CHANGED]"
            return (
                f"{header}{tag}\n"
                f"--- 2023 DEFINITION ({cite_2023}) ---\n{term_2023['definition_text']}\n\n"
                f"--- 2026 DEFINITION ({cite_2026}) ---\n{term_2026['definition_text']}\n"
            )
        if term_2026:
            return (
                f"{header} [NEWLY DEFINED TERM IN 2026]\n"
                f"--- 2023 DEFINITION ---\n[Term '{display_term}' was not defined in 2023 regulations]\n\n"
                f"--- 2026 DEFINITION ({cite_2026}) ---\n{term_2026['definition_text']}\n"
            )
        return (
            f"{header} [DELETED DEFINITION IN 2026]\n"
            f"--- 2023 DEFINITION ({cite_2023}) ---\n{term_2023['definition_text']}\n\n"
            f"--- 2026 DEFINITION ---\n[Term '{display_term}' was removed from 2026 regulations]\n"
        )

    def _definitions_overview(self) -> Tuple[str, List[str]]:
        """Deterministic glossary diff for a query about § 7001 as a whole."""
        keys_2023, keys_2026 = set(self._canon_2023), set(self._canon_2026)
        added = sorted(keys_2026 - keys_2023)
        removed = sorted(keys_2023 - keys_2026)
        changed = sorted(
            k for k in keys_2023 & keys_2026
            if self._norm_def(self._canon_2023[k]["definition_text"]) != self._norm_def(self._canon_2026[k]["definition_text"])
        )
        summary = (
            "=== § 7001 DEFINITIONS OVERVIEW (computed deterministically from the glossaries) ===\n"
            f"Newly defined in 2026 ({len(added)}): {', '.join(added) or 'none'}\n"
            f"Removed in 2026 ({len(removed)}): {', '.join(removed) or 'none'}\n"
            f"Wording changed ({len(changed)}): {', '.join(changed) or 'none'}\n"
            f"Unchanged wording: {len(keys_2023 & keys_2026) - len(changed)} terms\n"
        )
        order = added + removed + changed
        shown = order[: self.MAX_OVERVIEW_BLOCKS]
        blocks = [summary]
        blocks += [b for b in (self._definition_block(k) for k in shown) if b]
        if len(order) > len(shown):
            blocks[0] += f"(Full text shown for the first {len(shown)} of {len(order)} changed terms only.)\n"
        return "\n========================================\n".join(blocks), shown

    def _section_block(self, sec: str) -> Optional[str]:
        parents_2023 = self.parent_map_2023.get(sec, [])
        parents_2026 = self.parent_map_2026.get(sec, [])
        if not parents_2023 and not parents_2026:
            return None

        sec_title = (parents_2026 or parents_2023)[0].get("section_title", "General Provisions")
        header = f"=== SECTION § {sec.upper()}: {sec_title} ==="
        text_2023 = "\n\n".join(p["text"] for p in parents_2023)
        text_2026 = "\n\n".join(p["text"] for p in parents_2026)

        if parents_2023 and parents_2026:
            return (
                f"{header}\n"
                f"--- 2023 BASELINE TEXT (§ {sec}) ---\n{text_2023}\n\n"
                f"--- 2026 UPDATED TEXT (§ {sec}) ---\n{text_2026}\n"
            )
        if parents_2026:
            return (
                f"{header} [NEWLY ENACTED SECTION IN 2026]\n"
                f"--- 2023 BASELINE TEXT ---\n[Section § {sec} did not exist in the 2023 regulations]\n\n"
                f"--- 2026 UPDATED TEXT (§ {sec}) ---\n{text_2026}\n"
            )
        return (
            f"{header} [REPEALED / DELETED SECTION IN 2026]\n"
            f"--- 2023 BASELINE TEXT (§ {sec}) ---\n{text_2023}\n\n"
            f"--- 2026 UPDATED TEXT ---\n[Section § {sec} was repealed or deleted in the 2026 regulations]\n"
        )

    # ------------------------------------------------------------------ public API
    def build_context(
        self,
        user_query: str,
        extracted_sections: Optional[List[str]] = None,
        discovered_docs: Optional[List[Any]] = None,
    ) -> DiffContext:
        """Resolves a query to side-by-side 2023/2026 text without calling the LLM.

        `discovered_docs` lets a caller that already ran retrieval (e.g. the policy path of the
        RAG pipeline) reuse its results instead of triggering a second vector search.
        """
        target_sections = [s for s in (self._clean_section_id(x) for x in (extracted_sections or [])) if s]
        if not target_sections:
            target_sections = self._extract_sections_from_query(user_query)
        target_terms = self._find_matching_terms(user_query)

        if not target_sections and not target_terms:
            target_sections, target_terms = self._discover(user_query, discovered_docs)

        have_glossary = bool(self.glossary_2023 or self.glossary_2026)

        # § 7001 targeted, but the caller already retrieved specific definition chunks:
        # narrow the whole-section overview down to exactly those terms.
        if DEFINITIONS_SECTION in target_sections and not target_terms and discovered_docs and have_glossary:
            for d in discovered_docs:
                if self._clean_section_id(d.metadata.get("section")) == DEFINITIONS_SECTION:
                    key = self._term_for_doc(d)
                    if key and key not in target_terms and len(target_terms) < self.MAX_TERMS:
                        target_terms.append(key)
        blocks: List[str] = []
        analyzed_sections: List[str] = []
        analyzed_terms: List[str] = []

        # BRANCH A: term-level definition alignment (§ 7001 / glossary)
        for term_key in target_terms:
            block = self._definition_block(term_key)
            if block:
                blocks.append(block)
                analyzed_terms.append(term_key)
        if analyzed_terms:
            analyzed_sections.append(DEFINITIONS_SECTION)

        # § 7001 targeted as a whole (explicit "§ 7001", or an unresolved definitions hit)
        if DEFINITIONS_SECTION in target_sections and have_glossary and not analyzed_terms:
            overview, shown = self._definitions_overview()
            blocks.append(overview)
            analyzed_terms.extend(shown)
            analyzed_sections.append(DEFINITIONS_SECTION)

        # BRANCH B: standard parent-section alignment
        for sec in target_sections:
            if sec == DEFINITIONS_SECTION and have_glossary:
                continue  # handled above
            block = self._section_block(sec)
            if block:
                blocks.append(block)
                analyzed_sections.append(sec)

        return DiffContext(
            context_str="\n========================================\n".join(blocks),
            sections=analyzed_sections,
            terms=analyzed_terms,
        )

    def query_section_changes(
        self,
        user_query: str,
        extracted_sections: Optional[List[str]] = None,
        discovered_docs: Optional[List[Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> SectionKeyRAGResponse:
        """Executes targeted comparative analysis across standard sections and/or term definitions."""
        ctx = self.build_context(user_query, extracted_sections, discovered_docs)

        # Guardrail Gate: abstain if no blocks constructed
        if not ctx.context_str:
            return SectionKeyRAGResponse(
                query=user_query,
                answer="Unable to resolve relevant section or term context for cross-version comparison.",
                sections_analyzed=ctx.sections,
                terms_analyzed=ctx.terms,
                abstained=True,
            )

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
{ctx.context_str}
"""

        response = self.llm.invoke(prompt, config=llm_config) if llm_config else self.llm.invoke(prompt)
        answer = _get_text_content(response.content).strip()

        return SectionKeyRAGResponse(
            query=user_query,
            answer=answer,
            sections_analyzed=ctx.sections,
            terms_analyzed=ctx.terms,
            abstained=False,
            context=ctx.context_str,
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