import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import re
from typing import List, Set, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class CitationVerificationResult(BaseModel):
    """Structured report detailing citation grounding accuracy."""

    is_valid: bool = Field(
        description="True if all generated citations exist in the retrieved context."
    )
    verification_score: float = Field(
        ge=0.0, le=1.0,
        description="Ratio of valid citations to total cited sections (1.0 if no citations generated)."
    )
    generated_citations: List[str] = Field(
        default_factory=list,
        description="Distinct section citations extracted from LLM response."
    )
    valid_citations: List[str] = Field(
        default_factory=list,
        description="Citations confirmed present in retrieved context."
    )
    hallucinated_citations: List[str] = Field(
        default_factory=list,
        description="Citations fabricated by LLM not present in retrieved context."
    )
    summary: str = Field(
        description="Human-readable summary of the citation audit."
    )


class CitationVerifier:
    """Verifies that generated answer citations are strictly grounded in retrieved documents."""

    # Regex matching common legal section citations (e.g. § 7013, Section 7025, Sec. 7002(a))
    CITATION_REGEX = re.compile(
        r"(?:§|section|sec\.)\s*(\d{4}[a-z]?)",
        re.IGNORECASE
    )

    def extract_citations(self, text: str) -> List[str]:
        """Extracts unique section numbers referenced in text."""
        matches = self.CITATION_REGEX.findall(text)
        # Normalize to lowercase and deduplicate preserving order
        extracted: List[str] = []
        for m in matches:
            sec = m.lower().strip()
            if sec not in extracted:
                extracted.append(sec)
        return extracted

    def get_context_sections(self, context_documents: List[Document]) -> Set[str]:
        """Collects all valid section numbers available across retrieved parent documents."""
        sections: Set[str] = set()

        for doc in context_documents:
            # 1. Metadata section field
            meta_sec = str(doc.metadata.get("section", "")).lower().strip()
            if meta_sec and meta_sec != "n/a":
                # Clean section string (e.g. "§ 7013" -> "7013")
                clean_sec = re.sub(r"[^\d[a-z]]", "", meta_sec)
                if clean_sec:
                    sections.add(clean_sec)

            # 2. Search body text and title for embedded section headers
            text_to_search = f"{doc.metadata.get('section_title', '')} {doc.page_content}"
            found_in_text = self.CITATION_REGEX.findall(text_to_search)
            for s in found_in_text:
                sections.add(s.lower().strip())

        return sections

    def verify(
        self,
        generated_answer: str,
        context_documents: List[Document]
    ) -> CitationVerificationResult:
        """Audits generated response citations against retrieved context."""
        generated_citations = self.extract_citations(generated_answer)
        available_sections = self.get_context_sections(context_documents)

        # If LLM cited no specific sections, response passes citation check by default
        if not generated_citations:
            return CitationVerificationResult(
                is_valid=True,
                verification_score=1.0,
                generated_citations=[],
                valid_citations=[],
                hallucinated_citations=[],
                summary="No section citations found in response. Verification passed."
            )

        valid: List[str] = []
        hallucinated: List[str] = []

        for citation in generated_citations:
            if citation in available_sections:
                valid.append(citation)
            else:
                hallucinated.append(citation)

        total_cited = len(generated_citations)
        valid_count = len(valid)
        score = round(valid_count / total_cited, 2) if total_cited > 0 else 1.0
        is_valid = len(hallucinated) == 0

        if is_valid:
            summary = f"PASSED: All {total_cited} cited section(s) ({', '.join(valid)}) exist in context."
        else:
            summary = (
                f"FAILED: Found {len(hallucinated)} hallucinated citation(s) ({', '.join(hallucinated)}). "
                f"Valid context sections were: {sorted(list(available_sections))}."
            )

        return CitationVerificationResult(
            is_valid=is_valid,
            verification_score=score,
            generated_citations=generated_citations,
            valid_citations=valid,
            hallucinated_citations=hallucinated,
            summary=summary
        )


if __name__ == "__main__":
    verifier = CitationVerifier()

    # Mock retrieved context documents (§ 7013 and § 7004)
    sample_context = [
        Document(
            page_content="§ 7013. Notice of Right to Opt-out... A business shall provide notice...",
            metadata={"parent_id": "v2026_sec7013_p1", "section": "7013", "section_title": "Notice of Right to Opt-out"}
        ),
        Document(
            page_content="§ 7004. Requirements for Methods... Businesses shall design methods...",
            metadata={"parent_id": "v2026_sec7004_p1", "section": "7004", "section_title": "Requirements for Methods"}
        )
    ]

    print("\n[Citation Verifier Test]")
    print("=" * 85)

    # Test Case 1: Fully Grounded Answer
    grounded_answer = (
        "Under § 7013, businesses must provide clear notice of the right to opt-out. "
        "Furthermore, Section 7004 dictates how methods for submitting requests must be designed."
    )
    res1 = verifier.verify(grounded_answer, sample_context)
    print("\n1. Testing Grounded Answer:")
    print(f"Is Valid     : {res1.is_valid}")
    print(f"Score        : {res1.verification_score}")
    print(f"Valid Cites  : {res1.valid_citations}")
    print(f"Hallucinated : {res1.hallucinated_citations}")
    print(f"Summary      : {res1.summary}")

    # Test Case 2: Answer with Hallucinated Citation (§ 9999)
    hallucinated_answer = (
        "According to § 7013, notices must be accessible. "
        "However, § 9999 specifies penalties for non-compliance."
    )
    res2 = verifier.verify(hallucinated_answer, sample_context)
    print("\n2. Testing Hallucinated Answer:")
    print(f"Is Valid     : {res2.is_valid}")
    print(f"Score        : {res2.verification_score}")
    print(f"Valid Cites  : {res2.valid_citations}")
    print(f"Hallucinated : {res2.hallucinated_citations}")
    print(f"Summary      : {res2.summary}")