import re
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from src.config import settings


class RouteType(str, Enum):
    POLICY_LOOKUP = "POLICY_LOOKUP"
    VERSION_DIFF = "VERSION_DIFF"
    HYBRID_BOTH = "HYBRID_BOTH"


class QueryRoutingDecision(BaseModel):
    """Structured classification output for query routing."""

    route: RouteType = Field(
        description="The target routing category for the user query."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for the routing classification."
    )
    version_filter: Optional[str] = Field(
        default="2026",
        description="Target regulatory version to filter search results (e.g., '2026', '2023', or None for all).",
    )
    extracted_sections: List[str] = Field(
        default_factory=list,
        description="Explicit section numbers extracted from the query (e.g. ['7002', '7025']).",
    )
    reasoning: str = Field(
        description="Brief justification for the assigned route."
    )


class QueryRouter:
    """Semantic and heuristic query router for CCPA compliance workflows."""

    # Keywords that strongly indicate cross-version diff intent
    DIFF_KEYWORDS = {
        "change", "changed", "changes", "diff", "difference", "different",
        "amend", "amended", "amendment", "update", "updated", "updates",
        "versus", "vs", "compare", "compared", "comparison", "new rule",
        "previous", "prior", "2023", "historical", "redline", "strikethrough"
    }

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def _extract_sections(self, query: str) -> List[str]:
        """Extracts section references (e.g. '7002' from '§ 7002' or 'Section 7002')."""
        matches = re.findall(r"(?:§|section)\s*(\d+[a-z]?)", query, re.IGNORECASE)
        return sorted(list(set(m.lower() for m in matches)))

    def _heuristic_route(self, query: str) -> QueryRoutingDecision:
        """Lightweight, deterministic rule-based router as primary/fallback classifier."""
        query_lower = query.lower()
        extracted_sections = self._extract_sections(query)
        words = set(re.findall(r"\b\w+\b", query_lower))

        diff_matches = words.intersection(self.DIFF_KEYWORDS)

        # Explicit version comparison query
        if len(diff_matches) >= 2 or ("2023" in words and "2026" in words) or "what changed" in query_lower:
            return QueryRoutingDecision(
                route=RouteType.VERSION_DIFF,
                confidence=0.95,
                version_filter=None,  # Need cross-version scope
                extracted_sections=extracted_sections,
                reasoning=f"Query matched cross-version keywords: {list(diff_matches)}",
            )

        # Single diff keyword present
        if len(diff_matches) == 1:
            return QueryRoutingDecision(
                route=RouteType.HYBRID_BOTH,
                confidence=0.80,
                version_filter=None,
                extracted_sections=extracted_sections,
                reasoning=f"Query contains ambiguous update keyword '{list(diff_matches)[0]}'. Routing to hybrid policy + diff lookup.",
            )

        # Standard current policy lookup
        return QueryRoutingDecision(
            route=RouteType.POLICY_LOOKUP,
            confidence=0.90,
            version_filter="2026",
            extracted_sections=extracted_sections,
            reasoning="Standard regulatory policy lookup targeting active 2026 regulations.",
        )

    def route_query(self, query: str) -> QueryRoutingDecision:
        """Classifies incoming query to determine retrieval behavior."""
        if not self.llm:
            return self._heuristic_route(query)

        # LLM-assisted classification with fallback
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are a regulatory query classification router for California CCPA/CPRA law.\n"
                    "Classify the user query into one of three routes:\n"
                    "- POLICY_LOOKUP: User is asking what the active rules or requirements currently are.\n"
                    "- VERSION_DIFF: User is asking what changed, what was amended, or comparing 2023 vs 2026 versions.\n"
                    "- HYBRID_BOTH: User query requires both current rules and changelog context.\n"
                )),
                ("human", "{query}")
            ])
            
            structured_llm = self.llm.with_structured_output(QueryRoutingDecision)
            chain = prompt | structured_llm
            return chain.invoke({"query": query})
        except Exception as e:
            print(f"⚠️ LLM routing failed ({e}). Falling back to heuristic classifier.")
            return self._heuristic_route(query)


if __name__ == "__main__":
    router = QueryRouter()

    test_queries = [
        "What are the requirements for consumer opt-out links under § 7013?",
        "What changed in section 7025 between the 2023 regulations and the 2026 updates?",
        "Did the CPPA update the rules for Automated Decisionmaking Technology (ADMT)?",
        "How should a business display Do Not Sell or Share links?",
    ]

    print("\n[Query Router Test]")
    print("=" * 85)

    for q in test_queries:
        decision = router.route_query(q)
        print(f"\nQuery          : '{q}'")
        print(f"Route          : {decision.route.value}")
        print(f"Confidence     : {decision.confidence}")
        print(f"Version Filter : {decision.version_filter}")
        print(f"Sections       : {decision.extracted_sections}")
        print(f"Reasoning      : {decision.reasoning}")