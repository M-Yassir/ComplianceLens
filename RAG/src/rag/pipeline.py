import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from src.generation.llm_service import get_llm, _get_text_content
from src.generation.prompt_templates import get_prompt_template
from src.retrieval.router import QueryRouter, QueryRoutingDecision, RouteType
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.guardrails.abstention import AbstentionGuard, AbstentionDecision
from src.guardrails.citation_verifier import CitationVerifier, CitationVerificationResult


class RAGResponse(BaseModel):
    """End-to-end structured response payload from the RAG pipeline."""

    query: str
    answer: str
    routing_decision: QueryRoutingDecision
    abstention_decision: AbstentionDecision
    citation_verification: CitationVerificationResult
    source_documents: List[Dict[str, Any]] = Field(default_factory=list)


class RAGPipeline:
    """End-to-end production RAG orchestrator for CCPA compliance workflows."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        router: Optional[QueryRouter] = None,  # Injectable router instance
        use_llm_router: bool = False,         # Flag if creating router dynamically
        min_rerank_score: float = 0.35,
        top_k_candidates: int = 20,
        top_k_reranked: int = 8,
        max_per_section: int = 2,
    ):
        """
        Args:
            llm: Language model instance for generation.
            router: Explicit QueryRouter instance. If provided, overrides use_llm_router.
            use_llm_router: If True and router is None, uses LLM-backed QueryRouter.
                            If False and router is None, defaults to zero-API Heuristic QueryRouter.
            min_rerank_score: Confidence floor for abstention guardrail.
            top_k_candidates: Number of initial hybrid search results.
            top_k_reranked: Number of top cross-encoder reranked results.
        """
        self.llm = llm or get_llm()

        # Configurable router injection
        if router is not None:
            self.router = router
        else:
            router_llm = self.llm if use_llm_router else None
            self.router = QueryRouter(llm=router_llm)

        self.hybrid_retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.abstention_guard = AbstentionGuard(min_rerank_score=min_rerank_score)
        self.citation_verifier = CitationVerifier()

        self.top_k_candidates = top_k_candidates
        self.top_k_reranked = top_k_reranked
        self.max_per_section = max_per_section

    def _format_context(self, docs: List[Document]) -> str:
        """Formats retrieved documents into a clean text block for prompt injection."""
        formatted_blocks = []
        for idx, doc in enumerate(docs, 1):
            sec = doc.metadata.get("section", "N/A")
            title = doc.metadata.get("section_title", "")
            parent_id = doc.metadata.get("parent_id", f"doc_{idx}")

            block = (
                f"[Source {idx} | Section § {sec} | {title} | Ref: {parent_id}]\n"
                f"{doc.page_content}\n"
            )
            formatted_blocks.append(block)

        return "\n----------------------------------------\n".join(formatted_blocks)

    def query(self, user_query: str) -> RAGResponse:
        """Executes the full RAG workflow for an incoming compliance query."""

        # Step 1: Intent Routing
        routing_decision = self.router.route_query(user_query)

        # Check section presence early so both retrieval and abstention can use it
        has_sections = len(routing_decision.extracted_sections) > 0
        effective_threshold = 0.35 if has_sections else 0.22

        # Step 2: Candidate Retrieval
        top_k = self.top_k_candidates if has_sections else 25
        candidates = self.hybrid_retriever.get_relevant_documents(
            query=user_query,
            top_k=top_k,
        )

        # Step 3: Cross-Encoder Reranking
        reranked_docs = self.reranker.rerank_with_diversity(
            query=user_query,
            documents=candidates,
            top_k=self.top_k_reranked,
            max_per_section=self.max_per_section,
            fetch_k=15,
        )

        # Step 4: Guardrail #1 — Abstention / Confidence Evaluation
        abstention = self.abstention_guard.evaluate_retrieval(
            query=user_query,
            retrieved_docs=reranked_docs,
            threshold=effective_threshold
        )

        if abstention.should_abstain:
            empty_citation_check = CitationVerificationResult(
                is_valid=True,
                verification_score=1.0,
                summary="Query resulted in abstention. Citation verification skipped."
            )
            return RAGResponse(
                query=user_query,
                answer=abstention.fallback_response or "Unable to provide a confident answer.",
                routing_decision=routing_decision,
                abstention_decision=abstention,
                citation_verification=empty_citation_check,
                source_documents=[],
            )

        # Step 5: Prompt Selection & LLM Answer Generation
        prompt_template = get_prompt_template(routing_decision.route)
        context_str = self._format_context(reranked_docs)

        formatted_messages = prompt_template.format_messages(
            context=context_str,
            query=user_query,
        )

        llm_response = self.llm.invoke(formatted_messages)
        generated_answer = _get_text_content(llm_response.content).strip()

        # Step 6: Guardrail #2 — Citation Verification Audit
        citation_check = self.citation_verifier.verify(
            generated_answer=generated_answer,
            context_documents=reranked_docs,
        )

        # Step 7: Serialize Source Metadata
        sources = [
            {
                "parent_id": doc.metadata.get("parent_id"),
                "section": doc.metadata.get("section"),
                "section_title": doc.metadata.get("section_title"),
                "rerank_score": round(doc.metadata.get("rerank_score", 0.0), 4),
                "snippet": doc.page_content[:180] + "...",
            }
            for doc in reranked_docs
        ]

        return RAGResponse(
            query=user_query,
            answer=generated_answer,
            routing_decision=routing_decision,
            abstention_decision=abstention,
            citation_verification=citation_check,
            source_documents=sources,
        )


if __name__ == "__main__":
    test_query_1 = "Can an email marketing service provider analyze customer engagement with a company's emails to refine its delivery algorithms, and then apply those algorithmic improvements when serving another company?"
    test_query_2 = "If a social media company accepts a list of customer email addresses from a business to identify and target those specific users with ads on its platform, does the social media company qualify as a service provider under § 7050?"
    test_query_3 = "What must a contractor do if a consumer submits a CCPA deletion request directly to the contractor rather than to the business?"
    test_query_4 = "What are the regulatory implications for a business if it transfers personal information to a service vendor without a contract that complies with section 7051(a)?"
    test_query_5 = "What specific cooperation duties does a service provider owe to a business during a cybersecurity audit under Article 9 or a risk assessment under Article 10?"
    test_query_6 = "Within how many calendar days must a service provider forward a direct consumer deletion request to the primary business under § 7050?"
    print("\n[RAG Pipeline Test — Switchable Query Router]")
    print("=" * 85)
    print(f"Query: '{test_query_3}'\n")

    # Mode 1: Default Heuristic Router (0 extra API calls)
    print("--- MODE 1: Heuristic Router (Default) ---")
    pipeline_default = RAGPipeline(use_llm_router=False)
    res_default = pipeline_default.query(test_query_3)

    print(f"Route      : {res_default.routing_decision.route.value}")
    print(f"Reasoning  : {res_default.routing_decision.reasoning}")
    print(f"Abstained  : {res_default.abstention_decision.should_abstain}")
    print(f"Valid Cites: {res_default.citation_verification.is_valid}")
    print(f"Answer     :\n{res_default.answer}\n")

    print("--- RETRIEVED SOURCES ---")
    for s in res_default.source_documents:
        print(f"• Section § {s['section']} ({s['parent_id']}) | Score: {s['rerank_score']}")

    # Mode 2: Explicit Router Injection (Passing custom router instance)
    print("--- MODE 2: Custom Injected Router ---")
    llm = get_llm()
    custom_heuristic_router = QueryRouter(llm)
    pipeline_injected = RAGPipeline(router=custom_heuristic_router)
    res_injected = pipeline_injected.query(test_query_3)

    print(f"Route      : {res_injected.routing_decision.route.value}")
    print(f"Reasoning  : {res_injected.routing_decision.reasoning}\n")

    print("--- RETRIEVED SOURCES ---")
    for s in res_injected.source_documents:
        print(f"• Section § {s['section']} ({s['parent_id']}) | Score: {s['rerank_score']}")