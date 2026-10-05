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
        top_k_candidates: int = 10,
        top_k_reranked: int = 4,
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

        # Step 2: Candidate Retrieval
        candidates = self.hybrid_retriever.get_relevant_documents(
            query=user_query,
            top_k=self.top_k_candidates,
        )

        # Step 3: Cross-Encoder Reranking
        reranked_docs = self.reranker.rerank(
            query=user_query,
            documents=candidates,
            top_k=self.top_k_reranked,
        )

        # Step 4: Guardrail #1 — Abstention / Confidence Evaluation
        abstention = self.abstention_guard.evaluate_retrieval(
            query=user_query,
            retrieved_docs=reranked_docs,
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
    test_query = "What specific cooperation duties does a service provider owe to a business during a cybersecurity audit under Article 9 or a risk assessment under Article 10?"

    print("\n[RAG Pipeline Test — Switchable Query Router]")
    print("=" * 85)
    print(f"Query: '{test_query}'\n")

    # Mode 1: Default Heuristic Router (0 extra API calls)
    print("--- MODE 1: Heuristic Router (Default) ---")
    pipeline_default = RAGPipeline(use_llm_router=False)
    res_default = pipeline_default.query(test_query)

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
    custom_heuristic_router = QueryRouter(llm=None)
    pipeline_injected = RAGPipeline(router=custom_heuristic_router)
    res_injected = pipeline_injected.query(test_query)

    print(f"Route      : {res_injected.routing_decision.route.value}")
    print(f"Reasoning  : {res_injected.routing_decision.reasoning}\n")

    print("--- RETRIEVED SOURCES ---")
    for s in res_injected.source_documents:
        print(f"• Section § {s['section']} ({s['parent_id']}) | Score: {s['rerank_score']}")