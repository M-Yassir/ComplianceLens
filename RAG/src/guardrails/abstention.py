import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from src.config import settings


class AbstentionDecision(BaseModel):
    """Structured decision detailing whether the pipeline should refuse to answer."""

    should_abstain: bool = Field(
        description="True if the query cannot be safely or accurately answered from context."
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Top retrieval relevance score or overall system confidence."
    )
    reason: str = Field(
        description="Explanation for why abstention was or was not triggered."
    )
    fallback_response: Optional[str] = Field(
        default=None,
        description="Standard refusal response to return to the user if should_abstain is True."
    )


class AbstentionGuard:
    """Evaluates retrieval quality and context confidence to trigger safe refusals."""

    DEFAULT_FALLBACK_MESSAGE = (
        "I am unable to answer your query based on the active CCPA/CPRA regulations in the current knowledge base. "
        "The retrieved regulatory texts did not contain sufficient relevant details to provide a reliable answer."
    )

    def __init__(self, min_rerank_score: Optional[float] = None):
        """
        Args:
            min_rerank_score: Minimum required top-reranker score to proceed to generation.
                              Defaults to setting 'ABSTENTION_THRESHOLD' or 0.35.
        """
        self.min_rerank_score = (
            min_rerank_score
            if min_rerank_score is not None
            else getattr(settings, "ABSTENTION_THRESHOLD", 0.35)
        )

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        fallback_override: Optional[str] = None
    ) -> AbstentionDecision:
        """
        Evaluates retrieved documents against relevance and threshold metrics.
        
        Args:
            query: The incoming user query.
            retrieved_docs: List of documents returned by retriever/reranker.
            fallback_override: Custom refusal message if abstention is triggered.
        """
        fallback_text = fallback_override or self.DEFAULT_FALLBACK_MESSAGE

        # Case 1: Zero documents retrieved
        if not retrieved_docs:
            return AbstentionDecision(
                should_abstain=True,
                confidence_score=0.0,
                reason="No relevant regulatory context was found in the database.",
                fallback_response=fallback_text,
            )

        # Extract reranker score from metadata if present, else default to 1.0 (unscored)
        top_doc = retrieved_docs[0]
        top_score = float(top_doc.metadata.get("rerank_score", 1.0))

        # Case 2: Top reranker score falls below required confidence floor
        if top_score < self.min_rerank_score:
            return AbstentionDecision(
                should_abstain=True,
                confidence_score=round(top_score, 4),
                reason=(
                    f"Top retrieval relevance score ({top_score:.4f}) is below the required "
                    f"confidence threshold ({self.min_rerank_score:.2f})."
                ),
                fallback_response=fallback_text,
            )

        # Case 3: Sufficient confidence to proceed
        return AbstentionDecision(
            should_abstain=False,
            confidence_score=round(top_score, 4),
            reason=f"Retrieval confidence ({top_score:.4f}) meets threshold ({self.min_rerank_score:.2f}).",
            fallback_response=None,
        )


if __name__ == "__main__":
    guard = AbstentionGuard(min_rerank_score=0.40)

    print("\n[Abstention Guard Verification]")
    print("=" * 85)

    # 1. Test Empty Context
    res_empty = guard.evaluate_retrieval(
        query="What are the rules for quantum encryption under CCPA?",
        retrieved_docs=[]
    )
    print("\n1. Empty Retrieval Test:")
    print(f"Should Abstain : {res_empty.should_abstain}")
    print(f"Confidence     : {res_empty.confidence_score}")
    print(f"Reason         : {res_empty.reason}")
    print(f"Fallback       : {res_empty.fallback_response}")

    # 2. Test Low Confidence Retrieval
    low_score_doc = Document(
        page_content="Generic privacy policy text without specific rules...",
        metadata={"rerank_score": 0.22, "section": "7001"}
    )
    res_low = guard.evaluate_retrieval(
        query="What are the exact data retention periods?",
        retrieved_docs=[low_score_doc]
    )
    print("\n2. Low Score Retrieval Test (< 0.40):")
    print(f"Should Abstain : {res_low.should_abstain}")
    print(f"Confidence     : {res_low.confidence_score}")
    print(f"Reason         : {res_low.reason}")
    print(f"Fallback       : {res_low.fallback_response}")

    # 3. Test High Confidence Retrieval
    high_score_doc = Document(
        page_content="§ 7013. Notice of Right to Opt-Out... A business shall post...",
        metadata={"rerank_score": 0.88, "section": "7013"}
    )
    res_high = guard.evaluate_retrieval(
        query="What are opt-out link requirements?",
        retrieved_docs=[high_score_doc]
    )
    print("\n3. High Score Retrieval Test (>= 0.40):")
    print(f"Should Abstain : {res_high.should_abstain}")
    print(f"Confidence     : {res_high.confidence_score}")
    print(f"Reason         : {res_high.reason}")
    print(f"Fallback       : {res_high.fallback_response}")