import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from typing import List, Optional
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
    score_margin: Optional[float] = Field(
        default=None,
        description="top_score / second_score. None when only one document was retrieved."
    )


def _doc_score(doc: Document) -> float:
    return float(doc.metadata.get("rerank_score", 1.0))


class AbstentionGuard:
    """Evaluates retrieval quality and context confidence to trigger safe refusals.

    Why this isn't a single absolute threshold: the cross-encoder's raw
    scores are not calibrated probabilities. On dense legal text they run
    compressed (a clearly correct top chunk can score ~0.06 while its
    irrelevant neighbours score ~0.007), so any one cutoff either rejects
    correct answers or lets noise through. What IS stable is the shape of
    the score distribution: a relevant query produces one chunk that
    clearly separates from the rest; an out-of-scope query produces a low,
    flat pile. So the decision uses three signals:

      1. top_score < abs_floor            -> nothing relevant at all: abstain.
      2. top_score >= min_rerank_score    -> trusted outright: answer.
      3. in between                       -> answer only if the top chunk
                                             clearly separates from the
                                             runner-up (top/second >= min_margin).

    `select_context()` is the companion for the other half of the
    trade-off: which chunks reach the LLM. It keeps a small minimum plus
    anything within `relative_keep_ratio` of the top score, so the cutoff
    scales with the query instead of depending on the absolute score scale.

    The default values below are starting points, not calibrated numbers -
    tune them with eval/calibrate_abstention.py against your golden set.
    """

    DEFAULT_FALLBACK_MESSAGE = (
        "I am unable to answer your query based on the active CCPA/CPRA regulations in the current knowledge base. "
        "The retrieved regulatory texts did not contain sufficient relevant details to provide a reliable answer."
    )

    def __init__(
        self,
        min_rerank_score: Optional[float] = None,
        abs_floor: Optional[float] = None,
        min_margin: Optional[float] = None,
        relative_keep_ratio: Optional[float] = None,
        min_context_docs: Optional[int] = None,
    ):
        """
        Args:
            min_rerank_score: Top score at/above which retrieval is trusted outright
                              (no margin check). Defaults to setting
                              'ABSTENTION_THRESHOLD' or 0.35.
            abs_floor: Top score below which we always abstain. Keep this well
                       below the lowest top-1 you see on answerable queries.
            min_margin: Required top/second score ratio when the top score sits
                        between abs_floor and min_rerank_score.
            relative_keep_ratio: select_context() keeps docs scoring at least
                                 this fraction of the top score.
            min_context_docs: select_context() always keeps at least this many
                              top docs (a safety net for multi-section answers).
        """
        self.min_rerank_score = (
            min_rerank_score
            if min_rerank_score is not None
            else getattr(settings, "ABSTENTION_THRESHOLD", 0.35)
        )
        self.abs_floor = (
            abs_floor if abs_floor is not None else getattr(settings, "ABSTENTION_FLOOR", 0.01)
        )
        self.min_margin = (
            min_margin if min_margin is not None else getattr(settings, "ABSTENTION_MIN_MARGIN", 1.5)
        )
        self.relative_keep_ratio = (
            relative_keep_ratio
            if relative_keep_ratio is not None
            else getattr(settings, "CONTEXT_RELATIVE_KEEP_RATIO", 0.2)
        )
        self.min_context_docs = (
            min_context_docs
            if min_context_docs is not None
            else getattr(settings, "CONTEXT_MIN_DOCS", 2)
        )

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        fallback_override: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> AbstentionDecision:
        """
        Evaluates retrieved documents against relevance, floor and margin metrics.

        Args:
            query: The incoming user query.
            retrieved_docs: List of documents returned by retriever/reranker.
            fallback_override: Custom refusal message if abstention is triggered.
            threshold: Per-call override of `min_rerank_score` (the "trusted
                       outright" level). Kept for backward compatibility with
                       callers that already pass one.
        """
        fallback_text = fallback_override or self.DEFAULT_FALLBACK_MESSAGE
        trusted_level = threshold if threshold is not None else self.min_rerank_score

        # Case 1: Zero documents retrieved
        if not retrieved_docs:
            return AbstentionDecision(
                should_abstain=True,
                confidence_score=0.0,
                reason="No relevant regulatory context was found in the database.",
                fallback_response=fallback_text,
            )

        scores = sorted((_doc_score(d) for d in retrieved_docs), reverse=True)
        top_score = scores[0]
        second_score = scores[1] if len(scores) > 1 else None
        margin: Optional[float] = None
        if second_score is not None:
            margin = top_score / max(second_score, 1e-9)
        margin_out = round(margin, 3) if margin is not None else None

        # Case 2: Nothing relevant at all
        if top_score < self.abs_floor:
            return AbstentionDecision(
                should_abstain=True,
                confidence_score=round(top_score, 4),
                score_margin=margin_out,
                reason=(
                    f"Top retrieval relevance score ({top_score:.4f}) is below the absolute "
                    f"floor ({self.abs_floor:.4f})."
                ),
                fallback_response=fallback_text,
            )

        # Case 3: Trusted outright
        if top_score >= trusted_level:
            return AbstentionDecision(
                should_abstain=False,
                confidence_score=round(top_score, 4),
                score_margin=margin_out,
                reason=f"Retrieval confidence ({top_score:.4f}) meets trusted level ({trusted_level:.2f}).",
                fallback_response=None,
            )

        # Case 4: Low-but-nonzero score - answer only if there is a clear winner
        if margin is None or margin >= self.min_margin:
            margin_txt = "single result" if margin is None else f"{margin:.1f}x the runner-up"
            return AbstentionDecision(
                should_abstain=False,
                confidence_score=round(top_score, 4),
                score_margin=margin_out,
                reason=(
                    f"Top score ({top_score:.4f}) is below the trusted level ({trusted_level:.2f}) "
                    f"but clearly separates from the rest ({margin_txt})."
                ),
                fallback_response=None,
            )

        return AbstentionDecision(
            should_abstain=True,
            confidence_score=round(top_score, 4),
            score_margin=margin_out,
            reason=(
                f"Top score ({top_score:.4f}) is below the trusted level ({trusted_level:.2f}) "
                f"and no result clearly stands out (margin {margin:.2f}x < {self.min_margin:.2f}x)."
            ),
            fallback_response=fallback_text,
        )

    def select_context(self, retrieved_docs: List[Document]) -> List[Document]:
        """Chooses which retrieved docs are passed to the LLM.

        Keeps the top `min_context_docs`, plus any further doc scoring at
        least `relative_keep_ratio` x the top score. Relative to the top
        score, so it doesn't depend on the (uncalibrated) absolute scale.
        """
        if not retrieved_docs:
            return []

        ranked = sorted(retrieved_docs, key=_doc_score, reverse=True)
        top_score = _doc_score(ranked[0])
        cutoff = top_score * self.relative_keep_ratio

        selected: List[Document] = []
        for idx, doc in enumerate(ranked):
            if idx < self.min_context_docs or _doc_score(doc) >= cutoff:
                selected.append(doc)
        return selected


if __name__ == "__main__":
    guard = AbstentionGuard()

    print("\n[Abstention Guard Verification]")
    print("=" * 85)

    def mk(*scores: float) -> List[Document]:
        return [
            Document(page_content=f"doc {i}", metadata={"rerank_score": s, "section": str(7000 + i)})
            for i, s in enumerate(scores)
        ]

    cases = {
        "empty": [],
        "clear winner, low absolute (0.0637 vs 0.0088...)": mk(0.0637, 0.0088, 0.0071, 0.0066, 0.0064),
        "low and flat": mk(0.0120, 0.0110, 0.0105, 0.0100),
        "below floor": mk(0.004, 0.003),
        "high score": mk(0.88, 0.40),
    }
    for label, docs in cases.items():
        d = guard.evaluate_retrieval("q", docs)
        kept = [x.metadata["section"] for x in guard.select_context(docs)]
        print(f"\n{label}\n  abstain={d.should_abstain} margin={d.score_margin} kept={kept}\n  {d.reason}")