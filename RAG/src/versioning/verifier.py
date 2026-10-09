import re
from typing import List, Tuple, Set

from src.schemas.versioning import ChangelogCard, VerifiedChangelogCard


class VerificationError(Exception):
    """Raised when a ChangelogCard fails grounding verification against its raw diff."""
    pass


# Modal verbs that carry legal obligation strength. These must NOT be treated
# as stopwords: "businesses shall conduct audits" and "businesses may conduct
# audits" share every other content word, but mean opposite things. Stripping
# these before scoring (as the previous version did) made an obligation-
# direction flip invisible to the grounding check.
MANDATORY_MODALS = {"shall", "must"}
PERMISSIVE_MODALS = {"may"}


class CardVerificationGate:
    """
    Verifies that claims in LLM-generated ChangelogCards (executive summary and
    substantive changes) are strictly grounded in the deterministic raw_diff_summary.
    """

    def __init__(self, min_token_overlap_ratio: float = 0.60):
        self.min_token_overlap_ratio = min_token_overlap_ratio

    def verify(self, card: ChangelogCard) -> VerifiedChangelogCard:
        """
        Validates a ChangelogCard against card.raw_diff_summary.
        Returns a VerifiedChangelogCard with is_verified=True if grounded.
        Raises VerificationError if ungrounded claims are detected.
        """
        # For UNCHANGED, no substantive claims should exist
        if card.change_type.value == "unchanged":
            return VerifiedChangelogCard(
                **card.model_dump(),
                is_verified=True,
                verification_notes="Pass: Unchanged provision requires no diff verification.",
            )

        raw_diff = card.raw_diff_summary
        if not raw_diff or not raw_diff.strip():
            raise VerificationError(
                f"Card {card.card_id} for lineage {card.canonical_lineage_id} "
                "has empty raw_diff_summary."
            )

        claims_to_check: List[Tuple[str, str]] = [
            ("executive_summary", card.executive_summary)
        ] + [
            (f"substantive_change_{idx}", bullet)
            for idx, bullet in enumerate(card.substantive_changes)
        ]

        unsupported_claims: List[str] = []
        scores: List[float] = []

        for claim_label, claim_text in claims_to_check:
            is_grounded, overlap_score = self._verify_claim_grounding(
                claim=claim_text, diff_context=raw_diff
            )
            scores.append(overlap_score)

            if not is_grounded:
                unsupported_claims.append(
                    f"[{claim_label}] (overlap: {overlap_score:.2f}) -> '{claim_text}'"
                )

        if unsupported_claims:
            failure_details = "\n".join(unsupported_claims)
            raise VerificationError(
                f"Card verification failed for lineage {card.canonical_lineage_id} "
                f"(Card ID: {card.card_id}). Unsupported claims:\n{failure_details}"
            )

        avg_score = sum(scores) / len(scores) if scores else 1.0

        return VerifiedChangelogCard(
            **card.model_dump(),
            is_verified=True,
            verification_notes=(
                f"Pass: Grounding verified against raw diff context "
                f"(mean overlap score: {avg_score:.2f})."
            ),
        )

    def _verify_claim_grounding(
        self, claim: str, diff_context: str
    ) -> Tuple[bool, float]:
        """
        Evaluates token, modal-verb, and entity grounding of a single claim
        against the raw diff text. Returns (is_grounded, overlap_score).
        """
        # Normalize and tokenize, ignoring common legal/grammatical stopwords.
        # Modal verbs are deliberately NOT included here (see MANDATORY_MODALS /
        # PERMISSIVE_MODALS above) so they contribute to the overlap score and
        # are available for the dedicated obligation-direction check below.
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "over", "after",
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "this",
            "that", "these", "those", "section", "subsection", "paragraph", "clause"
        }

        claim_tokens: Set[str] = {
            w.lower()
            for w in re.findall(r"\b\w{3,}\b", claim)
            if w.lower() not in stopwords
        }

        diff_tokens: Set[str] = {
            w.lower()
            for w in re.findall(r"\b\w{3,}\b", diff_context)
            if w.lower() not in stopwords
        }

        if not claim_tokens:
            return True, 1.0

        # Calculate directional overlap: how much of the claim's content comes from the diff
        intersection = claim_tokens.intersection(diff_tokens)
        overlap_score = len(intersection) / len(claim_tokens)

        # Obligation-direction check: a claim that asserts a different
        # modal strength than the diff supports is a hallucination risk even
        # when every surrounding content word overlaps. This is checked as
        # an explicit hard-fail rather than folded into the general token
        # overlap ratio, because one flipped word ("must" -> "may") can
        # completely invert the legal meaning of an otherwise well-grounded
        # sentence.
        claim_lower = claim.lower()
        diff_lower = diff_context.lower()

        claim_is_mandatory = bool(re.search(r"\b(shall|must)\b", claim_lower))
        claim_is_permissive = bool(re.search(r"\bmay\b", claim_lower))
        diff_is_mandatory = bool(re.search(r"\b(shall|must)\b", diff_lower))
        diff_is_permissive = bool(re.search(r"\bmay\b", diff_lower))

        if claim_is_mandatory and not diff_is_mandatory:
            # Claim asserts an obligation the diff never states as mandatory.
            return False, overlap_score

        if claim_is_permissive and not claim_is_mandatory and diff_is_mandatory and not diff_is_permissive:
            # Diff only supports a mandatory reading, but the claim frames
            # the same provision as merely permissive.
            return False, overlap_score

        # Check key numeric/citation/entity matches (e.g., "15 days", "7002(b)")
        claim_entities = set(re.findall(r"\b\d+\b|§\s*\d+", claim))
        diff_entities = set(re.findall(r"\b\d+\b|§\s*\d+", diff_context))

        unsupported_entities = claim_entities - diff_entities
        if unsupported_entities:
            # Explicit failure if LLM invented numbers/section references not in raw diff.
            #
            # Known limitation (not fixed by this check): this only verifies
            # that each number APPEARS in the diff text, not that it's
            # correctly PAIRED with the right role. A claim saying "the
            # deadline increased from 15 to 45 days" would still pass
            # against a diff that actually shows a decrease from 45 to 15,
            # because both numbers are present in diff_entities regardless
            # of which direction the real change went. Catching that would
            # need structured (value, unit, direction) extraction rather
            # than set membership - treat this as a documented gap, not a
            # guarantee that directional numeric claims are verified.
            return False, overlap_score

        is_grounded = overlap_score >= self.min_token_overlap_ratio
        return is_grounded, overlap_score