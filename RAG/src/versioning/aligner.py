import difflib
import uuid
import re
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

from src.schemas.versioning import LineageRecord, ChangeType


def calculate_jaccard_overlap(text1: str, text2: str) -> float:
    """Calculates word-level Jaccard similarity score between two texts."""
    words1: Set[str] = set(re.findall(r"\w+", text1.lower()))
    words2: Set[str] = set(re.findall(r"\w+", text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    return float(intersection / union) if union > 0 else 0.0


def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity between two embedding vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class TwoPassAligner:
    """
    Aligns regulatory chunks between two version snapshots using:
      - Pass 1: Deterministic statutory citation/path matching + difflib content check.
      - Pass 2: Dual-signal (Dense Vector + Lexical Overlap) matching for unmatched pools.
    """

    def __init__(
        self,
        dense_weight: float = 0.6,
        min_composite_threshold: float = 0.70,
        min_lexical_threshold: float = 0.25,
    ):
        self.dense_weight = dense_weight
        self.lexical_weight = 1.0 - dense_weight
        self.min_composite_threshold = min_composite_threshold
        self.min_lexical_threshold = min_lexical_threshold

    def align(
        self,
        source_chunks: List[Dict[str, Any]],
        target_chunks: List[Dict[str, Any]],
        source_version: str,
        target_version: str,
    ) -> List[LineageRecord]:
        """
        Main entry point for aligning source (N-1) and target (N) chunk datasets.

        Expects chunk dictionaries with keys:
          - 'chunk_id': str
          - 'citation': str (e.g. "§ 7123(c)(1)")
          - 'text': str
          - 'lineage_id': Optional[str]
          - 'embedding': Optional[List[float]]
        """
        lineage_records: List[LineageRecord] = []

        unmatched_source: List[Dict[str, Any]] = []
        unmatched_target: List[Dict[str, Any]] = list(target_chunks)

        # Index target chunks by citation for Pass 1 matching
        target_by_citation: Dict[str, Dict[str, Any]] = {
            t["citation"]: t for t in target_chunks if t.get("citation")
        }

        # -------------------------------------------------------------------
        # PASS 1: Deterministic Structural Path + Content Check
        # -------------------------------------------------------------------
        for src in source_chunks:
            citation = src.get("citation")
            matched_target = target_by_citation.get(citation) if citation else None

            if matched_target:
                # Remove from unmatched target candidate pool
                unmatched_target = [
                    t for t in unmatched_target if t["chunk_id"] != matched_target["chunk_id"]
                ]

                src_text = src["text"]
                tgt_text = matched_target["text"]

                # Content Equality Check via SequenceMatcher ratio
                diff_ratio = difflib.SequenceMatcher(None, src_text, tgt_text).ratio()
                lexical_score = calculate_jaccard_overlap(src_text, tgt_text)

                lineage_id = src.get("lineage_id") or f"lineage_{uuid.uuid4().hex[:8]}"

                if diff_ratio >= 0.999:
                    change_type = ChangeType.UNCHANGED
                else:
                    change_type = ChangeType.MODIFIED

                lineage_records.append(
                    LineageRecord(
                        canonical_lineage_id=lineage_id,
                        parent_lineage_id=None,
                        source_version=source_version,
                        target_version=target_version,
                        source_chunk_id=src["chunk_id"],
                        target_chunk_id=matched_target["chunk_id"],
                        source_citation=src.get("citation"),
                        target_citation=matched_target.get("citation"),
                        change_type=change_type,
                        similarity_score=float(diff_ratio),
                        lexical_overlap_score=float(lexical_score),
                    )
                )
            else:
                unmatched_source.append(src)

        # -------------------------------------------------------------------
        # PASS 2: Dual-Signal (Dense + Lexical) Unmatched Alignment
        # -------------------------------------------------------------------
        matched_target_ids: Set[str] = set()

        for src in unmatched_source:
            best_target: Optional[Dict[str, Any]] = None
            best_composite_score = 0.0
            best_lexical_score = 0.0

            src_embed = src.get("embedding")
            src_text = src["text"]

            candidates_scores: List[Tuple[Dict[str, Any]], float, float] = []

            for tgt in unmatched_target:
                if tgt["chunk_id"] in matched_target_ids:
                    continue

                tgt_embed = tgt.get("embedding")
                tgt_text = tgt["text"]

                lexical_score = calculate_jaccard_overlap(src_text, tgt_text)

                # Skip if lexical overlap fails baseline requirement (avoids legal boilerplate false positives)
                if lexical_score < self.min_lexical_threshold:
                    continue

                if src_embed and tgt_embed:
                    dense_score = calculate_cosine_similarity(src_embed, tgt_embed)
                else:
                    dense_score = lexical_score

                composite_score = (self.dense_weight * dense_score) + (
                    self.lexical_weight * lexical_score
                )

                if composite_score >= self.min_composite_threshold:
                    candidates_scores.append((tgt, composite_score, lexical_score))

            # Sort candidate matches by composite score descending
            candidates_scores.sort(key=lambda x: x[1], reverse=True)

            if not candidates_scores:
                # No match found in target version -> DELETED
                lineage_id = src.get("lineage_id") or f"lineage_{uuid.uuid4().hex[:8]}"
                lineage_records.append(
                    LineageRecord(
                        canonical_lineage_id=lineage_id,
                        parent_lineage_id=None,
                        source_version=source_version,
                        target_version=target_version,
                        source_chunk_id=src["chunk_id"],
                        target_chunk_id=None,
                        source_citation=src.get("citation"),
                        target_citation=None,
                        change_type=ChangeType.DELETED,
                        similarity_score=None,
                        lexical_overlap_score=None,
                    )
                )
            elif len(candidates_scores) == 1:
                # Single match -> RENUMBERED or MODIFIED
                tgt, composite_score, lexical_score = candidates_scores[0]
                matched_target_ids.add(tgt["chunk_id"])

                lineage_id = src.get("lineage_id") or f"lineage_{uuid.uuid4().hex[:8]}"
                change_type = (
                    ChangeType.RENUMBERED
                    if src.get("citation") != tgt.get("citation")
                    else ChangeType.MODIFIED
                )

                lineage_records.append(
                    LineageRecord(
                        canonical_lineage_id=lineage_id,
                        parent_lineage_id=None,
                        source_version=source_version,
                        target_version=target_version,
                        source_chunk_id=src["chunk_id"],
                        target_chunk_id=tgt["chunk_id"],
                        source_citation=src.get("citation"),
                        target_citation=tgt.get("citation"),
                        change_type=change_type,
                        similarity_score=composite_score,
                        lexical_overlap_score=lexical_score,
                    )
                )
            else:
                # 1-to-Many Match -> SPLIT Event
                # Rule: Mint NEW canonical_lineage_id for each target branch.
                # Point parent_lineage_id back to origin lineage.
                origin_lineage_id = src.get("lineage_id") or f"lineage_{uuid.uuid4().hex[:8]}"

                for tgt, composite_score, lexical_score in candidates_scores:
                    matched_target_ids.add(tgt["chunk_id"])
                    branch_lineage_id = f"lineage_branch_{uuid.uuid4().hex[:8]}"

                    lineage_records.append(
                        LineageRecord(
                            canonical_lineage_id=branch_lineage_id,
                            parent_lineage_id=origin_lineage_id,
                            source_version=source_version,
                            target_version=target_version,
                            source_chunk_id=src["chunk_id"],
                            target_chunk_id=tgt["chunk_id"],
                            source_citation=src.get("citation"),
                            target_citation=tgt.get("citation"),
                            change_type=ChangeType.SPLIT,
                            similarity_score=composite_score,
                            lexical_overlap_score=lexical_score,
                        )
                    )

        # -------------------------------------------------------------------
        # Process Remaining Unmatched Target Chunks -> ADDED
        # -------------------------------------------------------------------
        for tgt in unmatched_target:
            if tgt["chunk_id"] not in matched_target_ids:
                new_lineage_id = f"lineage_{uuid.uuid4().hex[:8]}"
                lineage_records.append(
                    LineageRecord(
                        canonical_lineage_id=new_lineage_id,
                        parent_lineage_id=None,
                        source_version=source_version,
                        target_version=target_version,
                        source_chunk_id=None,
                        target_chunk_id=tgt["chunk_id"],
                        source_citation=None,
                        target_citation=tgt.get("citation"),
                        change_type=ChangeType.ADDED,
                        similarity_score=None,
                        lexical_overlap_score=None,
                    )
                )

        return lineage_records