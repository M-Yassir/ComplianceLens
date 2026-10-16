"""
Evaluation Harness for CCPA Cross-Version & Policy RAG Pipeline
File: eval/eval_harness.py

Executes systematic end-to-end benchmark evaluation over eval/golden_dataset.json across:
  1. Intent Routing Accuracy & Out-of-Scope Abstention Precision
  2. Section Retrieval Recall (Hit Rate), Top-1 Precision, and Precision@K
  3. Generation Quality (Key-Term Concept Recall, Citation Faithfulness, Semantic Similarity, LLM Judge)
"""

import os
import json
import argparse
import random
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

# Pipeline & Module Imports
from src.rag.pipeline import RAGPipeline, RAGResponse
from src.versioning.section_key_rag import SectionKeyCrossVersionRAG
from src.vector_store.embeddings import get_embedding_function

OUT_OF_SCOPE_TYPES = {"OUT_OF_SCOPE", "UNANSWERABLE", "ABSTAIN", "NO_ANSWER"}


# ---------------------------------------------------------------------------
# Dataset Helpers (Informed by eval_retrieval_tuning.py)
# ---------------------------------------------------------------------------
def parse_expected_sections(entry: Dict[str, Any]) -> List[str]:
    """Extracts normalized statutory section numbers as a clean list of strings."""
    val = entry.get("section", entry.get("expected_section"))
    if val is None or val == "":
        return []
    vals = val if isinstance(val, list) else [val]
    return [str(v).strip().lstrip("§").strip() for v in vals if str(v).strip()]


def is_query_answerable(entry: Dict[str, Any]) -> Optional[bool]:
    """Determines whether a query is answerable or out-of-scope."""
    if isinstance(entry.get("answerable"), bool):
        return entry["answerable"]
    if str(entry.get("query_type", "")).upper() in OUT_OF_SCOPE_TYPES:
        return False
    if parse_expected_sections(entry):
        return True
    return None  # Unlabeled


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------
@dataclass
class SingleEvalResult:
    eval_id: str
    query: str
    target_section: str
    expected_route: str
    actual_route: str
    route_correct: bool
    route_reasoning: str
    answerable: Optional[bool]
    abstained: bool
    section_hit: bool           # Target section present in final context (Recall@K)
    hit_in_top_1: bool          # Target section is Rank 1 (Top-1 Precision)
    precision_at_k: float       # Ratio of relevant chunks in context
    key_term_recall: float      # % of ground truth key terms present in answer
    citation_valid: bool        # Guardrail citation verification result
    ground_truth_answer: str
    generated_answer: str
    semantic_similarity: float  # Cosine similarity between ground truth & generated
    llm_judge_score: Optional[float] = None
    llm_judge_reasoning: Optional[str] = None
    execution_time_sec: float = 0.0
    retrieved_sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AggregateEvalMetrics:
    total_queries: int = 0
    answerable_queries: int = 0
    out_of_scope_queries: int = 0
    routing_accuracy: float = 0.0
    section_hit_rate: float = 0.0            # Recall@K (Answerable)
    top_1_precision_rate: float = 0.0        # Top-1 Hit Rate (Answerable)
    mean_precision_at_k: float = 0.0         # Precision@K (Answerable)
    false_abstention_rate: float = 0.0       # Answerable queries refused
    out_of_scope_abstention_rate: float = 0.0 # OOS queries correctly refused
    mean_key_term_recall: float = 0.0
    mean_semantic_similarity: float = 0.0
    citation_validity_rate: float = 0.0
    mean_llm_judge_score: Optional[float] = None
    by_route_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation Harness Class
# ---------------------------------------------------------------------------
class CCPAEvalHarness:
    """Automated end-to-end benchmark harness for validating CCPA RAG system accuracy."""

    def __init__(
        self,
        dataset_path: str = "eval/golden_dataset.json",
        use_llm_judge: bool = False,
        min_rerank_threshold: float = 0.35,
    ):
        self.dataset_path = dataset_path
        self.use_llm_judge = use_llm_judge

        # Initialize core pipeline modules
        self.rag_pipeline = RAGPipeline(min_rerank_score=min_rerank_threshold)
        self.cross_version_rag = SectionKeyCrossVersionRAG()
        self.embedder = get_embedding_function()

    def load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Golden dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _compute_key_term_recall(generated_text: str, key_terms: List[str]) -> float:
        """Calculates proportion of ground-truth key terms present in generated answer."""
        if not key_terms:
            return 1.0
        text_lower = generated_text.lower()
        hits = sum(1 for term in key_terms if term.lower() in text_lower)
        return round(hits / len(key_terms), 4)

    def _compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Calculates cosine similarity between embeddings of ground-truth and generated answers."""
        if not text_a or not text_b:
            return 0.0
        try:
            vec_a = self.embedder.embed_query(text_a)
            vec_b = self.embedder.embed_query(text_b)

            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = sum(a * a for a in vec_a) ** 0.5
            norm_b = sum(b * b for b in vec_b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return round(dot / (norm_a * norm_b), 4)
        except Exception as e:
            return 0.0

    def _evaluate_llm_judge(self, query: str, ground_truth: str, generated: str) -> tuple[float, str]:
        """Evaluates answer correctness using LLM-as-a-Judge on a 1-5 scale."""
        prompt = f"""You are a legal evaluator judging a CCPA regulatory RAG system.
Evaluate the candidate answer against the ground truth answer for the query below.

Query: {query}
Ground Truth: {ground_truth}
Candidate Answer: {generated}

Score the answer from 1 to 5 based on legal accuracy, completeness, and lack of hallucination:
1 = Factually incorrect or completely missed the rule.
3 = Partially correct but missed key statutory nuances.
5 = Legally precise, complete, and fully aligned with ground truth.

Respond ONLY with JSON formatted exactly like:
{{"score": 5.0, "reasoning": "Direct explanation of evaluation"}}
"""
        try:
            response = self.rag_pipeline.llm.invoke(prompt)
            content = getattr(response, "content", str(response)).strip()
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(content)
            return float(data.get("score", 0.0)), data.get("reasoning", "")
        except Exception as e:
            return 0.0, f"Judge evaluation error: {str(e)}"

    def run_evaluation(
        self,
        max_samples: Optional[int] = None,
        in_order: bool = False,
        seed: int = 7,
        include_diff: bool = False,
    ) -> tuple[AggregateEvalMetrics, List[SingleEvalResult]]:
        
        raw_dataset = self.load_dataset()

        # Filter VERSION_DIFF rows if not requested
        dataset = raw_dataset
        if not include_diff:
            dataset = [e for e in raw_dataset if str(e.get("query_type", "")).upper() != "VERSION_DIFF"]

        # Sample subset if max_samples is set
        if max_samples and max_samples < len(dataset):
            if in_order:
                dataset = dataset[:max_samples]
            else:
                dataset = random.Random(seed).sample(dataset, max_samples)

        results: List[SingleEvalResult] = []

        print(f"\n🚀 Starting End-to-End Evaluation over {len(dataset)} Golden Test Cases...\n" + "=" * 80)

        for idx, entry in enumerate(dataset, 1):
            eval_id = entry.get("id", f"eval_{idx:03d}")
            query = entry["query"]
            expected_route = entry.get("query_type", "POLICY_LOOKUP")
            expected_secs = parse_expected_sections(entry)
            target_sec_str = ",".join(expected_secs)
            ground_truth = entry.get("ground_truth_answer", "")
            key_terms = entry.get("key_terms", [])
            answerable = is_query_answerable(entry)

            start_time = time.time()

            # Execute via Cross-Version RAG engine or Main RAG Pipeline
            if expected_route == "VERSION_DIFF":
                sec_to_compare = expected_secs[0] if expected_secs else "7001"
                comp_res = self.cross_version_rag.query_section_changes(
                    user_query=query, extracted_sections=expected_secs
                )
                generated_answer = comp_res.answer
                actual_route = "VERSION_DIFF"
                route_reasoning = "Executed via SectionKeyCrossVersionRAG engine."
                abstained = comp_res.abstained
                citation_valid = not abstained
                sources_meta = []
                retrieved_secs = [str(s) for s in comp_res.sections_analyzed]
            else:
                rag_resp: RAGResponse = self.rag_pipeline.query(query)
                generated_answer = rag_resp.answer
                actual_route = rag_resp.routing_decision.route.value
                route_reasoning = rag_resp.routing_decision.reasoning
                abstained = rag_resp.abstention_decision.should_abstain
                citation_valid = rag_resp.citation_verification.is_valid
                sources_meta = rag_resp.source_documents
                retrieved_secs = [str(doc.get("section", "")).strip() for doc in sources_meta if doc.get("section")]

            exec_time = round(time.time() - start_time, 3)

            # Retrieval Metrics
            section_hit = bool(expected_secs) and any(s in expected_secs for s in retrieved_secs)
            hit_in_top_1 = bool(expected_secs) and len(retrieved_secs) > 0 and (retrieved_secs[0] in expected_secs)

            relevant_count = sum(1 for s in retrieved_secs if s in expected_secs) if expected_secs else 0
            precision_at_k = round(relevant_count / len(retrieved_secs), 4) if retrieved_secs else 0.0

            # Answer Metrics
            route_correct = (actual_route == expected_route)
            term_recall = self._compute_key_term_recall(generated_answer, key_terms)
            sem_sim = self._compute_semantic_similarity(ground_truth, generated_answer)

            # LLM Judge Evaluation
            llm_score, llm_reason = None, None
            if self.use_llm_judge and not abstained and ground_truth:
                llm_score, llm_reason = self._evaluate_llm_judge(query, ground_truth, generated_answer)

            res = SingleEvalResult(
                eval_id=eval_id,
                query=query,
                target_section=target_sec_str,
                expected_route=expected_route,
                actual_route=actual_route,
                route_correct=route_correct,
                route_reasoning=route_reasoning,
                answerable=answerable,
                abstained=abstained,
                section_hit=section_hit,
                hit_in_top_1=hit_in_top_1,
                precision_at_k=precision_at_k,
                key_term_recall=term_recall,
                citation_valid=citation_valid,
                ground_truth_answer=ground_truth,
                generated_answer=generated_answer,
                semantic_similarity=sem_sim,
                llm_judge_score=llm_score,
                llm_judge_reasoning=llm_reason,
                execution_time_sec=exec_time,
                retrieved_sources=sources_meta,
            )
            results.append(res)

            # Format CLI status string
            if answerable is False:
                status = "✅ OOS ABSTAIN" if abstained else "❌ FAILED TO ABSTAIN"
            elif section_hit and term_recall >= 0.6:
                status = "✅ PASS"
            elif abstained:
                status = "⛔ FALSE ABSTAIN"
            else:
                status = "⚠️ WARN"

            print(f"[{idx:03d}/{len(dataset):03d}] {eval_id:<14} | Target §{target_sec_str or '-':<6} | "
                  f"Route: {actual_route:<13} | Term Recall: {term_recall*100:.1f}% | SemSim: {sem_sim:.3f} | {status}")

        aggregates = self._calculate_aggregates(results)
        return aggregates, results

    def _calculate_aggregates(self, results: List[SingleEvalResult]) -> AggregateEvalMetrics:
        total = len(results)
        if total == 0:
            return AggregateEvalMetrics()

        ans_results = [r for r in results if r.answerable is not False]
        ans_with_target = [r for r in ans_results if r.target_section]
        oos_results = [r for r in results if r.answerable is False]

        route_hits = sum(1 for r in results if r.route_correct)
        
        # Retrieval Aggregates (Answerable queries)
        sec_hits = sum(1 for r in ans_with_target if r.section_hit)
        top1_hits = sum(1 for r in ans_with_target if r.hit_in_top_1)
        mean_prec = (sum(r.precision_at_k for r in ans_with_target) / len(ans_with_target)) if ans_with_target else 0.0

        # Abstention Aggregates
        false_abstains = sum(1 for r in ans_results if r.abstained)
        oos_abstains = sum(1 for r in oos_results if r.abstained)

        # Generation Aggregates
        mean_term_recall = (sum(r.key_term_recall for r in ans_with_target) / len(ans_with_target)) if ans_with_target else 0.0
        mean_sem_sim = (sum(r.semantic_similarity for r in ans_with_target) / len(ans_with_target)) if ans_with_target else 0.0
        
        non_abstained = [r for r in results if not r.abstained]
        citation_hits = sum(1 for r in non_abstained if r.citation_valid)

        judge_scores = [r.llm_judge_score for r in results if r.llm_judge_score is not None]
        mean_judge = (sum(judge_scores) / len(judge_scores)) if judge_scores else None

        # Per-Route Breakdown
        by_route: Dict[str, Dict[str, float]] = {}
        for route_name in ["POLICY_LOOKUP", "VERSION_DIFF", "HYBRID_BOTH", "OUT_OF_SCOPE"]:
            route_subset = [r for r in results if r.expected_route == route_name]
            if route_subset:
                sub_total = len(route_subset)
                sub_target = [r for r in route_subset if r.target_section]
                by_route[route_name] = {
                    "count": sub_total,
                    "section_hit_rate": round(sum(1 for r in sub_target if r.section_hit) / len(sub_target), 4) if sub_target else 0.0,
                    "mean_term_recall": round(sum(r.key_term_recall for r in route_subset) / sub_total, 4),
                    "mean_semantic_similarity": round(sum(r.semantic_similarity for r in route_subset) / sub_total, 4),
                }

        return AggregateEvalMetrics(
            total_queries=total,
            answerable_queries=len(ans_results),
            out_of_scope_queries=len(oos_results),
            routing_accuracy=round(route_hits / total, 4),
            section_hit_rate=round(sec_hits / len(ans_with_target), 4) if ans_with_target else 0.0,
            top_1_precision_rate=round(top1_hits / len(ans_with_target), 4) if ans_with_target else 0.0,
            mean_precision_at_k=round(mean_prec, 4),
            false_abstention_rate=round(false_abstains / len(ans_results), 4) if ans_results else 0.0,
            out_of_scope_abstention_rate=round(oos_abstains / len(oos_results), 4) if oos_results else 0.0,
            mean_key_term_recall=round(mean_term_recall, 4),
            mean_semantic_similarity=round(mean_sem_sim, 4),
            citation_validity_rate=round(citation_hits / len(non_abstained), 4) if non_abstained else 1.0,
            mean_llm_judge_score=round(mean_judge, 2) if mean_judge else None,
            by_route_performance=by_route,
        )

    def print_summary_report(self, aggregates: AggregateEvalMetrics):
        """Prints a clean CLI executive benchmark summary."""
        print("\n" + "=" * 80)
        print("📊 CCPA RAG SYSTEM END-TO-END BENCHMARK REPORT")
        print("=" * 80)
        print(f"Total Test Cases Evaluated : {aggregates.total_queries} (Answerable: {aggregates.answerable_queries} | Out-of-Scope: {aggregates.out_of_scope_queries})")
        print(f"Routing Classification Acc : {aggregates.routing_accuracy * 100:.2f}%")
        print("-" * 80)
        print("RETRIEVAL PERFORMANCE:")
        print(f"  • Section Hit Rate (Recall) : {aggregates.section_hit_rate * 100:.2f}% (Target section in final context)")
        print(f"  • Top-1 Section Precision   : {aggregates.top_1_precision_rate * 100:.2f}% (Target section at Rank 1)")
        print(f"  • Context Precision@K       : {aggregates.mean_precision_at_k * 100:.2f}% (Relevant sources / total sources)")
        print("-" * 80)
        print("GUARDRAIL & ABSTENTION PERFORMANCE:")
        print(f"  • False Abstention Rate     : {aggregates.false_abstention_rate * 100:.2f}% (Valid queries refused)")
        if aggregates.out_of_scope_queries > 0:
            print(f"  • Out-of-Scope Abstention   : {aggregates.out_of_scope_abstention_rate * 100:.2f}% (Irrelevant queries correctly refused)")
        print(f"  • Citation Verification Rate: {aggregates.citation_validity_rate * 100:.2f}% (Valid citations in non-abstained answers)")
        print("-" * 80)
        print("GENERATION & ANSWER QUALITY:")
        print(f"  • Key-Term Concept Recall   : {aggregates.mean_key_term_recall * 100:.2f}%")
        print(f"  • Semantic Answer Similarity: {aggregates.mean_semantic_similarity:.4f}")
        if aggregates.mean_llm_judge_score:
            print(f"  • LLM Judge Quality Score   : {aggregates.mean_llm_judge_score:.2f} / 5.00")
        print("-" * 80)
        print("PER-ROUTE BREAKDOWN:")
        for route, metrics in aggregates.by_route_performance.items():
            print(f"  • {route:<15} | Count: {metrics['count']:<2} | Sec Hit: {metrics['section_hit_rate']*100:.1f}% | Term Recall: {metrics['mean_term_recall']*100:.1f}% | SemSim: {metrics['mean_semantic_similarity']:.3f}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CCPA RAG Pipeline against Golden Dataset")
    parser.add_argument("--dataset", type=str, default="eval/golden_dataset.json", help="Path to golden dataset JSON")
    parser.add_argument("--output", type=str, default="eval/eval_results.json", help="Path to export evaluation results JSON")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM-as-a-Judge scoring")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--in-order", action="store_true", help="Take first N samples sequentially instead of random sampling")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling")
    parser.add_argument("--include-diff", action="store_true", help="Include VERSION_DIFF query types")

    args = parser.parse_args()

    harness = CCPAEvalHarness(
        dataset_path=args.dataset,
        use_llm_judge=args.llm_judge,
    )

    aggregates, detailed_results = harness.run_evaluation(
        max_samples=args.max_samples,
        in_order=args.in_order,
        seed=args.seed,
        include_diff=args.include_diff,
    )
    
    harness.print_summary_report(aggregates)

    export_payload = {
        "summary": asdict(aggregates),
        "results": [asdict(r) for r in detailed_results],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, default=str)

    print(f"📁 Detailed evaluation results exported to: {args.output}")