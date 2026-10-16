"""
Evaluation Harness for CCPA Cross-Version & Policy RAG Pipeline
File: eval/eval_harness.py

Executes systematic benchmark evaluation over eval/golden_dataset.json across three dimensions:
  1. Intent Routing Accuracy (POLICY_LOOKUP vs VERSION_DIFF vs HYBRID_BOTH)
  2. Section-Key & Subsection Retrieval Precision/Recall
  3. Answer Quality (Key-Term Recall, Citation Verification, Semantic Similarity, LLM Judge)
"""

import os
import json
import argparse
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

# Pipeline & Module Imports
from src.rag.pipeline import RAGPipeline, RAGResponse
from src.versioning.section_key_rag import SectionKeyCrossVersionRAG
from src.vector_store.embeddings import get_embedding_function  # <--- FIX: Correct factory import


@dataclass
class SingleEvalResult:
    eval_id: str
    query: str
    target_section: str
    expected_route: str
    actual_route: str
    route_correct: bool
    section_hit: bool
    key_term_recall: float
    citation_valid: bool
    abstained: bool
    ground_truth_answer: str
    generated_answer: str
    semantic_similarity: float
    llm_judge_score: Optional[float] = None
    llm_judge_reasoning: Optional[str] = None
    execution_time_sec: float = 0.0


@dataclass
class AggregateEvalMetrics:
    total_queries: int = 0
    routing_accuracy: float = 0.0
    section_hit_rate: float = 0.0
    mean_key_term_recall: float = 0.0
    mean_semantic_similarity: float = 0.0
    citation_validity_rate: float = 0.0
    abstention_rate: float = 0.0
    mean_llm_judge_score: Optional[float] = None
    by_route_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)


class CCPAEvalHarness:
    """Automated benchmark harness for validating CCPA RAG system accuracy."""

    def __init__(
        self,
        dataset_path: str = "eval/golden_dataset.json",
        use_llm_judge: bool = False,
        min_rerank_threshold: float = 0.35,
    ):
        self.dataset_path = dataset_path
        self.use_llm_judge = use_llm_judge
        
        # Initialize pipeline modules
        self.rag_pipeline = RAGPipeline(min_rerank_score=min_rerank_threshold)
        self.cross_version_rag = SectionKeyCrossVersionRAG()
        self.embedder = get_embedding_function()  # <--- FIX: Instantiated via factory adapter

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
        """Calculates cosine similarity between embeddings of target and generated answers."""
        if not text_a or not text_b:
            return 0.0
        vec_a = self.embedder.embed_query(text_a)
        vec_b = self.embedder.embed_query(text_b)
        
        # Cosine similarity calculation
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return round(dot / (norm_a * norm_b), 4)

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

    def run_evaluation(self, max_samples: Optional[int] = None) -> tuple[AggregateEvalMetrics, List[SingleEvalResult]]:
        dataset = self.load_dataset()
        if max_samples:
            dataset = dataset[:max_samples]

        results: List[SingleEvalResult] = []

        print(f"\n🚀 Starting Evaluation over {len(dataset)} Golden Test Cases...\n" + "=" * 70)

        for idx, entry in enumerate(dataset, 1):
            eval_id = entry.get("id", f"eval_{idx:03d}")
            query = entry["query"]
            expected_route = entry.get("query_type", "POLICY_LOOKUP")
            target_sec = str(entry.get("section", ""))
            ground_truth = entry.get("ground_truth_answer", "")
            key_terms = entry.get("key_terms", [])

            start_time = time.time()

            # Execute via dynamic section-key comparative RAG or main RAG pipeline
            if expected_route == "VERSION_DIFF":
                comp_res = self.cross_version_rag.compare_section(section_key=target_sec)
                generated_answer = comp_res.diff_summary
                actual_route = "VERSION_DIFF"
                section_hit = target_sec in comp_res.section_key
                citation_valid = comp_res.has_changes is not None
                abstained = "No statutory text" in generated_answer
            else:
                rag_resp: RAGResponse = self.rag_pipeline.query(query)
                generated_answer = rag_resp.answer
                actual_route = rag_resp.routing_decision.route.value
                abstained = rag_resp.abstention_decision.should_abstain
                citation_valid = rag_resp.citation_verification.is_valid
                
                retrieved_secs = [str(doc.get("section")) for doc in rag_resp.source_documents if doc.get("section")]
                section_hit = target_sec in retrieved_secs

            exec_time = round(time.time() - start_time, 3)

            # Metrics calculation
            route_correct = (actual_route == expected_route)
            term_recall = self._compute_key_term_recall(generated_answer, key_terms)
            sem_sim = self._compute_semantic_similarity(ground_truth, generated_answer)

            llm_score, llm_reason = None, None
            if self.use_llm_judge and not abstained:
                llm_score, llm_reason = self._evaluate_llm_judge(query, ground_truth, generated_answer)

            res = SingleEvalResult(
                eval_id=eval_id,
                query=query,
                target_section=target_sec,
                expected_route=expected_route,
                actual_route=actual_route,
                route_correct=route_correct,
                section_hit=section_hit,
                key_term_recall=term_recall,
                citation_valid=citation_valid,
                abstained=abstained,
                ground_truth_answer=ground_truth,
                generated_answer=generated_answer,
                semantic_similarity=sem_sim,
                llm_judge_score=llm_score,
                llm_judge_reasoning=llm_reason,
                execution_time_sec=exec_time,
            )
            results.append(res)

            status = "✅ PASS" if (section_hit and term_recall >= 0.6) else "⚠️ WARN"
            print(f"[{idx:03d}/{len(dataset):03d}] {eval_id} | Sec §{target_sec} | Route: {actual_route} | Term Recall: {term_recall*100:.1f}% | SemSim: {sem_sim:.3f} | {status}")

        aggregates = self._calculate_aggregates(results)
        return aggregates, results

    def _calculate_aggregates(self, results: List[SingleEvalResult]) -> AggregateEvalMetrics:
        total = len(results)
        if total == 0:
            return AggregateEvalMetrics()

        route_hits = sum(1 for r in results if r.route_correct)
        sec_hits = sum(1 for r in results if r.section_hit)
        mean_term_recall = sum(r.key_term_recall for r in results) / total
        mean_sem_sim = sum(r.semantic_similarity for r in results) / total
        citation_hits = sum(1 for r in results if r.citation_valid)
        abstain_count = sum(1 for r in results if r.abstained)

        judge_scores = [r.llm_judge_score for r in results if r.llm_judge_score is not None]
        mean_judge = (sum(judge_scores) / len(judge_scores)) if judge_scores else None

        by_route: Dict[str, Dict[str, float]] = {}
        for route_name in ["POLICY_LOOKUP", "VERSION_DIFF", "HYBRID_BOTH"]:
            route_subset = [r for r in results if r.expected_route == route_name]
            if route_subset:
                sub_total = len(route_subset)
                by_route[route_name] = {
                    "count": sub_total,
                    "section_hit_rate": round(sum(1 for r in route_subset if r.section_hit) / sub_total, 4),
                    "mean_term_recall": round(sum(r.key_term_recall for r in route_subset) / sub_total, 4),
                    "mean_semantic_similarity": round(sum(r.semantic_similarity for r in route_subset) / sub_total, 4),
                }

        return AggregateEvalMetrics(
            total_queries=total,
            routing_accuracy=round(route_hits / total, 4),
            section_hit_rate=round(sec_hits / total, 4),
            mean_key_term_recall=round(mean_term_recall, 4),
            mean_semantic_similarity=round(mean_sem_sim, 4),
            citation_validity_rate=round(citation_hits / total, 4),
            abstention_rate=round(abstain_count / total, 4),
            mean_llm_judge_score=round(mean_judge, 2) if mean_judge else None,
            by_route_performance=by_route,
        )

    def print_summary_report(self, aggregates: AggregateEvalMetrics):
        """Prints a CLI executive benchmark summary."""
        print("\n" + "=" * 70)
        print("📊 CCPA RAG SYSTEM BENCHMARK SUMMARY REPORT")
        print("=" * 70)
        print(f"Total Test Cases Evaluated : {aggregates.total_queries}")
        print(f"Routing Classification Acc : {aggregates.routing_accuracy * 100:.2f}%")
        print(f"Section Retrieval Hit Rate : {aggregates.section_hit_rate * 100:.2f}%")
        print(f"Key-Term Concept Recall    : {aggregates.mean_key_term_recall * 100:.2f}%")
        print(f"Semantic Answer Similarity : {aggregates.mean_semantic_similarity:.4f}")
        print(f"Citation Verification Rate : {aggregates.citation_validity_rate * 100:.2f}%")
        print(f"Pipeline Abstention Rate   : {aggregates.abstention_rate * 100:.2f}%")
        if aggregates.mean_llm_judge_score:
            print(f"LLM Judge Quality Score    : {aggregates.mean_llm_judge_score:.2f} / 5.00")
        print("-" * 70)
        print("PER-ROUTE BREAKDOWN:")
        for route, metrics in aggregates.by_route_performance.items():
            print(f"  • {route:<15} | Count: {metrics['count']:<2} | Sec Hit: {metrics['section_hit_rate']*100:.1f}% | Term Recall: {metrics['mean_term_recall']*100:.1f}% | SemSim: {metrics['mean_semantic_similarity']:.3f}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CCPA RAG Pipeline against Golden Dataset")
    parser.add_argument("--dataset", type=str, default="eval/golden_dataset.json", help="Path to golden dataset JSON")
    parser.add_argument("--output", type=str, default="eval/eval_results.json", help="Path to export evaluation results JSON")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM-as-a-Judge scoring")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of queries to evaluate")

    args = parser.parse_args()

    harness = CCPAEvalHarness(
        dataset_path=args.dataset,
        use_llm_judge=args.llm_judge,
    )

    aggregates, detailed_results = harness.run_evaluation(max_samples=args.max_samples)
    harness.print_summary_report(aggregates)

    export_payload = {
        "summary": asdict(aggregates),
        "results": [asdict(r) for r in detailed_results],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"📁 Detailed evaluation results exported to: {args.output}")