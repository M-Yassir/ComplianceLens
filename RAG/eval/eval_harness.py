"""
Evaluation Harness for CCPA Cross-Version & Policy RAG Pipeline
File: eval/eval_harness.py

Executes systematic end-to-end benchmark evaluation over eval/golden_dataset.json across:
  1. Intent Routing Accuracy & Out-of-Scope Abstention Precision
  2. Section Retrieval Recall (Hit Rate), Top-1 Precision, and Precision@K
  3. Generation Quality (Key-Term Concept Recall, Citation Faithfulness, Semantic Similarity, LLM Judge)
"""

import os
import re
import json
import argparse
import random
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field

# Pipeline & Module Imports
from src.rag.pipeline import RAGPipeline, RAGResponse
from src.generation.llm_service import get_llm, _get_text_content
from src.vector_store.embeddings import get_embedding_function
# NOTE: the cross-version engine is imported lazily (only with --include-diff) so a
# problem in that module can't take down the policy-lookup evaluation.

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


def fmt_pct(v: Optional[float]) -> str:
    return "  -  " if v is None else f"{v * 100:.1f}%"


def fmt_num(v: Optional[float]) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


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
    key_term_recall: Optional[float]  # % of key terms in answer (None: no key_terms or abstained)
    citation_valid: bool        # Guardrail citation verification result
    ground_truth_answer: str
    generated_answer: str
    semantic_similarity: Optional[float]  # Cosine sim (None: abstained, or embedding call failed)
    llm_judge_score: Optional[float] = None
    llm_judge_reasoning: Optional[str] = None
    model_refusal: bool = False  # out-of-scope question the retrieval guard let through but the model declined (judge >= 4)
    execution_time_sec: float = 0.0
    retrieved_sources: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None  # set when the pipeline call itself failed (excluded from metrics)


@dataclass
class AggregateEvalMetrics:
    total_queries: int = 0
    errored_queries: int = 0
    answered_queries: int = 0                # answerable AND not abstained (generation metrics use these)
    answerable_queries: int = 0
    out_of_scope_queries: int = 0
    routing_accuracy: float = 0.0
    section_hit_rate: float = 0.0            # Recall@K (Answerable)
    top_1_precision_rate: float = 0.0        # Top-1 Hit Rate (Answerable)
    mean_precision_at_k: float = 0.0         # Precision@K (Answerable)
    false_abstention_rate: float = 0.0       # Answerable queries refused
    out_of_scope_abstention_rate: float = 0.0 # OOS queries correctly refused (guard OR model, see below)
    out_of_scope_model_refusals: int = 0      # ...of which refused by the model, detected by the LLM judge
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
        include_diff: bool = False,
    ):
        self.dataset_path = dataset_path
        self.use_llm_judge = use_llm_judge
        # temperature 0 so judge scores are reproducible between runs
        self.judge_llm = get_llm(temperature=0) if use_llm_judge else None

        # Initialize core pipeline modules
        self.rag_pipeline = RAGPipeline(min_rerank_score=min_rerank_threshold)
        self.cross_version_rag = None
        if include_diff:
            from src.versioning.section_key_rag import SectionKeyCrossVersionRAG
            self.cross_version_rag = SectionKeyCrossVersionRAG()
        self.embedder = get_embedding_function()

    def load_dataset(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Golden dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _compute_key_term_recall(generated_text: str, key_terms: List[str]) -> Optional[float]:
        """Proportion of ground-truth key terms present in the answer (None if there are none:
        returning 1.0 there would silently inflate the mean)."""
        if not key_terms:
            return None
        text_lower = generated_text.lower()
        hits = sum(1 for term in key_terms if term.lower() in text_lower)
        return round(hits / len(key_terms), 4)

    def _compute_semantic_similarity(self, text_a: str, text_b: str) -> Optional[float]:
        """Calculates cosine similarity between embeddings of ground-truth and generated answers."""
        if not text_a or not text_b:
            return None
        try:
            vec_a = self.embedder.embed_query(text_a)
            vec_b = self.embedder.embed_query(text_b)

            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = sum(a * a for a in vec_a) ** 0.5
            norm_b = sum(b * b for b in vec_b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return None
            return round(dot / (norm_a * norm_b), 4)
        except Exception as e:
            # A failed embedding call (e.g. 429) must not be recorded as "similarity 0.0".
            print(f"    ⚠️ semantic similarity unavailable: {str(e)[:100]}")
            return None

    def _evaluate_llm_judge(self, query: str, ground_truth: str, generated: str) -> tuple[Optional[float], str]:
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

If the Ground Truth says the system should decline (e.g. "Unable to provide a confident answer"), score by whether the candidate declines:
5 = clearly says it cannot answer / the information is not available, without answering the question anyway;
3 = declines part of the question but substantively answers the rest;
1 = gives a substantive answer anyway.

Respond ONLY with JSON formatted exactly like:
{{"score": 5.0, "reasoning": "Direct explanation of evaluation"}}
"""
        try:
            response = self.judge_llm.invoke(prompt)
            # Gemini-style responses can be a list of parts; the pipeline already normalises this.
            content = _get_text_content(getattr(response, "content", response)).strip()
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)  # tolerate ```json fences / prose
            data = json.loads(match.group(0) if match else content)
            return float(data["score"]), data.get("reasoning", "")
        except Exception as e:
            # Judge failures are None (excluded from the mean), never a fake 0.0 score.
            return None, f"Judge evaluation error: {str(e)}"

    def run_evaluation(
        self,
        max_samples: Optional[int] = None,
        in_order: bool = False,
        seed: int = 7,
        include_diff: bool = False,
        sleep_between: float = 0.0,
        output_path: Optional[str] = None,
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

            # Execute via Cross-Version RAG engine or Main RAG Pipeline.
            # One failing query (429, malformed LLM output...) must not lose the whole run.
            try:
                if expected_route == "VERSION_DIFF":
                    if self.cross_version_rag is None:
                        raise RuntimeError("VERSION_DIFF row but cross-version engine not loaded (use --include-diff)")
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
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"[{idx:03d}/{len(dataset):03d}] {eval_id:<14} | 💥 ERROR {msg}")
                results.append(SingleEvalResult(
                    eval_id=eval_id, query=query, target_section=target_sec_str,
                    expected_route=expected_route, actual_route="ERROR", route_correct=False,
                    route_reasoning="", answerable=answerable, abstained=False, section_hit=False,
                    hit_in_top_1=False, precision_at_k=0.0, key_term_recall=None, citation_valid=False,
                    ground_truth_answer=ground_truth, generated_answer="", semantic_similarity=None,
                    execution_time_sec=round(time.time() - start_time, 3), error=msg,
                ))
                continue

            exec_time = round(time.time() - start_time, 3)

            # Retrieval Metrics
            section_hit = bool(expected_secs) and any(s in expected_secs for s in retrieved_secs)
            hit_in_top_1 = bool(expected_secs) and len(retrieved_secs) > 0 and (retrieved_secs[0] in expected_secs)

            relevant_count = sum(1 for s in retrieved_secs if s in expected_secs) if expected_secs else 0
            precision_at_k = round(relevant_count / len(retrieved_secs), 4) if retrieved_secs else 0.0

            # Answer Metrics
            route_correct = (actual_route == expected_route)
            # Generation metrics only make sense for a real answer to an answerable question;
            # on an abstention they would score the canned refusal text.
            scored_answer = (answerable is not False) and not abstained
            term_recall = self._compute_key_term_recall(generated_answer, key_terms) if scored_answer else None
            sem_sim = self._compute_semantic_similarity(ground_truth, generated_answer) if scored_answer else None

            # LLM Judge Evaluation
            llm_score, llm_reason = None, None
            # Judge answered questions, and also out-of-scope questions the guard did NOT abstain on
            # (their ground truth is a refusal, so the judge checks whether the model declined).
            if self.use_llm_judge and not abstained and ground_truth and (scored_answer or answerable is False):
                llm_score, llm_reason = self._evaluate_llm_judge(query, ground_truth, generated_answer)

            # Effective abstention: the retrieval score can't tell "on topic" from "answers the question", so an
            # out-of-scope question counts as refused when the model itself declined (judge score >= 4).
            model_refusal = bool(answerable is False and not abstained and llm_score is not None and llm_score >= 4)
            if model_refusal:
                abstained = True

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
                model_refusal=model_refusal,
                execution_time_sec=exec_time,
                retrieved_sources=sources_meta,
            )
            results.append(res)

            # Format CLI status string
            if answerable is False:
                status = ("✅ OOS REFUSED (model)" if model_refusal else "✅ OOS ABSTAIN") if abstained else "❌ FAILED TO ABSTAIN"
            elif section_hit and (term_recall is None or term_recall >= 0.6):
                status = "✅ PASS"
            elif abstained:
                status = "⛔ FALSE ABSTAIN"
            else:
                status = "⚠️ WARN"

            print(f"[{idx:03d}/{len(dataset):03d}] {eval_id:<14} | Target §{target_sec_str or '-':<6} | "
                  f"Route: {actual_route:<13} | Term Recall: {fmt_pct(term_recall)} | SemSim: {fmt_num(sem_sim)} | {status}")

            if sleep_between:
                time.sleep(sleep_between)
            if output_path and idx % 5 == 0:  # checkpoint so a crash/Ctrl-C doesn't lose the run
                self._checkpoint(output_path, results)

        aggregates = self._calculate_aggregates(results)
        return aggregates, results

    @staticmethod
    def _checkpoint(path: str, results: List[SingleEvalResult]) -> None:
        with open(path + ".partial", "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)

    def _calculate_aggregates(self, all_results: List[SingleEvalResult]) -> AggregateEvalMetrics:
        errored = sum(1 for r in all_results if r.error)
        results = [r for r in all_results if not r.error]   # failed calls are not scored
        total = len(results)
        if total == 0:
            return AggregateEvalMetrics(total_queries=len(all_results), errored_queries=errored)

        ans_results = [r for r in results if r.answerable is not False]
        ans_with_target = [r for r in ans_results if r.target_section]
        oos_results = [r for r in results if r.answerable is False]

        # Routing: out-of-scope rows have no route label the router can emit, so they are
        # excluded here (their correctness is the abstention rate instead).
        routable = [r for r in results if r.answerable is not False]
        route_hits = sum(1 for r in routable if r.route_correct)

        # Retrieval Aggregates (answerable queries; an abstained query has no context -> counts as a miss,
        # use eval_retrieval_tuning.py to separate "retrieval missed" from "abstained")
        sec_hits = sum(1 for r in ans_with_target if r.section_hit)
        top1_hits = sum(1 for r in ans_with_target if r.hit_in_top_1)
        mean_prec = (sum(r.precision_at_k for r in ans_with_target) / len(ans_with_target)) if ans_with_target else 0.0

        # Abstention Aggregates
        false_abstains = sum(1 for r in ans_results if r.abstained)
        oos_abstains = sum(1 for r in oos_results if r.abstained)

        # Generation Aggregates: only answerable questions that actually got an answer
        answered = [r for r in ans_results if not r.abstained]
        mean_term_recall = _mean([r.key_term_recall for r in answered])
        mean_sem_sim = _mean([r.semantic_similarity for r in answered])

        non_abstained = [r for r in results if not r.abstained]
        citation_hits = sum(1 for r in non_abstained if r.citation_valid)

        mean_judge = _mean([r.llm_judge_score for r in ans_results])  # quality of answers only, not refusal checks

        # Per-Route Breakdown
        by_route: Dict[str, Dict[str, Any]] = {}
        for route_name in ["POLICY_LOOKUP", "VERSION_DIFF", "HYBRID_BOTH", "OUT_OF_SCOPE"]:
            route_subset = [r for r in results if r.expected_route == route_name]
            if route_subset:
                sub_target = [r for r in route_subset if r.target_section and r.answerable is not False]
                sub_answered = [r for r in route_subset if r.answerable is not False and not r.abstained]
                tr = _mean([r.key_term_recall for r in sub_answered])
                ss = _mean([r.semantic_similarity for r in sub_answered])
                by_route[route_name] = {
                    "count": len(route_subset),
                    "section_hit_rate": round(sum(1 for r in sub_target if r.section_hit) / len(sub_target), 4) if sub_target else None,
                    "mean_term_recall": round(tr, 4) if tr is not None else None,
                    "mean_semantic_similarity": round(ss, 4) if ss is not None else None,
                    "abstention_rate": round(sum(1 for r in route_subset if r.abstained) / len(route_subset), 4),
                }

        return AggregateEvalMetrics(
            total_queries=total,
            errored_queries=errored,
            answered_queries=len(answered),
            answerable_queries=len(ans_results),
            out_of_scope_queries=len(oos_results),
            routing_accuracy=round(route_hits / len(routable), 4) if routable else 0.0,
            section_hit_rate=round(sec_hits / len(ans_with_target), 4) if ans_with_target else 0.0,
            top_1_precision_rate=round(top1_hits / len(ans_with_target), 4) if ans_with_target else 0.0,
            mean_precision_at_k=round(mean_prec, 4),
            false_abstention_rate=round(false_abstains / len(ans_results), 4) if ans_results else 0.0,
            out_of_scope_abstention_rate=round(oos_abstains / len(oos_results), 4) if oos_results else 0.0,
            out_of_scope_model_refusals=sum(1 for r in oos_results if r.model_refusal),
            mean_key_term_recall=round(mean_term_recall, 4) if mean_term_recall is not None else 0.0,
            mean_semantic_similarity=round(mean_sem_sim, 4) if mean_sem_sim is not None else 0.0,
            citation_validity_rate=round(citation_hits / len(non_abstained), 4) if non_abstained else 1.0,
            mean_llm_judge_score=round(mean_judge, 2) if mean_judge is not None else None,
            by_route_performance=by_route,
        )

    def print_summary_report(self, aggregates: AggregateEvalMetrics):
        """Prints a clean CLI executive benchmark summary."""
        print("\n" + "=" * 80)
        print("📊 CCPA RAG SYSTEM END-TO-END BENCHMARK REPORT")
        print("=" * 80)
        print(f"Total Test Cases Evaluated : {aggregates.total_queries} (Answerable: {aggregates.answerable_queries} | Out-of-Scope: {aggregates.out_of_scope_queries})"
              + (f"  ⚠️ {aggregates.errored_queries} errored (excluded)" if aggregates.errored_queries else ""))
        print(f"Routing Classification Acc : {aggregates.routing_accuracy * 100:.2f}% (answerable queries only)")
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
            print(f"      of which refused by the model (needs --llm-judge): {aggregates.out_of_scope_model_refusals}")
        print(f"  • Citation Verification Rate: {aggregates.citation_validity_rate * 100:.2f}% (Valid citations in non-abstained answers)")
        print("-" * 80)
        print("GENERATION & ANSWER QUALITY:")
        print(f"  • (over {aggregates.answered_queries} answerable queries that got an answer, abstentions excluded)")
        print(f"  • Key-Term Concept Recall   : {aggregates.mean_key_term_recall * 100:.2f}%")
        print(f"  • Semantic Answer Similarity: {aggregates.mean_semantic_similarity:.4f}")
        if aggregates.mean_llm_judge_score is not None:
            print(f"  • LLM Judge Quality Score   : {aggregates.mean_llm_judge_score:.2f} / 5.00")
        print("-" * 80)
        print("PER-ROUTE BREAKDOWN:")
        for route, metrics in aggregates.by_route_performance.items():
            print(f"  • {route:<15} | Count: {metrics['count']:<3} | Sec Hit: {fmt_pct(metrics['section_hit_rate'])} | "
                  f"Term Recall: {fmt_pct(metrics['mean_term_recall'])} | SemSim: {fmt_num(metrics['mean_semantic_similarity'])} | "
                  f"Abstain: {fmt_pct(metrics['abstention_rate'])}")
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
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between queries (API rate limits)")

    args = parser.parse_args()

    harness = CCPAEvalHarness(
        dataset_path=args.dataset,
        use_llm_judge=args.llm_judge,
        include_diff=args.include_diff,
    )

    aggregates, detailed_results = harness.run_evaluation(
        max_samples=args.max_samples,
        in_order=args.in_order,
        seed=args.seed,
        include_diff=args.include_diff,
        sleep_between=args.sleep,
        output_path=args.output,
    )
    
    harness.print_summary_report(aggregates)

    export_payload = {
        "summary": asdict(aggregates),
        "results": [asdict(r) for r in detailed_results],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, default=str)

    print(f"📁 Detailed evaluation results exported to: {args.output}")