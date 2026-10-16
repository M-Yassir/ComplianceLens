"""
Retrieval Parameter Tuning & Diagnostic Harness for CCPA RAG Pipeline.
File: eval/eval_retrieval_tuning.py

Executes the retrieval-only benchmark (Routing -> Hybrid Search -> Reranking ->
Abstention/Context Selection) without LLM answer generation.

Two modes:
  1. Single config (default): runs the pipeline once per question with the CLI
     parameters and reports hit rates, abstention behaviour, per-question detail.
  2. --sweep: after the base run, finds good parameters WITHOUT re-running the
     expensive stages for every combination:
       - abstention / context-selection params (abs_floor, min_margin,
         keep_ratio, min_context_docs) are replayed offline on the cached
         reranked results;
       - top_k_reranked / max_per_section are replayed from one full scoring
         of the candidate pool per question (cross-encoder scores don't depend
         on those params, only the selection does).
     top_k_candidates changes the hybrid retrieval itself, so sweep it by
     running this script once per value.

Dataset fields used (JSON list):
  "query"                              required
  "section": "7001" | ["7001","7002"]  correct section(s); enables hit rates
  "query_type"                         POLICY_LOOKUP / HYBRID_BOTH / VERSION_DIFF /
                                       OUT_OF_SCOPE ("answerable": false also works)
VERSION_DIFF rows are skipped (they don't use this retrieval path) unless
--include-diff. OUT_OF_SCOPE rows (no "section") are what make the
abstention sweep meaningful - without them any "never abstain" setting wins.
"""

import os
import json
import argparse
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

from src.retrieval.router import QueryRouter
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.guardrails.abstention import AbstentionGuard

OUT_OF_SCOPE_TYPES = {"OUT_OF_SCOPE", "UNANSWERABLE", "ABSTAIN", "NO_ANSWER"}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def expected_sections(entry: Dict[str, Any]) -> List[str]:
    val = entry.get("section", entry.get("expected_section"))
    if val is None or val == "":
        return []
    vals = val if isinstance(val, list) else [val]
    return [str(v).strip().lstrip("§").strip() for v in vals if str(v).strip()]


def is_answerable(entry: Dict[str, Any]) -> Optional[bool]:
    if isinstance(entry.get("answerable"), bool):
        return entry["answerable"]
    if str(entry.get("query_type", "")).upper() in OUT_OF_SCOPE_TYPES:
        return False
    if expected_sections(entry):
        return True
    return None  # unlabeled


def diversify(ranked: List[Any], top_k: int, max_per_section: int, fetch_k: int) -> List[Any]:
    """Same selection logic as CrossEncoderReranker.rerank_with_diversity, applied
    to an already fully-scored list (so it can be replayed for many params)."""
    selected: List[Any] = []
    counts: Dict[str, int] = {}
    for doc in ranked[:fetch_k]:
        sec = doc.metadata.get("section", "unknown")
        if counts.get(sec, 0) < max_per_section:
            selected.append(doc)
            counts[sec] = counts.get(sec, 0) + 1
        if len(selected) == top_k:
            break
    return selected


def _sections(docs: List[Any]) -> List[str]:
    return [str(d.metadata.get("section", "")).strip() for d in docs]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class SingleRetrievalResult:
    eval_id: str
    query: str
    target_section: str
    expected_route: str
    actual_route: str
    route_reasoning: str
    extracted_sections: List[str]
    abstained: bool
    section_hit: bool          # correct section is in the FINAL context (after abstention + selection)
    hit_in_top_1: bool
    retrieved_sources: List[Dict[str, Any]]
    execution_time_sec: float
    answerable: Optional[bool] = None
    hit_in_reranked: bool = False   # correct section was in the reranked list (before abstention/selection)
    top_score: float = 0.0
    score_margin: Optional[float] = None
    context_size: int = 0
    reranked_sections: List[str] = field(default_factory=list)


@dataclass
class _Retrieved:
    """Everything expensive for one question, cached so sweeps can replay it."""
    entry: Dict[str, Any]
    eval_id: str
    query: str
    expected: List[str]
    answerable: Optional[bool]
    expected_route: str
    routing: Any
    has_sections: bool
    candidates: List[Any]
    ranked_all: Optional[List[Any]] = None
    reranked: List[Any] = field(default_factory=list)
    elapsed: float = 0.0


class RetrievalTuner:
    """Diagnostic harness for tuning retrieval and reranking parameters."""

    def __init__(
        self,
        min_rerank_score: float = 0.35,
        top_k_candidates: int = 20,
        top_k_reranked: int = 8,
        max_per_section: int = 2,
        use_llm_router: bool = False,
        abs_floor: Optional[float] = None,
        min_margin: Optional[float] = None,
        keep_ratio: Optional[float] = None,
        min_context_docs: Optional[int] = None,
        fetch_k: int = 15,
        keep_full_ranking: bool = False,
    ):
        self.min_rerank_score = min_rerank_score
        self.top_k_candidates = top_k_candidates
        self.top_k_reranked = top_k_reranked
        self.max_per_section = max_per_section
        self.fetch_k = fetch_k
        self.keep_full_ranking = keep_full_ranking
        self.guard_kwargs = dict(
            min_rerank_score=min_rerank_score,
            abs_floor=abs_floor,
            min_margin=min_margin,
            relative_keep_ratio=keep_ratio,
            min_context_docs=min_context_docs,
        )

        router_llm = None
        if use_llm_router:
            from src.generation.llm_service import get_llm
            router_llm = get_llm()

        # Pipeline modules
        self.router = QueryRouter(llm=router_llm)
        self.hybrid_retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.abstention_guard = AbstentionGuard(**self.guard_kwargs)

    # -- stages ------------------------------------------------------------
    def effective_threshold(self, has_sections: bool) -> float:
        return self.min_rerank_score if has_sections else max(self.min_rerank_score - 0.13, 0.0)

    def retrieve(self, entry: Dict[str, Any], idx: int) -> _Retrieved:
        start = time.time()
        query = entry["query"]

        routing = self.router.route_query(query)
        has_sections = len(routing.extracted_sections) > 0

        top_k = self.top_k_candidates if has_sections else max(self.top_k_candidates, 25)
        candidates = self.hybrid_retriever.get_relevant_documents(query=query, top_k=top_k)

        ranked_all = None
        if self.keep_full_ranking and candidates:
            ranked_all = self.reranker.rerank(query=query, documents=candidates, top_k=len(candidates))

        reranked = self.reranker.rerank_with_diversity(
            query=query,
            documents=candidates,
            top_k=self.top_k_reranked,
            max_per_section=self.max_per_section,
            fetch_k=self.fetch_k,
        )

        return _Retrieved(
            entry=entry,
            eval_id=entry.get("id", f"eval_{idx:03d}"),
            query=query,
            expected=expected_sections(entry),
            answerable=is_answerable(entry),
            expected_route=entry.get("query_type", "POLICY_LOOKUP"),
            routing=routing,
            has_sections=has_sections,
            candidates=candidates,
            ranked_all=ranked_all,
            reranked=reranked,
            elapsed=round(time.time() - start, 3),
        )

    def decide(self, r: _Retrieved, guard: Optional[AbstentionGuard] = None,
               reranked: Optional[List[Any]] = None) -> Tuple[Any, List[Any]]:
        guard = guard or self.abstention_guard
        docs = r.reranked if reranked is None else reranked
        abstention = guard.evaluate_retrieval(
            query=r.query,
            retrieved_docs=docs,
            threshold=self.effective_threshold(r.has_sections),
        )
        context = [] if abstention.should_abstain else guard.select_context(docs)
        return abstention, context

    def to_result(self, r: _Retrieved) -> SingleRetrievalResult:
        abstention, context_docs = self.decide(r)

        retrieved_sources = []
        retrieved_sections = []
        for rank, doc in enumerate(context_docs, 1):
            sec = str(doc.metadata.get("section", "")).strip()
            retrieved_sections.append(sec)
            retrieved_sources.append({
                "rank": rank,
                "parent_id": doc.metadata.get("parent_id"),
                "section": sec,
                "section_title": doc.metadata.get("section_title"),
                "rerank_score": round(float(doc.metadata.get("rerank_score", 0.0)), 4),
                "snippet": doc.page_content[:200] + "...",
            })

        expected = r.expected
        section_hit = bool(expected) and any(s in expected for s in retrieved_sections)
        hit_in_top_1 = bool(expected) and len(retrieved_sections) > 0 and retrieved_sections[0] in expected
        reranked_sections = _sections(r.reranked)
        hit_in_reranked = bool(expected) and any(s in expected for s in reranked_sections)

        return SingleRetrievalResult(
            eval_id=r.eval_id,
            query=r.query,
            target_section=",".join(expected),
            expected_route=r.expected_route,
            actual_route=r.routing.route.value,
            route_reasoning=r.routing.reasoning,
            extracted_sections=r.routing.extracted_sections,
            abstained=abstention.should_abstain,
            section_hit=section_hit,
            hit_in_top_1=hit_in_top_1,
            retrieved_sources=retrieved_sources,
            execution_time_sec=r.elapsed,
            answerable=r.answerable,
            hit_in_reranked=hit_in_reranked,
            top_score=abstention.confidence_score,
            score_margin=abstention.score_margin,
            context_size=len(context_docs),
            reranked_sections=reranked_sections,
        )

    def evaluate_sample(self, entry: Dict[str, Any], idx: int) -> SingleRetrievalResult:
        return self.to_result(self.retrieve(entry, idx))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def summarize(results: List[SingleRetrievalResult]) -> Dict[str, Any]:
    total = len(results)
    ans = [r for r in results if r.answerable is not False]          # answerable + unlabeled
    ans_with_target = [r for r in ans if r.target_section]
    oos = [r for r in results if r.answerable is False]

    return {
        # kept from the original report (hit rates now over answerable questions only)
        "section_hit_rate": _rate(sum(r.section_hit for r in ans_with_target), len(ans_with_target)),
        "top_1_hit_rate": _rate(sum(r.hit_in_top_1 for r in ans_with_target), len(ans_with_target)),
        "routing_accuracy": _rate(sum(r.actual_route == r.expected_route for r in results), total),
        "abstention_rate": _rate(sum(r.abstained for r in results), total),
        # new
        "hit_in_reranked_rate": _rate(sum(r.hit_in_reranked for r in ans_with_target), len(ans_with_target)),
        "false_abstention_rate": _rate(sum(r.abstained for r in ans), len(ans)),
        "out_of_scope_count": len(oos),
        "out_of_scope_correct_abstention_rate": _rate(sum(r.abstained for r in oos), len(oos)),
        "avg_context_docs": round(sum(r.context_size for r in results if not r.abstained)
                                  / max(1, sum(1 for r in results if not r.abstained)), 2),
    }


# ---------------------------------------------------------------------------
# Sweeps (offline replays of cached retrieval results)
# ---------------------------------------------------------------------------
def _case_ok(answerable: Optional[bool], expected: List[str], abstained: bool, context: List[Any]) -> Optional[bool]:
    if answerable is None:
        return None
    if answerable is False:
        return abstained
    if abstained:
        return False
    return (not expected) or any(s in expected for s in _sections(context))


def sweep_abstention(tuner: RetrievalTuner, cached: List[_Retrieved], top_n: int = 10) -> Dict[str, Any]:
    labelled = [r for r in cached if r.answerable is not None]
    n_oos = sum(1 for r in labelled if r.answerable is False)
    base = AbstentionGuard(**tuner.guard_kwargs)

    tops = sorted({float(r.reranked[0].metadata.get("rerank_score", 0.0)) for r in labelled if r.reranked})
    if n_oos == 0:
        print("\n⚠️  No OUT_OF_SCOPE questions in the dataset: abs_floor / min_margin can't be tuned "
              "(any 'never abstain' setting would win). Sweeping only keep_ratio / min_context_docs "
              "with the current floor/margin; add ~10-15 out-of-scope questions to tune abstention.")
        floors = [base.abs_floor]
        margins = [base.min_margin]
    else:
        fixed = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
        step = max(1, len(tops) // 12)
        floors = sorted(set(fixed + [round(t * 0.999, 5) for t in tops[::step]]))
        margins = [1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    ratios = [0.1, 0.2, 0.3, 0.5]
    min_docs_grid = [1, 2, 3]

    def evaluate(floor, margin, ratio, min_docs) -> Dict[str, Any]:
        kwargs = dict(tuner.guard_kwargs)
        kwargs.update(abs_floor=floor, min_margin=margin, relative_keep_ratio=ratio, min_context_docs=min_docs)
        guard = AbstentionGuard(**kwargs)
        ok = ans_ok = ans_n = oos_ok = ctx_total = ctx_n = 0
        for r in labelled:
            abstention, context = tuner.decide(r, guard=guard)
            res = _case_ok(r.answerable, r.expected, abstention.should_abstain, context)
            ok += bool(res)
            if r.answerable:
                ans_n += 1
                ans_ok += bool(res)
            else:
                oos_ok += bool(res)
            if not abstention.should_abstain:
                ctx_total += len(context)
                ctx_n += 1
        return {
            "abs_floor": floor, "min_margin": margin, "keep_ratio": ratio, "min_context_docs": min_docs,
            "accuracy": round(ok / len(labelled), 4) if labelled else 0.0,
            "answerable_ok": round(ans_ok / ans_n, 4) if ans_n else None,
            "out_of_scope_ok": round(oos_ok / n_oos, 4) if n_oos else None,
            "avg_context_docs": round(ctx_total / ctx_n, 2) if ctx_n else 0.0,
        }

    rows = [evaluate(f, m, ra, md) for f in floors for m in margins for ra in ratios for md in min_docs_grid]
    rows.sort(key=lambda x: (-x["accuracy"], x["avg_context_docs"]))
    current = evaluate(base.abs_floor, base.min_margin, base.relative_keep_ratio, base.min_context_docs)

    print("\n🎛️  ABSTENTION / CONTEXT-SELECTION SWEEP "
          f"({len(labelled)} labelled questions, {n_oos} out-of-scope)")
    print("   accuracy = answerable: answered AND correct section in context | out-of-scope: abstained")
    print(f"   {'acc':>6} {'ans_ok':>7} {'oos_ok':>7} {'ctx':>5} {'floor':>8} {'margin':>7} {'ratio':>6} {'min_docs':>8}")

    def fmt(x):
        return (f"   {x['accuracy']:>6.3f} {('%.3f' % x['answerable_ok']) if x['answerable_ok'] is not None else '-':>7} "
                f"{('%.3f' % x['out_of_scope_ok']) if x['out_of_scope_ok'] is not None else '-':>7} "
                f"{x['avg_context_docs']:>5.1f} {x['abs_floor']:>8.4f} {x['min_margin']:>7.2f} "
                f"{x['keep_ratio']:>6.2f} {x['min_context_docs']:>8}")

    shown, seen = 0, set()
    for x in rows:
        key = (x["accuracy"], x["avg_context_docs"], x["abs_floor"], x["min_margin"], x["keep_ratio"], x["min_context_docs"])
        if key in seen:
            continue
        seen.add(key)
        print(fmt(x))
        shown += 1
        if shown >= top_n:
            break
    print("   current:")
    print(fmt(current))
    print("   Prefer a setting from the middle of a plateau of equal rows over the single sharpest one -")
    print("   with a small dataset the exact optimum is mostly noise.")
    return {"best": rows[:top_n], "current": current, "labelled": len(labelled), "out_of_scope": n_oos}


def sweep_retrieval(tuner: RetrievalTuner, cached: List[_Retrieved]) -> Dict[str, Any]:
    targets = [r for r in cached if r.answerable is not False and r.expected and r.ranked_all is not None]
    if not targets:
        print("\n(retrieval sweep skipped: no answerable questions with a 'section' label)")
        return {}

    # Sanity check: the offline replay must reproduce what the real reranker returned.
    mismatches = sum(
        1 for r in cached if r.ranked_all is not None and
        _sections(diversify(r.ranked_all, tuner.top_k_reranked, tuner.max_per_section, tuner.fetch_k)) != _sections(r.reranked)
    )
    if mismatches:
        print(f"\n⚠️  Replay differs from the real reranker output on {mismatches} question(s); "
              "retrieval-sweep numbers may be off (did rerank_with_diversity change?).")

    rows = []
    for top_k in (4, 6, 8, 10, 12):
        for per_sec in (1, 2, 3):
            hits = size = 0
            for r in targets:
                sel = diversify(r.ranked_all, top_k, per_sec, tuner.fetch_k)
                hits += any(s in r.expected for s in _sections(sel))
                size += len(sel)
            rows.append({"top_k_reranked": top_k, "max_per_section": per_sec,
                         "hit_in_reranked": round(hits / len(targets), 4),
                         "avg_docs": round(size / len(targets), 2)})
    rows.sort(key=lambda x: (-x["hit_in_reranked"], x["avg_docs"]))

    print(f"\n🎛️  RERANK-SELECTION SWEEP ({len(targets)} answerable questions; before abstention/selection)")
    print(f"   {'hit':>6} {'avg_docs':>9} {'top_k_reranked':>15} {'max_per_section':>16}")
    for x in rows[:8]:
        print(f"   {x['hit_in_reranked']:>6.3f} {x['avg_docs']:>9.1f} {x['top_k_reranked']:>15} {x['max_per_section']:>16}")
    return {"best": rows[:8], "replay_mismatches": mismatches}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tune & Diagnose CCPA Retrieval Pipeline")
    parser.add_argument("--dataset", type=str, default="eval/golden_dataset.json", help="Path to golden dataset")
    parser.add_argument("--output", type=str, default="eval/retrieval_tuning_results.json", help="Export path for JSON results")
    parser.add_argument("--max-samples", type=int, default=0, help="Samples to evaluate (0 = all)")
    parser.add_argument("--in-order", action="store_true", help="With --max-samples, take the first N instead of a random N")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-diff", action="store_true", help="Also evaluate VERSION_DIFF rows")
    parser.add_argument("--min-rerank-score", type=float, default=0.25, help="'Trusted outright' score (pipeline: 0.35 / 0.22)")
    parser.add_argument("--top-k-candidates", type=int, default=20, help="Initial candidate count from hybrid search")
    parser.add_argument("--top-k-reranked", type=int, default=6, help="Final reranked limit (pipeline default: 6)")
    parser.add_argument("--max-per-section", type=int, default=2, help="Diversity cap per parent section")
    parser.add_argument("--abs-floor", type=float, default=0.01)
    parser.add_argument("--min-margin", type=float, default=1.5)
    parser.add_argument("--keep-ratio", type=float, default=0.2)
    parser.add_argument("--min-context-docs", type=int, default=2)
    parser.add_argument("--sweep", action="store_true", help="Also search for good parameters (offline replay)")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Golden dataset not found at {args.dataset}")

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    skipped_diff = 0
    if not args.include_diff:
        kept = [e for e in dataset if str(e.get("query_type", "")).upper() != "VERSION_DIFF"]
        skipped_diff = len(dataset) - len(kept)
        dataset = kept

    samples = dataset
    if args.max_samples:
        if args.in_order:
            samples = dataset[: args.max_samples]
        else:
            samples = random.Random(args.seed).sample(dataset, min(args.max_samples, len(dataset)))

    tuner = RetrievalTuner(
        min_rerank_score=args.min_rerank_score,
        top_k_candidates=args.top_k_candidates,
        top_k_reranked=args.top_k_reranked,
        max_per_section=args.max_per_section,
        abs_floor=args.abs_floor,
        min_margin=args.min_margin,
        keep_ratio=args.keep_ratio,
        min_context_docs=args.min_context_docs,
        keep_full_ranking=args.sweep,
    )

    g = tuner.abstention_guard
    print(f"\n🔍 Tuning Retrieval Pipeline over {len(samples)} samples"
          + (f" (skipped {skipped_diff} VERSION_DIFF)" if skipped_diff else "") + "...")
    print(f"⚙️ Config: trusted={args.min_rerank_score} | floor={g.abs_floor} | margin={g.min_margin} | "
          f"keep_ratio={g.relative_keep_ratio} | min_ctx={g.min_context_docs} | candidates={args.top_k_candidates} | "
          f"top_k_reranked={args.top_k_reranked} | max_per_sec={args.max_per_section}\n" + "=" * 80)

    results: List[SingleRetrievalResult] = []
    cached: List[_Retrieved] = []

    for idx, entry in enumerate(samples, 1):
        r = tuner.retrieve(entry, idx)
        cached.append(r)
        res = tuner.to_result(r)
        results.append(res)

        if res.answerable is False:
            status = "✅ CORRECT ABSTAIN" if res.abstained else "❌ SHOULD HAVE ABSTAINED"
        elif res.section_hit:
            status = "✅ HIT"
        elif res.abstained:
            status = "⛔ ABSTAINED"
        elif res.hit_in_reranked:
            status = "✂️ CUT (in reranked, dropped by selection)"
        else:
            status = "❌ MISS"
        top1_flag = " (Top-1)" if res.hit_in_top_1 else ""
        margin_txt = f"{res.score_margin:.1f}x" if res.score_margin is not None else "-"
        print(f"[{idx:03d}/{len(samples):03d}] {res.eval_id:<14} | Target §{res.target_section or '-':<7} | "
              f"Route: {res.actual_route:<13} | top={res.top_score:.4f} margin={margin_txt:<6} | {status}{top1_flag}")

        if res.retrieved_sources:
            for src in res.retrieved_sources:
                print(f"    └─ Rank {src['rank']} | §{src['section']} | Score: {src['rerank_score']:.4f} | {src['parent_id']}")
        else:
            print("    └─ [No context sources selected — Abstained]")
        print("-" * 80)

    metrics = summarize(results)
    total = len(results)
    summary = {
        "config": {
            "min_rerank_score": args.min_rerank_score,
            "abs_floor": g.abs_floor,
            "min_margin": g.min_margin,
            "keep_ratio": g.relative_keep_ratio,
            "min_context_docs": g.min_context_docs,
            "top_k_candidates": args.top_k_candidates,
            "top_k_reranked": args.top_k_reranked,
            "max_per_section": args.max_per_section,
            "samples_evaluated": total,
        },
        "metrics": metrics,
    }

    print("\n📊 RETRIEVAL TUNING SUMMARY REPORT")
    print("=" * 80)
    print(f"Samples Evaluated        : {total}  (out-of-scope: {metrics['out_of_scope_count']})")
    print(f"Section Hit Rate         : {metrics['section_hit_rate'] * 100:.2f}%   (correct section in final context)")
    print(f"  - in reranked list     : {metrics['hit_in_reranked_rate'] * 100:.2f}%   (before abstention/selection)")
    print(f"Top-1 Section Precision  : {metrics['top_1_hit_rate'] * 100:.2f}%")
    print(f"False Abstention Rate    : {metrics['false_abstention_rate'] * 100:.2f}%   (answerable questions refused)")
    if metrics["out_of_scope_count"]:
        print(f"Out-of-scope Abstained   : {metrics['out_of_scope_correct_abstention_rate'] * 100:.2f}%")
    print(f"Routing Accuracy         : {metrics['routing_accuracy'] * 100:.2f}%")
    print(f"Avg Context Docs         : {metrics['avg_context_docs']}")
    print("=" * 80)

    export_payload: Dict[str, Any] = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    if args.sweep:
        export_payload["sweep"] = {
            "abstention": sweep_abstention(tuner, cached),
            "retrieval": sweep_retrieval(tuner, cached),
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, default=str)

    print(f"\n📁 Full diagnostic result exported to: {args.output}\n")


if __name__ == "__main__":
    main()