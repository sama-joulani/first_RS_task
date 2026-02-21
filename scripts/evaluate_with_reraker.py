"""Evaluation comparing before/after reranking on the mini eval set."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime

from scripts.evaluate_retrieval import RetrievalEvaluator, EvalResult, EvaluationReport
from scripts.evaluation_data import REALSOFT_EVAL_DATASET
from scripts.reranker import CrossEncoderReranker, ScoreBasedReranker


@dataclass
class RerankerEvalResult:
    """Result comparing before and after reranking."""
    query: str
    language: str
    expected_pages: List[int]
    
    # Before reranking
    before_found: bool
    before_position: int
    
    # After reranking
    after_found: bool
    after_position: int
    
    # Improvement metrics
    position_improvement: int  # Positive = better (lower rank)
    found_improvement: bool  # True if went from not found to found
    
    category: str
    reranker_method: str


class RerankerEvaluator:
    """Evaluate impact of reranking."""
    
    def __init__(self, base_retriever: RetrievalEvaluator, reranker):
        self.base_retriever = base_retriever
        self.reranker = reranker
    
    def evaluate_question(self, question, language: str = 'en') -> RerankerEvalResult:
        """Evaluate a single question before and after reranking."""
        query = question.query_en if language == 'en' else question.query_ar
        
        # Get base results
        if self.base_retriever.use_hybrid:
            results, _ = self.base_retriever.retriever.search(query, use_cache=True)
        else:
            response = self.base_retriever.pipeline.search(query, top_k=10)
            results = response.results
        
        # Evaluate before
        before_pages = [r.page for r in results[:5]]
        before_found = bool(set(before_pages) & set(question.expected_pages))
        before_position = None
        for i, page in enumerate(before_pages, 1):
            if page in question.expected_pages:
                before_position = i
                break
        
        # Apply reranking
        reranked, _ = self.reranker.rerank(query, results, top_k=5)
        
        # Evaluate after
        after_pages = [r.page for r in reranked]
        after_found = bool(set(after_pages) & set(question.expected_pages))
        after_position = None
        for i, r in enumerate(reranked, 1):
            if r.page in question.expected_pages:
                after_position = i
                break
        
        # Calculate improvement
        position_improvement = 0
        if before_position and after_position:
            position_improvement = before_position - after_position  # Positive = better
        elif not before_position and after_position:
            position_improvement = 5 - after_position  # Went from not found to found
        
        found_improvement = (not before_found) and after_found
        
        return RerankerEvalResult(
            query=query,
            language=language,
            expected_pages=question.expected_pages,
            before_found=before_found,
            before_position=before_position,
            after_found=after_found,
            after_position=after_position,
            position_improvement=position_improvement,
            found_improvement=found_improvement,
            category=question.category,
            reranker_method=self.reranker.__class__.__name__
        )
    
    def run_evaluation(self, languages: List[str] = ['en', 'ar']) -> Dict:
        """Run full evaluation with reranking comparison."""
        results: List[RerankerEvalResult] = []
        
        print(f"\nEvaluating with Reranker: {self.reranker.__class__.__name__}")
        print("=" * 70)
        
        for question in REALSOFT_EVAL_DATASET:
            for lang in languages:
                result = self.evaluate_question(question, lang)
                results.append(result)
                
                # Print progress
                status = "✓" if result.after_found else "✗"
                improvement = ""
                if result.found_improvement:
                    improvement = " [NEW FIND!]"
                elif result.position_improvement > 0:
                    improvement = f" [↑{result.position_improvement}]"
                
                print(f"{status} [{lang.upper()}] {result.query[:45]}...{improvement}")
        
        # Calculate metrics
        total = len(results)
        before_correct = sum(1 for r in results if r.before_found)
        after_correct = sum(1 for r in results if r.after_found)
        newly_found = sum(1 for r in results if r.found_improvement)
        worsened = sum(1 for r in results if r.position_improvement < 0)
        
        # MRR calculation
        before_rr = sum(1/r.before_position for r in results if r.before_position)
        after_rr = sum(1/r.after_position for r in results if r.after_position)
        before_mrr = before_rr / total if total > 0 else 0
        after_mrr = after_rr / total if total > 0 else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "reranker": self.reranker.__class__.__name__,
            "base_method": "hybrid" if self.base_retriever.use_hybrid else "dense",
            "total_questions": total,
            "metrics": {
                "before": {
                    "accuracy": before_correct / total,
                    "mrr": before_mrr,
                    "correct_count": before_correct
                },
                "after": {
                    "accuracy": after_correct / total,
                    "mrr": after_mrr,
                    "correct_count": after_correct
                },
                "improvements": {
                    "newly_found": newly_found,
                    "worsened": worsened,
                    "accuracy_gain": (after_correct - before_correct) / total,
                    "mrr_gain": after_mrr - before_mrr
                }
            },
            "results": [asdict(r) for r in results]
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print comparison report."""
        print("\n" + "=" * 70)
        print("RERANKER COMPARISON REPORT")
        print("=" * 70)
        print(f"Reranker: {report['reranker']}")
        print(f"Base Method: {report['base_method']}")
        print(f"Total Questions: {report['total_questions']}")
        print("-" * 70)
        
        m = report['metrics']
        print(f"{'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}")
        print("-" * 70)
        print(f"{'Accuracy':<25} {m['before']['accuracy']:>11.2%} {m['after']['accuracy']:>11.2%} {m['improvements']['accuracy_gain']:>+11.2%}")
        print(f"{'MRR':<25} {m['before']['mrr']:>12.4f} {m['after']['mrr']:>12.4f} {m['improvements']['mrr_gain']:>+12.4f}")
        print(f"{'Correct Count':<25} {m['before']['correct_count']:>12} {m['after']['correct_count']:>12} {m['improvements']['newly_found']:>+12}")
        print("-" * 70)
        print(f"Newly Found: {m['improvements']['newly_found']}")
        print(f"Worsened: {m['improvements']['worsened']}")
        print("=" * 70)


def compare_rerankers():
    """Compare different rerankers on the same base results."""
    from app.core.embedding_service import EmbeddingService
    
    print("\n" + "=" * 70)
    print("RERANKER COMPARISON")
    print("=" * 70)
    
    # Base retriever (hybrid RRF)
    print("\n>>> Initializing base retriever (Hybrid RRF)...")
    base = RetrievalEvaluator(use_hybrid=True, fusion_method='rrf', top_k=10)
    
    # Test both rerankers
    rerankers = [
        CrossEncoderReranker(embedding_service=base.embedding_service),
        ScoreBasedReranker()
    ]
    
    all_reports = []
    
    for reranker in rerankers:
        evaluator = RerankerEvaluator(base, reranker)
        report = evaluator.run_evaluation(languages=['en', 'ar'])
        evaluator.print_report(report)
        all_reports.append(report)
        
        # Save report
        filename = f"reranker_eval_{reranker.__class__.__name__}_{datetime.now():%Y%m%d_%H%M%S}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {filepath}\n")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: RERANKER COMPARISON")
    print("=" * 70)
    print(f"{'Reranker':<25} {'Acc Before':>12} {'Acc After':>12} {'Acc Gain':>12}")
    print("-" * 70)
    for r in all_reports:
        m = r['metrics']
        print(f"{r['reranker']:<25} {m['before']['accuracy']:>11.2%} {m['after']['accuracy']:>11.2%} {m['improvements']['accuracy_gain']:>+11.2%}")
    print("=" * 70)


if __name__ == "__main__":
    compare_rerankers()
