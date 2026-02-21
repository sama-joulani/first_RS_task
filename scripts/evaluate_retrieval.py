"""Mini evaluation script for hybrid retrieval on RealSoft dataset."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

from qdrant_client import QdrantClient

from app.config import get_settings
from app.core.embedding_service import EmbeddingService
from app.core.hybrid_retrieval import HybridRetriever
from app.core.rag_pipeline import RAGPipeline
from app.core.prompt_manager import PromptManager
from app.core.llm_service import LLMService
from scripts.evaluation_data import REALSOFT_EVAL_DATASET, EvalQuestion


@dataclass
class EvalResult:
    """Result for a single evaluation question."""
    query: str
    language: str  # 'ar' or 'en'
    expected_pages: List[int]
    retrieved_pages: List[int]
    found_in_top_k: bool
    top_k_position: Optional[int]  # Position of first correct result (1-indexed)
    category: str
    search_method: str  # 'dense' or 'hybrid'
    fusion_method: Optional[str]
    response_time_ms: float


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    total_questions: int
    method: str  # 'dense', 'hybrid_rrf', or 'hybrid_weighted'
    top_k: int
    
    # Metrics
    accuracy_at_k: float  # % of questions with correct answer in top-k
    mean_reciprocal_rank: float  # MRR
    avg_response_time_ms: float
    
    # Breakdown by category
    category_results: Dict[str, Dict]
    
    # Detailed results
    results: List[Dict]


class RetrievalEvaluator:
    """Evaluator for retrieval performance."""
    
    def __init__(self, use_hybrid: bool = False, fusion_method: Optional[str] = None, top_k: int = 5):
        self.settings = get_settings()
        self.use_hybrid = use_hybrid
        self.fusion_method = fusion_method
        self.top_k = top_k
        
        # Initialize services
        self.embedding_service = EmbeddingService()
        self.qdrant_client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port
        )
        
        if use_hybrid:
            self.retriever = HybridRetriever(
                embedding_service=self.embedding_service,
                qdrant_client=self.qdrant_client,
                collection_name=self.settings.qdrant_collection,
            )
            # Override fusion method if specified
            if fusion_method:
                from app.core.retrieval_config import get_retrieval_settings
                retrieval_settings = get_retrieval_settings()
                retrieval_settings.fusion_method = fusion_method
        else:
            # Standard dense retrieval via RAGPipeline
            llm_service = LLMService()
            prompt_manager = PromptManager()
            self.pipeline = RAGPipeline(
                embedding_service=self.embedding_service,
                llm_service=llm_service,
                prompt_manager=prompt_manager,
                qdrant_client=self.qdrant_client,
                collection_name=self.settings.qdrant_collection,
            )
    
    def evaluate_question(self, question: EvalQuestion, language: str = 'en') -> EvalResult:
        """Evaluate a single question."""
        query = question.query_en if language == 'en' else question.query_ar
        
        start_time = time.time()
        
        # Perform search
        if self.use_hybrid:
            results, debug_info = self.retriever.search(query, use_cache=True)
            search_method = "hybrid"
            fusion = debug_info.get("fusion_method")
        else:
            response = self.pipeline.search(query, top_k=self.top_k)
            results = response.results
            search_method = "dense"
            fusion = None
        
        response_time_ms = (time.time() - start_time) * 1000
        
        # Extract retrieved pages
        retrieved_pages = [r.page for r in results[:self.top_k]]
        
        # Check if any expected page is in retrieved pages
        found_pages = set(retrieved_pages) & set(question.expected_pages)
        found_in_top_k = len(found_pages) > 0
        
        # Find position of first correct result
        top_k_position = None
        for i, page in enumerate(retrieved_pages, 1):
            if page in question.expected_pages:
                top_k_position = i
                break
        
        return EvalResult(
            query=query,
            language=language,
            expected_pages=question.expected_pages,
            retrieved_pages=retrieved_pages,
            found_in_top_k=found_in_top_k,
            top_k_position=top_k_position,
            category=question.category,
            search_method=search_method,
            fusion_method=fusion,
            response_time_ms=response_time_ms,
        )
    
    def run_evaluation(self, languages: List[str] = ['en', 'ar']) -> EvaluationReport:
        """Run full evaluation on dataset."""
        all_results: List[EvalResult] = []
        
        print(f"\nRunning evaluation: {'Hybrid' if self.use_hybrid else 'Dense'} (top_k={self.top_k})")
        if self.fusion_method:
            print(f"Fusion method: {self.fusion_method}")
        print("=" * 60)
        
        for question in REALSOFT_EVAL_DATASET:
            for lang in languages:
                result = self.evaluate_question(question, lang)
                all_results.append(result)
                
                # Print progress
                status = "✓" if result.found_in_top_k else "✗"
                pos = f"pos={result.top_k_position}" if result.top_k_position else "not found"
                print(f"{status} [{lang.upper()}] {result.query[:50]}... ({pos})")
        
        # Calculate metrics
        correct = sum(1 for r in all_results if r.found_in_top_k)
        accuracy = correct / len(all_results) if all_results else 0
        
        # MRR calculation
        rr_sum = sum(1/r.top_k_position for r in all_results if r.top_k_position)
        mrr = rr_sum / len(all_results) if all_results else 0
        
        avg_time = sum(r.response_time_ms for r in all_results) / len(all_results) if all_results else 0
        
        # Category breakdown
        from collections import defaultdict
        category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for r in all_results:
            category_stats[r.category]['total'] += 1
            if r.found_in_top_k:
                category_stats[r.category]['correct'] += 1
        
        category_results = {
            cat: {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            }
            for cat, stats in category_stats.items()
        }
        
        method_name = 'dense'
        if self.use_hybrid:
            method_name = f'hybrid_{self.fusion_method or "rrf"}'
        
        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_questions=len(all_results),
            method=method_name,
            top_k=self.top_k,
            accuracy_at_k=accuracy,
            mean_reciprocal_rank=mrr,
            avg_response_time_ms=avg_time,
            category_results=category_results,
            results=[asdict(r) for r in all_results]
        )
    
    def print_report(self, report: EvaluationReport):
        """Print evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"Method: {report.method}")
        print(f"Top-K: {report.top_k}")
        print(f"Total Questions: {report.total_questions}")
        print(f"Timestamp: {report.timestamp}")
        print("-" * 60)
        print(f"Accuracy @ K: {report.accuracy_at_k:.2%}")
        print(f"Mean Reciprocal Rank (MRR): {report.mean_reciprocal_rank:.4f}")
        print(f"Avg Response Time: {report.avg_response_time_ms:.2f} ms")
        print("-" * 60)
        print("Results by Category:")
        for cat, stats in sorted(report.category_results.items()):
            print(f"  {cat:20s}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.2%}")
        print("=" * 60)
    
    def save_report(self, report: EvaluationReport, filename: Optional[str] = None):
        """Save report to JSON file."""
        if filename is None:
            filename = f"eval_report_{report.method}_k{report.top_k}_{datetime.now():%Y%m%d_%H%M%S}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        
        print(f"\nReport saved to: {filepath}")
        return filepath


def compare_methods():
    """Compare dense vs hybrid retrieval."""
    print("\n" + "=" * 70)
    print("RETRIEVAL METHOD COMPARISON")
    print("=" * 70)
    
    results = {}
    
    # 1. Dense only
    print("\n>>> Testing DENSE retrieval...")
    evaluator_dense = RetrievalEvaluator(use_hybrid=False, top_k=5)
    report_dense = evaluator_dense.run_evaluation(languages=['en', 'ar'])
    evaluator_dense.print_report(report_dense)
    results['dense'] = report_dense
    evaluator_dense.save_report(report_dense)
    
    # 2. Hybrid with RRF
    print("\n>>> Testing HYBRID with RRF...")
    evaluator_rrf = RetrievalEvaluator(use_hybrid=True, fusion_method='rrf', top_k=5)
    report_rrf = evaluator_rrf.run_evaluation(languages=['en', 'ar'])
    evaluator_rrf.print_report(report_rrf)
    results['hybrid_rrf'] = report_rrf
    evaluator_rrf.save_report(report_rrf)
    
    # 3. Hybrid with Weighted
    print("\n>>> Testing HYBRID with WEIGHTED...")
    evaluator_weighted = RetrievalEvaluator(use_hybrid=True, fusion_method='weighted', top_k=5)
    report_weighted = evaluator_weighted.run_evaluation(languages=['en', 'ar'])
    evaluator_weighted.print_report(report_weighted)
    results['hybrid_weighted'] = report_weighted
    evaluator_weighted.save_report(report_weighted)
    
    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Accuracy':>12} {'MRR':>12} {'Avg Time':>12}")
    print("-" * 70)
    for name, report in results.items():
        print(f"{name:<20} {report.accuracy_at_k:>11.2%} {report.mean_reciprocal_rank:>12.4f} {report.avg_response_time_ms:>10.2f}ms")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument("--method", choices=["dense", "hybrid_rrf", "hybrid_weighted", "compare"], 
                       default="compare", help="Retrieval method to test")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K value for evaluation")
    parser.add_argument("--lang", nargs="+", default=["en", "ar"], choices=["en", "ar"],
                       help="Languages to test")
    
    args = parser.parse_args()
    
    if args.method == "compare":
        compare_methods()
    else:
        use_hybrid = "hybrid" in args.method
        fusion = args.method.split("_")[1] if use_hybrid else None
        
        evaluator = RetrievalEvaluator(
            use_hybrid=use_hybrid,
            fusion_method=fusion,
            top_k=args.top_k
        )
        report = evaluator.run_evaluation(languages=args.lang)
        evaluator.print_report(report)
        evaluator.save_report(report)
