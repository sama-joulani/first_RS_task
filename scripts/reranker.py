"""Optional reranker for improving hybrid retrieval results."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
from dataclasses import dataclass

import numpy as np

from app.core.embedding_service import EmbeddingService


@dataclass
class RerankedResult:
    """Result after reranking."""
    original_rank: int
    new_rank: int
    chunk_text: str
    score: float
    document_id: str
    title: str
    author: str
    page: int
    section: str
    chunk_index: int
    tags: List[str]
    dense_score: float = 0.0
    keyword_score: float = 0.0


class CrossEncoderReranker:
    """
    Simple cross-encoder style reranker using cosine similarity
    between query and document embeddings.
    
    Note: For production, consider using a dedicated cross-encoder model
    like BAAI/bge-reranker-base or similar.
    """
    
    def __init__(self, embedding_service: EmbeddingService = None):
        self.embedding_service = embedding_service or EmbeddingService()
    
    def rerank(
        self,
        query: str,
        results: List,
        top_k: int = 5
    ) -> Tuple[List[RerankedResult], dict]:
        """
        Rerank results based on query-document relevance.
        
        Returns:
            Tuple of (reranked_results, debug_info)
        """
        if not results:
            return [], {"method": "none", "results_count": 0}
        
        # Get query embedding
        query_embedding = np.array(self.embedding_service.embed_text(query))
        
        # Score each result
        scored_results = []
        for i, result in enumerate(results):
            # Get document embedding (re-embed for freshness)
            doc_embedding = np.array(self.embedding_service.embed_text(result.chunk_text))
            
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            # Combine with original score (weighted average)
            combined_score = 0.6 * float(similarity) + 0.4 * result.score
            
            scored_results.append({
                'original_rank': i + 1,
                'result': result,
                'rerank_score': combined_score,
                'semantic_similarity': float(similarity)
            })
        
        # Sort by rerank score
        scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Create reranked results
        reranked = []
        for new_rank, item in enumerate(scored_results[:top_k], 1):
            r = item['result']
            reranked.append(RerankedResult(
                original_rank=item['original_rank'],
                new_rank=new_rank,
                chunk_text=r.chunk_text,
                score=item['rerank_score'],
                document_id=r.document_id,
                title=r.title,
                author=r.author,
                page=r.page,
                section=r.section,
                chunk_index=r.chunk_index,
                tags=r.tags,
                dense_score=getattr(r, 'dense_score', 0.0),
                keyword_score=getattr(r, 'keyword_score', 0.0)
            ))
        
        debug_info = {
            "method": "cross_encoder_cosine",
            "results_count": len(results),
            "reranked_count": len(reranked),
            "avg_score_change": sum(
                abs(r.score - results[r.original_rank - 1].score) 
                for r in reranked
            ) / len(reranked) if reranked else 0
        }
        
        return reranked, debug_info


class ScoreBasedReranker:
    """
    Simple reranker that adjusts scores based on additional signals
    like exact match boosts and position penalties.
    """
    
    def __init__(self):
        pass
    
    def rerank(
        self,
        query: str,
        results: List,
        top_k: int = 5
    ) -> Tuple[List[RerankedResult], dict]:
        """
        Rerank using heuristics:
        - Exact match boost
        - Position diversity bonus
        """
        if not results:
            return [], {"method": "none", "results_count": 0}
        
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        scored_results = []
        seen_pages = set()
        
        for i, result in enumerate(results):
            r = result
            base_score = r.score
            
            # Exact phrase match boost
            exact_match_boost = 0.0
            if query_lower in r.chunk_text.lower():
                exact_match_boost = 0.15
            
            # Term overlap boost
            text_terms = set(r.chunk_text.lower().split())
            overlap = len(query_terms & text_terms) / len(query_terms) if query_terms else 0
            term_boost = overlap * 0.1
            
            # Diversity bonus (prefer results from different pages)
            diversity_bonus = 0.0
            if r.page not in seen_pages:
                diversity_bonus = 0.05
                seen_pages.add(r.page)
            
            final_score = base_score + exact_match_boost + term_boost + diversity_bonus
            
            scored_results.append({
                'original_rank': i + 1,
                'result': r,
                'rerank_score': final_score,
                'exact_match': exact_match_boost > 0,
                'term_overlap': overlap
            })
        
        # Sort by rerank score
        scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Create reranked results
        reranked = []
        for new_rank, item in enumerate(scored_results[:top_k], 1):
            r = item['result']
            reranked.append(RerankedResult(
                original_rank=item['original_rank'],
                new_rank=new_rank,
                chunk_text=r.chunk_text,
                score=item['rerank_score'],
                document_id=r.document_id,
                title=r.title,
                author=r.author,
                page=r.page,
                section=r.section,
                chunk_index=r.chunk_index,
                tags=r.tags,
                dense_score=getattr(r, 'dense_score', 0.0),
                keyword_score=getattr(r, 'keyword_score', 0.0)
            ))
        
        # Count rank changes
        rank_changes = sum(
            1 for r in reranked 
            if r.original_rank != r.new_rank
        )
        
        debug_info = {
            "method": "heuristic_score_based",
            "results_count": len(results),
            "reranked_count": len(reranked),
            "rank_changes": rank_changes,
            "exact_matches_boosted": sum(1 for item in scored_results if item['exact_match'])
        }
        
        return reranked, debug_info


def demonstrate_reranker():
    """Demonstrate reranker with sample query."""
    from app.config import get_settings
    from app.core.hybrid_retrieval import HybridRetriever
    from qdrant_client import QdrantClient
    
    settings = get_settings()
    
    # Initialize services
    embedding_service = EmbeddingService()
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    
    retriever = HybridRetriever(
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
        collection_name=settings.qdrant_collection,
    )
    
    # Test query
    query = "What is RealSoft's vision and mission?"
    
    print("=" * 70)
    print("RERANKER DEMONSTRATION")
    print("=" * 70)
    print(f"Query: {query}")
    print()
    
    # Get hybrid results
    results, debug_info = retriever.search(query)
    
    print("BEFORE RERANKING (Hybrid RRF):")
    print("-" * 70)
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. [{r.score:.4f}] Page {r.page}: {r.chunk_text[:80]}...")
    
    # Apply reranking
    print("\n" + "=" * 70)
    print("AFTER RERANKING (Cross-Encoder):")
    print("-" * 70)
    
    reranker = CrossEncoderReranker(embedding_service)
    reranked, rerank_info = reranker.rerank(query, results, top_k=5)
    
    for r in reranked:
        change = f"(was #{r.original_rank})" if r.original_rank != r.new_rank else "(same)"
        print(f"{r.new_rank}. [{r.score:.4f}] Page {r.page} {change}: {r.chunk_text[:80]}...")
    
    print(f"\nReranker Info: {rerank_info}")
    
    # Also show heuristic reranker
    print("\n" + "=" * 70)
    print("AFTER RERANKING (Heuristic):")
    print("-" * 70)
    
    heuristic_reranker = ScoreBasedReranker()
    reranked2, rerank_info2 = heuristic_reranker.rerank(query, results, top_k=5)
    
    for r in reranked2:
        change = f"(was #{r.original_rank})" if r.original_rank != r.new_rank else "(same)"
        print(f"{r.new_rank}. [{r.score:.4f}] Page {r.page} {change}: {r.chunk_text[:80]}...")
    
    print(f"\nReranker Info: {rerank_info2}")


if __name__ == "__main__":
    demonstrate_reranker()
