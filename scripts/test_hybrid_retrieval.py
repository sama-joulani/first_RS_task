"""Test script for hybrid retrieval functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings
from app.core.embedding_service import EmbeddingService
from app.core.hybrid_retrieval import HybridRetriever
from app.core.retrieval_config import get_retrieval_settings
from qdrant_client import QdrantClient


def test_hybrid_search():
    """Test hybrid retrieval with sample queries."""
    print("=" * 60)
    print("Hybrid Retrieval Test")
    print("=" * 60)
    
    # Load settings
    settings = get_settings()
    retrieval_settings = get_retrieval_settings()
    
    print(f"\nRetrieval Configuration:")
    print(f"  - Dense top_k: {retrieval_settings.top_k_dense}")
    print(f"  - Keyword top_k: {retrieval_settings.top_k_keyword}")
    print(f"  - Fusion method: {retrieval_settings.fusion_method}")
    print(f"  - Final top_k: {retrieval_settings.final_top_k}")
    
    if retrieval_settings.fusion_method == "rrf":
        print(f"  - RRF k: {retrieval_settings.rrf_k}")
    else:
        print(f"  - Dense weight: {retrieval_settings.dense_weight}")
        print(f"  - Keyword weight: {retrieval_settings.keyword_weight}")
    
    # Initialize services
    print("\nInitializing services...")
    embedding_service = EmbeddingService()
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port
    )
    
    # Create retriever
    retriever = HybridRetriever(
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
        collection_name=settings.qdrant_collection,
    )
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "document processing workflow",
        "authentication and security",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        
        results, debug_info = retriever.search(query)
        
        print(f"\nDebug Info:")
        for key, value in debug_info.items():
            print(f"  - {key}: {value}")
        
        print(f"\nResults ({len(results)} returned):")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. [{result.score:.4f}] {result.title}")
            print(f"     Doc: {result.document_id}, Page: {result.page}")
            print(f"     Text: {result.chunk_text[:150]}...")
            if result.dense_score is not None:
                print(f"     Dense: {result.dense_score:.4f}, Keyword: {result.keyword_score:.4f}")


if __name__ == "__main__":
    test_hybrid_search()
