"""Test RAG pipeline retrieval."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.core.embedding_service import EmbeddingService
from app.core.llm_service import LLMService
from app.core.prompt_manager import PromptManager
from app.core.rag_pipeline import RAGPipeline
from qdrant_client import QdrantClient


def main():
    settings = get_settings()
    
    print("Initializing RAG pipeline...")
    embedding = EmbeddingService()
    llm = LLMService()
    prompt = PromptManager()
    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    
    pipeline = RAGPipeline(
        embedding_service=embedding,
        llm_service=llm,
        prompt_manager=prompt,
        qdrant_client=qdrant,
        collection_name=settings.qdrant_collection,
    )
    
    # Test search
    query = input("Enter your question: ").strip()
    print(f"\nSearching for: {query}")
    
    results = pipeline.search(query, top_k=5)
    print(f"\nFound {results.total_count} results:")
    
    for i, r in enumerate(results.results, 1):
        print(f"\n{i}. Score: {r.score:.3f}")
        print(f"   Title: {r.title}")
        print(f"   Page: {r.page}")
        print(f"   Text: {r.chunk_text[:150]}...")
    
    if results.total_count == 0:
        print("\n❌ No results found! Documents may need to be re-ingested.")
    else:
        print("\n✓ Retrieval is working. Testing full RAG...")
        response = pipeline.rag(query, top_k=5)
        print(f"\nAnswer: {response.answer}")


if __name__ == "__main__":
    main()
