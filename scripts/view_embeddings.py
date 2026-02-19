"""View embedding vectors from Qdrant."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from app.config import get_settings


def main():
    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    
    # Get collection info
    collection = client.get_collection(settings.qdrant_collection)
    print(f"Collection: {settings.qdrant_collection}")
    print(f"Total points: {collection.points_count}")
    print(f"Vector dimension: {collection.config.params.vectors.size}")
    print("=" * 80)
    
    # Retrieve points with vectors
    points = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=2,  # Show first 2 points
        with_vectors=True  # Include the embedding vectors
    )[0]
    
    for i, point in enumerate(points, 1):
        print(f"\n--- Point {i} ---")
        print(f"ID: {point.id}")
        print(f"Document ID: {point.payload.get('document_id', 'N/A')}")
        print(f"Title: {point.payload.get('title', 'N/A')}")
        print(f"Page: {point.payload.get('page', 'N/A')}")
        print(f"Chunk Index: {point.payload.get('chunk_index', 'N/A')}")
        print(f"Text preview: {point.payload.get('text', '')[:150]}...")
        
        # Show embedding vector (first 10 values and last 10 values)
        vector = point.vector
        print(f"\nEmbedding vector ({len(vector)} dimensions):")
        print(f"  First 10 values: {vector[:10]}")
        print(f"  Last 10 values:  {vector[-10:]}")
        print(f"  Sample middle:   {vector[100:110]}")
        print("=" * 80)


if __name__ == "__main__":
    main()
