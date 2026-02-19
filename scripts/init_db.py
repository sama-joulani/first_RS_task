"""Initialize database tables and Qdrant collection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.core.embedding_service import EmbeddingService
from app.db.database import create_tables


def main():
    print("Creating database tables...")
    create_tables()
    print("Database tables created successfully.")

    settings = get_settings()
    print(f"Ensuring Qdrant collection '{settings.qdrant_collection}' exists...")

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        collections = [c.name for c in client.get_collections().collections]

        # Get dimension from embedding service
        embedding = EmbeddingService()
        dimension = embedding.dimension

        if settings.qdrant_collection not in collections:
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection: {settings.qdrant_collection} (dim={dimension})")
        else:
            print(f"Qdrant collection '{settings.qdrant_collection}' already exists.")
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant ({e}). Start Qdrant and retry.")


if __name__ == "__main__":
    main()
