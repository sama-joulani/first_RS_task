"""Reset Qdrant collection and re-ingest all documents with new metadata."""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient

from app.config import get_settings
from app.core.document_ingestor import DocumentIngestor
from app.core.embedding_service import EmbeddingService
from app.db.database import create_tables, get_session_factory
from app.db.repositories.document_repository import DocumentRepository
from app.models.documents import DocumentMetadata


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def main():
    directory = Path(__file__).resolve().parent.parent
    
    files = [f for f in directory.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        print(f"No PDF/DOCX files found in {directory}")
        return

    print(f"Found {len(files)} document(s) to ingest.")

    settings = get_settings()
    create_tables()

    # Delete existing collection
    print("Deleting existing Qdrant collection...")
    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    try:
        qdrant.delete_collection(settings.qdrant_collection)
        print(f"Deleted collection: {settings.qdrant_collection}")
    except Exception as e:
        print(f"Collection may not exist: {e}")

    print("Loading embedding model (this may take a moment)...")
    embedding = EmbeddingService()
    ingestor = DocumentIngestor(
        embedding_service=embedding,
        qdrant_client=qdrant,
        collection_name=settings.qdrant_collection,
    )

    factory = get_session_factory()
    db = factory()
    doc_repo = DocumentRepository(db)

    try:
        for file_path in files:
            doc_id = str(uuid.uuid4())
            print(f"\nIngesting: {file_path.name} (id={doc_id})")

            metadata = DocumentMetadata(
                document_id=doc_id,
                title=file_path.stem,
                path=str(file_path),
                url=f"file://{file_path.name}",  # Add source URL
                category="document",  # Add category
            )

            result = ingestor.ingest(file_path, metadata)

            doc_repo.create(metadata, file_path.name)
            doc_repo.update_status(doc_id, result.status, result.chunk_count)

            print(f"  Status: {result.status.value}")
            print(f"  Pages:  {result.page_count}")
            print(f"  Chunks: {result.chunk_count}")
            print(f"  Language: {metadata.language}")
            if result.errors:
                print(f"  Errors: {result.errors}")
    finally:
        db.close()

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
