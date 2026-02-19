import json
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import Document
from app.models.documents import DocumentMetadata, DocumentStatus


class DocumentRepository:
    def __init__(self, db: Session):
        self._db = db

    def create(self, metadata: DocumentMetadata, filename: str, uploaded_by: int | None = None) -> Document:
        doc = Document(
            document_id=metadata.document_id,
            filename=filename,
            path=metadata.path,
            title=metadata.title,
            author=metadata.author,
            tags=json.dumps(metadata.tags),
            status=metadata.status,
            page_count=metadata.page_count,
            uploaded_by=uploaded_by,
        )
        self._db.add(doc)
        self._db.commit()
        self._db.refresh(doc)
        return doc

    def get_by_id(self, document_id: str) -> Document | None:
        return self._db.query(Document).filter(Document.document_id == document_id).first()

    def list_all(self, offset: int = 0, limit: int = 20) -> list[Document]:
        return self._db.query(Document).offset(offset).limit(limit).all()

    def count(self) -> int:
        return self._db.query(Document).count()

    def update_status(self, document_id: str, status: DocumentStatus, chunk_count: int = 0) -> Document | None:
        doc = self.get_by_id(document_id)
        if doc:
            doc.status = status
            doc.chunk_count = chunk_count
            if status == DocumentStatus.INDEXED:
                doc.indexed_at = datetime.utcnow()
            self._db.commit()
            self._db.refresh(doc)
        return doc

    def delete(self, document_id: str) -> bool:
        doc = self.get_by_id(document_id)
        if doc:
            self._db.delete(doc)
            self._db.commit()
            return True
        return False

    def get_tags(self, doc: Document) -> list[str]:
        try:
            return json.loads(doc.tags) if doc.tags else []
        except (json.JSONDecodeError, TypeError):
            return []
