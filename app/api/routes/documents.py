from __future__ import annotations

import json
import shutil
import tempfile
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import (
    dep_current_user,
    dep_document_ingestor,
    require_roles,
    dep_db,
)
from app.core.document_ingestor import DocumentIngestor
from app.db.models import User
from app.db.repositories.document_repository import DocumentRepository
from app.models.auth import UserRole
from app.models.common import PaginatedResponse, PaginationParams
from app.models.documents import (
    DocumentListItem,
    DocumentMetadata,
    DocumentStatus,
    DocumentUploadResponse,
)
from sqlalchemy.orm import Session

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    author: str = Form(default=""),
    tags: str = Form(default=""),
    url: str = Form(default=""),
    current_user: User = Depends(require_roles(UserRole.CONTRIBUTOR, UserRole.ADMIN)),
    ingestor: DocumentIngestor = Depends(dep_document_ingestor),
    db: Session = Depends(dep_db),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    suffix = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    doc_id = str(uuid.uuid4())
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    metadata = DocumentMetadata(
        document_id=doc_id,
        title=title or file.filename,
        author=author,
        path=tmp_path,
        tags=tag_list,
    )

    result = ingestor.ingest(tmp_path, metadata)

    doc_repo = DocumentRepository(db)
    doc_repo.create(metadata, file.filename, uploaded_by=current_user.id)
    doc_repo.update_status(doc_id, result.status, result.chunk_count)

    return DocumentUploadResponse(
        document_id=doc_id,
        status=result.status,
        chunk_count=result.chunk_count,
        message=f"Document ingested: {result.chunk_count} chunks from {result.page_count} pages",
    )


@router.get("", response_model=PaginatedResponse[DocumentListItem])
def list_documents(
    offset: int = 0,
    limit: int = 20,
    _user: User = Depends(dep_current_user),
    db: Session = Depends(dep_db),
):
    repo = DocumentRepository(db)
    docs = repo.list_all(offset, limit)
    total = repo.count()
    items = [
        DocumentListItem(
            document_id=d.document_id,
            title=d.title,
            author=d.author,
            path=d.path,
            tags=repo.get_tags(d),
            page_count=d.page_count,
            status=DocumentStatus(d.status),
            chunk_count=d.chunk_count,
            created_at=d.created_at,
        )
        for d in docs
    ]
    return PaginatedResponse(items=items, total=total, offset=offset, limit=limit)


@router.get("/{document_id}", response_model=DocumentListItem)
def get_document(
    document_id: str,
    _user: User = Depends(dep_current_user),
    db: Session = Depends(dep_db),
):
    repo = DocumentRepository(db)
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentListItem(
        document_id=doc.document_id,
        title=doc.title,
        author=doc.author,
        path=doc.path,
        tags=repo.get_tags(doc),
        page_count=doc.page_count,
        status=DocumentStatus(doc.status),
        chunk_count=doc.chunk_count,
        created_at=doc.created_at,
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    current_user: User = Depends(dep_current_user),
    ingestor: DocumentIngestor = Depends(dep_document_ingestor),
    db: Session = Depends(dep_db),
):
    repo = DocumentRepository(db)
    doc = repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    user_role = UserRole(current_user.role)
    if user_role != UserRole.ADMIN and doc.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot delete another user's document")

    ingestor.delete_document(document_id)
    repo.delete(document_id)
