from __future__ import annotations

import enum
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    document_id: str
    title: str = ""
    author: str = ""
    path: str = ""
    url: str = ""  # Source URL for the document
    tags: list[str] = Field(default_factory=list)
    page_count: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Additional metadata fields
    category: str = ""  # Document category/type
    language: str = ""  # Document language (en, ar, etc.)
    subject: str = ""  # Subject/topic
    keywords: list[str] = Field(default_factory=list)  # Extracted keywords
    file_size: int = 0  # File size in bytes
    file_type: str = ""  # MIME type or extension
    description: str = ""  # Brief description
    version: str = ""  # Document version
    source: str = ""  # Source system or origin


class ChunkMetadata(BaseModel):
    document_id: str
    title: str = ""
    author: str = ""
    path: str = ""
    tags: list[str] = Field(default_factory=list)
    page: int = 0
    section: str = ""
    url: str = ""
    chunk_index: int = 0


class DocumentUploadRequest(BaseModel):
    title: str | None = None
    author: str | None = None
    tags: list[str] = Field(default_factory=list)
    url: str | None = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    chunk_count: int
    message: str


class DocumentListItem(BaseModel):
    document_id: str
    title: str
    author: str
    path: str
    tags: list[str]
    page_count: int
    status: DocumentStatus
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}
