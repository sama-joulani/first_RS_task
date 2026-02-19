from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.common import MetadataFilter


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: list[MetadataFilter] = Field(default_factory=list)


class SearchResult(BaseModel):
    chunk_text: str
    score: float
    document_id: str
    title: str
    author: str
    page: int
    section: str
    chunk_index: int
    tags: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_count: int
    query: str
