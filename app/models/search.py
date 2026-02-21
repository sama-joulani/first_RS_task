from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.common import MetadataFilter


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: list[MetadataFilter] = Field(default_factory=list)
    # New hybrid search options
    use_hybrid: bool = Field(default=False, description="Enable hybrid search (dense + keyword)")
    fusion_method: str | None = Field(default=None, description="Override fusion method: 'rrf' or 'weighted'")


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
    # New fields for hybrid search debugging
    dense_score: float | None = Field(default=None, description="Dense retrieval score")
    keyword_score: float | None = Field(default=None, description="Keyword retrieval score")


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_count: int
    query: str
    # New fields for hybrid search info
    search_method: str = Field(default="dense", description="Search method used: 'dense' or 'hybrid'")
    fusion_method: str | None = Field(default=None, description="Fusion method used for hybrid search")
    debug_info: dict | None = Field(default=None, description="Debug information about retrieval")
