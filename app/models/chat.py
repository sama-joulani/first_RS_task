from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.common import MetadataFilter


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: list[MetadataFilter] = Field(default_factory=list)
    stream: bool = False


class Citation(BaseModel):
    index: int
    document_title: str
    page: int
    section: str
    chunk_text: str


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    token_usage: TokenUsage | None = None


class StreamChunk(BaseModel):
    delta: str
    citations: list[Citation] | None = None
    done: bool = False
