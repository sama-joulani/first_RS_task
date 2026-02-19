from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import dep_current_user, dep_rag_pipeline
from app.core.rag_pipeline import RAGPipeline
from app.db.models import User
from app.models.search import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
def search(
    body: SearchRequest,
    _user: User = Depends(dep_current_user),
    pipeline: RAGPipeline = Depends(dep_rag_pipeline),
):
    return pipeline.search(query=body.query, top_k=body.top_k, filters=body.filters or None)
