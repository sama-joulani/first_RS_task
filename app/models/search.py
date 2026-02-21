from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import (
    dep_current_user,
    dep_embedding_service,
    dep_qdrant_client,
    dep_rag_pipeline,
)
from app.config import get_settings
from app.core.hybrid_retrieval import HybridRetriever
from app.core.rag_pipeline import RAGPipeline
from app.core.retrieval_config import get_retrieval_settings
from app.db.models import User
from app.models.search import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["Search"])


@router.post("", response_model=SearchResponse)
def search(
    body: SearchRequest,
    _user: User = Depends(dep_current_user),
    pipeline: RAGPipeline = Depends(dep_rag_pipeline),
    embedding_service=Depends(dep_embedding_service),
    qdrant_client=Depends(dep_qdrant_client),
):
    """Search with optional hybrid retrieval."""
    settings = get_settings()
    
    if body.use_hybrid:
        # Use hybrid retrieval
        retriever = HybridRetriever(
            embedding_service=embedding_service,
            qdrant_client=qdrant_client,
            collection_name=settings.qdrant_collection,
        )
        
        # Override fusion method if specified
        retrieval_settings = get_retrieval_settings()
        if body.fusion_method:
            # Create a temporary override (this is a simple approach)
            original_method = retrieval_settings.fusion_method
            retrieval_settings.fusion_method = body.fusion_method
        
        results, debug_info = retriever.search(
            query=body.query,
            filters=body.filters or None
        )
        
        # Restore original method if overridden
        if body.fusion_method:
            retrieval_settings.fusion_method = original_method
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query=body.query,
            search_method="hybrid",
            fusion_method=retrieval_settings.fusion_method,
            debug_info=debug_info
        )
    else:
        # Use standard dense retrieval
        return pipeline.search(query=body.query, top_k=body.top_k, filters=body.filters or None)


@router.post("/hybrid", response_model=SearchResponse)
def search_hybrid(
    body: SearchRequest,
    _user: User = Depends(dep_current_user),
    embedding_service=Depends(dep_embedding_service),
    qdrant_client=Depends(dep_qdrant_client),
):
    """Explicit hybrid search endpoint."""
    settings = get_settings()
    
    retriever = HybridRetriever(
        embedding_service=embedding_service,
        qdrant_client=qdrant_client,
        collection_name=settings.qdrant_collection,
    )
    
    # Override fusion method if specified in request
    retrieval_settings = get_retrieval_settings()
    if body.fusion_method:
        original_method = retrieval_settings.fusion_method
        retrieval_settings.fusion_method = body.fusion_method
    
    results, debug_info = retriever.search(
        query=body.query,
        filters=body.filters or None
    )
    
    # Restore if overridden
    if body.fusion_method:
        retrieval_settings.fusion_method = original_method
    
    return SearchResponse(
        results=results,
        total_count=len(results),
        query=body.query,
        search_method="hybrid",
        fusion_method=retrieval_settings.fusion_method,
        debug_info=debug_info
    )
