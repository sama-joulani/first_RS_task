"""Hybrid retrieval combining dense embeddings with BM25 keyword search."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.core.embedding_service import EmbeddingService
from app.core.keyword_search import BM25Searcher, KeywordResult
from app.core.retrieval_config import get_retrieval_settings
from app.models.common import MetadataFilter
from app.models.search import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result after fusion of dense and keyword scores."""
    chunk_id: str
    chunk_text: str
    dense_score: float
    keyword_score: float
    fused_score: float
    document_id: str
    title: str
    author: str
    page: int
    section: str
    chunk_index: int
    tags: list[str]


class HybridRetriever:
    """Hybrid retriever combining dense and keyword search with fusion."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_client: QdrantClient,
        collection_name: str,
    ):
        self._embedding = embedding_service
        self._qdrant = qdrant_client
        self._collection = collection_name
        self._bm25 = BM25Searcher()
        self._settings = get_retrieval_settings()
        
        # Cache for keyword index
        self._index_built: bool = False
    
    def search(
        self,
        query: str,
        filters: list[MetadataFilter] | None = None,
        use_cache: bool = True
    ) -> tuple[list[SearchResult], dict]:
        """
        Perform hybrid search and return fused results.
        
        Returns:
            Tuple of (search_results, debug_info)
        """
        settings = self._settings
        
        # Step 1: Dense retrieval from Qdrant
        dense_results = self._dense_search(query, settings.top_k_dense, filters)
        
        # Step 2: Keyword retrieval (build index if needed or not cached)
        keyword_results = self._keyword_search(query, settings.top_k_keyword, use_cache)
        
        # Step 3: Fuse results
        fused_results = self._fuse_results(dense_results, keyword_results)
        
        # Step 4: Return top_k final results
        final_results = fused_results[:settings.final_top_k]
        
        # Convert to SearchResult format
        search_results = [
            SearchResult(
                chunk_text=r.chunk_text,
                score=r.fused_score,
                document_id=r.document_id,
                title=r.title,
                author=r.author,
                page=r.page,
                section=r.section,
                chunk_index=r.chunk_index,
                tags=r.tags,
                dense_score=r.dense_score,
                keyword_score=r.keyword_score,
            )
            for r in final_results
        ]
        
        debug_info = {
            "dense_count": len(dense_results),
            "keyword_count": len(keyword_results),
            "fusion_method": settings.fusion_method,
            "dense_weight": settings.dense_weight if settings.fusion_method == "weighted" else None,
            "keyword_weight": settings.keyword_weight if settings.fusion_method == "weighted" else None,
            "rrf_k": settings.rrf_k if settings.fusion_method == "rrf" else None,
        }
        
        return search_results, debug_info
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        filters: list[MetadataFilter] | None = None
    ) -> list[SearchResult]:
        """Perform dense vector search using Qdrant."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range
        
        query_vector = self._embedding.embed_text(query)
        
        # Build filter if provided
        qdrant_filter = None
        if filters:
            conditions = []
            for f in filters:
                if f.operator == "eq":
                    conditions.append(FieldCondition(key=f.field, match=MatchValue(value=f.value)))
                elif f.operator == "in" and isinstance(f.value, list):
                    conditions.append(FieldCondition(key=f.field, match=MatchAny(any=f.value)))
                elif f.operator in ("gt", "gte", "lt", "lte"):
                    range_kwargs = {f.operator: f.value}
                    conditions.append(FieldCondition(key=f.field, range=Range(**range_kwargs)))
            qdrant_filter = Filter(must=conditions) if conditions else None
        
        results = self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ).points
        
        return [
            SearchResult(
                chunk_text=r.payload.get("text", ""),
                score=r.score,
                document_id=r.payload.get("document_id", ""),
                title=r.payload.get("title", ""),
                author=r.payload.get("author", ""),
                page=r.payload.get("page", 0),
                section=r.payload.get("section", ""),
                chunk_index=r.payload.get("chunk_index", 0),
                tags=r.payload.get("tags", []),
            )
            for r in results
        ]
    
    def _keyword_search(self, query: str, top_k: int, use_cache: bool) -> list[KeywordResult]:
        """Perform BM25 keyword search."""
        # Build index if not already done
        if not self._index_built or not use_cache:
            self._build_keyword_index()
        
        return self._bm25.search(query, top_k)
    
    def _build_keyword_index(self):
        """Build BM25 index from all documents in Qdrant."""
        logger.info("Building BM25 keyword index...")
        self._bm25.clear()
        
        # Scroll through all points in collection
        offset = None
        batch_size = 100
        total_indexed = 0
        
        while True:
            response = self._qdrant.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
            )
            
            points = response[0]
            if not points:
                break
            
            for point in points:
                payload = point.payload or {}
                chunk_id = str(point.id)
                text = payload.get("text", "")
                
                if text:
                    metadata = {
                        "document_id": payload.get("document_id", ""),
                        "title": payload.get("title", ""),
                        "author": payload.get("author", ""),
                        "page": payload.get("page", 0),
                        "section": payload.get("section", ""),
                        "chunk_index": payload.get("chunk_index", 0),
                        "tags": payload.get("tags", []),
                    }
                    self._bm25.add_document(chunk_id, text, metadata)
                    total_indexed += 1
            
            offset = response[1]
            if offset is None:
                break
        
        self._bm25.build_vocab()
        self._index_built = True
        logger.info("BM25 index built with %d documents", total_indexed)
    
    def _fuse_results(
        self,
        dense_results: list[SearchResult],
        keyword_results: list[KeywordResult]
    ) -> list[FusionResult]:
        """Fuse dense and keyword results using configured fusion method."""
        settings = self._settings
        
        # Create lookup by document+chunk for deduplication
        # Use (document_id, chunk_index) as unique key
        all_results: dict[tuple, FusionResult] = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            key = (result.document_id, result.chunk_index)
            all_results[key] = FusionResult(
                chunk_id=f"{result.document_id}_{result.chunk_index}",
                chunk_text=result.chunk_text,
                dense_score=result.score,
                keyword_score=0.0,
                fused_score=0.0,  # Will be computed
                document_id=result.document_id,
                title=result.title,
                author=result.author,
                page=result.page,
                section=result.section,
                chunk_index=result.chunk_index,
                tags=result.tags,
            )
        
        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            key = (result.document_id, result.chunk_index)
            if key in all_results:
                all_results[key].keyword_score = result.score
            else:
                all_results[key] = FusionResult(
                    chunk_id=result.chunk_id,
                    chunk_text=result.chunk_text,
                    dense_score=0.0,
                    keyword_score=result.score,
                    fused_score=0.0,
                    document_id=result.document_id,
                    title=result.title,
                    author=result.author,
                    page=result.page,
                    section=result.section,
                    chunk_index=result.chunk_index,
                    tags=result.tags,
                )
        
        # Apply fusion method
        if settings.fusion_method == "rrf":
            fused = self._reciprocal_rank_fusion(list(all_results.values()), dense_results, keyword_results)
        else:
            fused = self._weighted_fusion(list(all_results.values()))
        
        # Sort by fused score descending
        fused.sort(key=lambda x: x.fused_score, reverse=True)
        
        return fused
    
    def _reciprocal_rank_fusion(
        self,
        all_results: list[FusionResult],
        dense_results: list[SearchResult],
        keyword_results: list[KeywordResult]
    ) -> list[FusionResult]:
        """Apply Reciprocal Rank Fusion (RRF)."""
        k = self._settings.rrf_k
        
        # Create rank lookups
        dense_ranks = {(r.document_id, r.chunk_index): rank for rank, r in enumerate(dense_results, 1)}
        keyword_ranks = {(r.document_id, r.chunk_index): rank for rank, r in enumerate(keyword_results, 1)}
        
        for result in all_results:
            dense_rank = dense_ranks.get((result.document_id, result.chunk_index), 0)
            keyword_rank = keyword_ranks.get((result.document_id, result.chunk_index), 0)
            
            # RRF score: sum of 1/(k + rank) for each list where item appears
            score = 0.0
            if dense_rank > 0:
                score += 1.0 / (k + dense_rank)
            if keyword_rank > 0:
                score += 1.0 / (k + keyword_rank)
            
            result.fused_score = score
        
        return all_results
    
    def _weighted_fusion(self, all_results: list[FusionResult]) -> list[FusionResult]:
        """Apply weighted score fusion."""
        settings = self._settings
        
        # Normalize scores to [0, 1] range
        max_dense = max((r.dense_score for r in all_results), default=1.0)
        max_keyword = max((r.keyword_score for r in all_results), default=1.0)
        
        for result in all_results:
            norm_dense = result.dense_score / max_dense if max_dense > 0 else 0
            norm_keyword = result.keyword_score / max_keyword if max_keyword > 0 else 0
            
            result.fused_score = (
                settings.dense_weight * norm_dense +
                settings.keyword_weight * norm_keyword
            )
        
        return all_results
    
    def clear_index(self):
        """Clear the keyword index (useful for reindexing)."""
        self._bm25.clear()
        self._index_built = False
