from __future__ import annotations

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from app.core.embedding_service import EmbeddingService
from app.core.llm_service import LLMResponse, LLMService
from app.core.prompt_manager import PromptManager
from app.models.chat import ChatResponse, Citation, TokenUsage
from app.models.common import MetadataFilter
from app.models.search import SearchResponse, SearchResult
from app.utils.text_chunker import Chunk

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline: search Qdrant, inject context, generate with citations."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        prompt_manager: PromptManager,
        qdrant_client: QdrantClient,
        collection_name: str,
    ):
        self._embedding = embedding_service
        self._llm = llm_service
        self._prompt = prompt_manager
        self._qdrant = qdrant_client
        self._collection = collection_name

    def search(
        self, query: str, top_k: int = 5, filters: list[MetadataFilter] | None = None
    ) -> SearchResponse:
        query_vector = self._embedding.embed_text(query)
        qdrant_filter = self._build_filter(filters) if filters else None

        results = self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        ).points

        search_results = [
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

        return SearchResponse(results=search_results, total_count=len(search_results), query=query)

    def retrieve(
        self, query: str, top_k: int = 5, filters: list[MetadataFilter] | None = None
    ) -> list[Chunk]:
        response = self.search(query, top_k, filters)
        return [
            Chunk(
                text=r.chunk_text,
                chunk_index=r.chunk_index,
                metadata={
                    "document_id": r.document_id,
                    "title": r.title,
                    "author": r.author,
                    "page": r.page,
                    "section": r.section,
                    "tags": r.tags,
                },
            )
            for r in response.results
        ]

    def generate(self, query: str, context_chunks: list[Chunk]) -> ChatResponse:
        system_prompt = self._prompt.get_system_prompt()
        user_prompt = self._prompt.format_rag_prompt(query, context_chunks)

        llm_response: LLMResponse = self._llm.generate(user_prompt, system_prompt)

        citations = [
            Citation(
                index=i + 1,
                document_title=c.metadata.get("title", "Unknown"),
                page=c.metadata.get("page", 0),
                section=c.metadata.get("section", ""),
                chunk_text=c.text[:200],
            )
            for i, c in enumerate(context_chunks)
        ]

        return ChatResponse(
            answer=llm_response.content,
            citations=citations,
            token_usage=TokenUsage(
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.total_tokens,
            ),
        )

    def rag(
        self, query: str, top_k: int = 5, filters: list[MetadataFilter] | None = None
    ) -> ChatResponse:
        chunks = self.retrieve(query, top_k, filters)
        if not chunks:
            return ChatResponse(
                answer="I don't have enough information in the provided documents to answer this question.",
                citations=[],
            )
        return self.generate(query, chunks)

    async def arag(
        self, query: str, top_k: int = 5, filters: list[MetadataFilter] | None = None
    ) -> ChatResponse:
        chunks = self.retrieve(query, top_k, filters)
        if not chunks:
            return ChatResponse(
                answer="I don't have enough information in the provided documents to answer this question.",
                citations=[],
            )

        system_prompt = self._prompt.get_system_prompt()
        user_prompt = self._prompt.format_rag_prompt(query, chunks)
        llm_response = await self._llm.agenerate(user_prompt, system_prompt)

        citations = [
            Citation(
                index=i + 1,
                document_title=c.metadata.get("title", "Unknown"),
                page=c.metadata.get("page", 0),
                section=c.metadata.get("section", ""),
                chunk_text=c.text[:200],
            )
            for i, c in enumerate(chunks)
        ]

        return ChatResponse(
            answer=llm_response.content,
            citations=citations,
            token_usage=TokenUsage(
                prompt_tokens=llm_response.prompt_tokens,
                completion_tokens=llm_response.completion_tokens,
                total_tokens=llm_response.total_tokens,
            ),
        )

    async def arag_stream(
        self, query: str, top_k: int = 5, filters: list[MetadataFilter] | None = None
    ):
        chunks = self.retrieve(query, top_k, filters)
        if not chunks:
            yield {
                "delta": "I don't have enough information in the provided documents to answer this question.",
                "citations": [],
                "done": True,
            }
            return

        system_prompt = self._prompt.get_system_prompt()
        user_prompt = self._prompt.format_rag_prompt(query, chunks)

        citations = [
            Citation(
                index=i + 1,
                document_title=c.metadata.get("title", "Unknown"),
                page=c.metadata.get("page", 0),
                section=c.metadata.get("section", ""),
                chunk_text=c.text[:200],
            )
            for i, c in enumerate(chunks)
        ]

        async for token in self._llm.astream(user_prompt, system_prompt):
            yield {"delta": token, "citations": None, "done": False}

        yield {"delta": "", "citations": citations, "done": True}

    def _build_filter(self, filters: list[MetadataFilter]) -> Filter:
        conditions = []
        for f in filters:
            if f.operator == "eq":
                conditions.append(FieldCondition(key=f.field, match=MatchValue(value=f.value)))
            elif f.operator == "in" and isinstance(f.value, list):
                conditions.append(FieldCondition(key=f.field, match=MatchAny(any=f.value)))
            elif f.operator in ("gt", "gte", "lt", "lte"):
                range_kwargs = {f.operator: f.value}
                conditions.append(FieldCondition(key=f.field, range=Range(**range_kwargs)))
        return Filter(must=conditions) if conditions else None
