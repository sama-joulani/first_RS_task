from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import dep_current_user, dep_rag_pipeline
from app.core.rag_pipeline import RAGPipeline
from app.db.models import User
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    _user: User = Depends(dep_current_user),
    pipeline: RAGPipeline = Depends(dep_rag_pipeline),
):
    if body.stream:
        return StreamingResponse(
            _stream_response(pipeline, body),
            media_type="text/event-stream",
        )
    return await pipeline.arag(query=body.query, top_k=body.top_k, filters=body.filters or None)


async def _stream_response(pipeline: RAGPipeline, body: ChatRequest):
    async for chunk in pipeline.arag_stream(query=body.query, top_k=body.top_k, filters=body.filters or None):
        data = json.dumps(chunk, default=str)
        yield f"data: {data}\n\n"
    yield "data: [DONE]\n\n"
