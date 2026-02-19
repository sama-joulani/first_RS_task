from __future__ import annotations

import logging
import time
import uuid

import logfire

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ObservabilityManager:
    """Structured logging and monitoring via Logfire."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._configured = False
        self._configure()

    def _configure(self) -> None:
        if self._configured:
            return
        if self._settings.logfire_token:
            logfire.configure(token=self._settings.logfire_token)
            self._configured = True
        else:
            logger.warning("Logfire token not set; observability will use stdlib logging only")

    @staticmethod
    def generate_correlation_id() -> str:
        return str(uuid.uuid4())

    def log_request(self, method: str, path: str, correlation_id: str, user_id: int | None = None) -> None:
        data = {
            "event": "http_request",
            "method": method,
            "path": path,
            "correlation_id": correlation_id,
            "user_id": user_id,
        }
        if self._configured:
            logfire.info("HTTP {method} {path}", **data)
        logger.info("request %s %s cid=%s user=%s", method, path, correlation_id, user_id)

    def log_response(
        self, method: str, path: str, status_code: int,
        correlation_id: str, latency_ms: float, user_id: int | None = None,
    ) -> None:
        data = {
            "event": "http_response",
            "method": method,
            "path": path,
            "status_code": status_code,
            "correlation_id": correlation_id,
            "latency_ms": round(latency_ms, 2),
            "user_id": user_id,
        }
        if self._configured:
            logfire.info("HTTP {method} {path} -> {status_code} ({latency_ms}ms)", **data)
        logger.info(
            "response %s %s status=%d latency=%.2fms cid=%s",
            method, path, status_code, latency_ms, correlation_id,
        )

    def log_llm_call(
        self, model: str, prompt_tokens: int, completion_tokens: int,
        latency_ms: float, correlation_id: str,
    ) -> None:
        total = prompt_tokens + completion_tokens
        data = {
            "event": "llm_call",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "latency_ms": round(latency_ms, 2),
            "correlation_id": correlation_id,
        }
        if self._configured:
            logfire.info("LLM call {model}: {total_tokens} tokens ({latency_ms}ms)", **data)
        logger.info(
            "llm model=%s tokens=%d latency=%.2fms cid=%s",
            model, total, latency_ms, correlation_id,
        )

    def log_retrieval(
        self, query: str, num_results: int, latency_ms: float, correlation_id: str,
    ) -> None:
        data = {
            "event": "retrieval",
            "query": query[:200],
            "num_results": num_results,
            "latency_ms": round(latency_ms, 2),
            "correlation_id": correlation_id,
        }
        if self._configured:
            logfire.info("Retrieval: {num_results} results ({latency_ms}ms)", **data)
        logger.info(
            "retrieval results=%d latency=%.2fms cid=%s",
            num_results, latency_ms, correlation_id,
        )

    def log_ingestion(
        self, document_id: str, chunks: int, latency_ms: float, correlation_id: str,
    ) -> None:
        data = {
            "event": "ingestion",
            "document_id": document_id,
            "chunks": chunks,
            "latency_ms": round(latency_ms, 2),
            "correlation_id": correlation_id,
        }
        if self._configured:
            logfire.info("Ingested {document_id}: {chunks} chunks ({latency_ms}ms)", **data)
        logger.info(
            "ingestion doc=%s chunks=%d latency=%.2fms cid=%s",
            document_id, chunks, latency_ms, correlation_id,
        )

    def log_error(self, error: Exception, correlation_id: str, context: dict | None = None) -> None:
        data = {
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "correlation_id": correlation_id,
            **(context or {}),
        }
        if self._configured:
            logfire.error("Error: {error_type}: {error_message}", **data)
        logger.error("error %s: %s cid=%s", type(error).__name__, error, correlation_id)
