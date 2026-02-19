from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.observability.observability_manager import ObservabilityManager


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request/response with latency via ObservabilityManager."""

    def __init__(self, app, obs: ObservabilityManager):
        super().__init__(app)
        self._obs = obs

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        user_id = getattr(request.state, "user_id", None)

        self._obs.log_request(request.method, request.url.path, correlation_id, user_id)
        start = time.perf_counter()

        response = await call_next(request)

        latency_ms = (time.perf_counter() - start) * 1000
        self._obs.log_response(
            request.method, request.url.path, response.status_code,
            correlation_id, latency_ms, user_id,
        )
        return response
