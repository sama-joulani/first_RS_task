import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.middleware.correlation import CorrelationMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.api.routes import admin, auth, chat, documents, search
from app.config import get_settings
from app.db.database import create_tables
from app.observability.observability_manager import ObservabilityManager
from app.security.rate_limiter import RateLimiter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

settings = get_settings()
obs = ObservabilityManager(settings)
rate_limiter = RateLimiter(settings)

app = FastAPI(
    title="RAG Semantic Search API",
    description="Retrieval-Augmented Generation API with citations, RBAC, and observability",
    version="1.0.0",
)

# ---- Middleware (order matters: last added = first executed) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware, obs=obs)
app.add_middleware(CorrelationMiddleware)

# ---- Rate limiting ----
app.state.limiter = rate_limiter.limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---- Routers ----
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(admin.router)

# ---- Static files (UI) with no-cache headers ----
@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serve index.html with cache-busting headers."""
    return FileResponse(
        os.path.join("static", "index.html"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ---- Startup ----
@app.on_event("startup")
def on_startup():
    create_tables()
    logging.getLogger(__name__).info("Database tables created, application ready")


# ---- Health check ----
@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


# ---- Global error handler ----
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    cid = getattr(request.state, "correlation_id", "unknown")
    obs.log_error(exc, cid)
    return JSONResponse(
        status_code=500,
        content={"error_code": "INTERNAL_ERROR", "message": "An unexpected error occurred", "details": None},
    )
