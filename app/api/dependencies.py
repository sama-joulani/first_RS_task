from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.core.embedding_service import EmbeddingService
from app.core.llm_service import LLMService
from app.core.prompt_manager import PromptManager
from app.core.document_ingestor import DocumentIngestor
from app.core.rag_pipeline import RAGPipeline
from app.db.database import get_db
from app.db.models import User
from app.models.auth import UserRole
from app.observability.observability_manager import ObservabilityManager
from app.security.auth_manager import AuthManager
from app.security.security_manager import SecurityManager

bearer_scheme = HTTPBearer(auto_error=False)

# ---- Singletons (lazily initialized) ----

_embedding_service: EmbeddingService | None = None
_llm_service: LLMService | None = None
_prompt_manager: PromptManager | None = None
_qdrant_client: QdrantClient | None = None
_obs_manager: ObservabilityManager | None = None
_security_manager: SecurityManager | None = None


def dep_settings() -> Settings:
    return get_settings()


def dep_db(db: Session = Depends(get_db)) -> Session:
    return db


def dep_embedding_service(settings: Settings = Depends(dep_settings)) -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(settings.embedding_model)
    return _embedding_service


def dep_llm_service(settings: Settings = Depends(dep_settings)) -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )
    return _llm_service


def dep_prompt_manager() -> PromptManager:
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def dep_qdrant_client(settings: Settings = Depends(dep_settings)) -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return _qdrant_client


def dep_observability(settings: Settings = Depends(dep_settings)) -> ObservabilityManager:
    global _obs_manager
    if _obs_manager is None:
        _obs_manager = ObservabilityManager(settings)
    return _obs_manager


def dep_security_manager() -> SecurityManager:
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def dep_auth_manager(db: Session = Depends(dep_db), settings: Settings = Depends(dep_settings)) -> AuthManager:
    return AuthManager(db, settings)


def dep_document_ingestor(
    embedding: EmbeddingService = Depends(dep_embedding_service),
    qdrant: QdrantClient = Depends(dep_qdrant_client),
    settings: Settings = Depends(dep_settings),
) -> DocumentIngestor:
    return DocumentIngestor(
        embedding_service=embedding,
        qdrant_client=qdrant,
        collection_name=settings.qdrant_collection,
    )


def dep_rag_pipeline(
    embedding: EmbeddingService = Depends(dep_embedding_service),
    llm: LLMService = Depends(dep_llm_service),
    prompt: PromptManager = Depends(dep_prompt_manager),
    qdrant: QdrantClient = Depends(dep_qdrant_client),
    settings: Settings = Depends(dep_settings),
) -> RAGPipeline:
    return RAGPipeline(
        embedding_service=embedding,
        llm_service=llm,
        prompt_manager=prompt,
        qdrant_client=qdrant,
        collection_name=settings.qdrant_collection,
    )


# ---- Auth dependencies ----

async def dep_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(dep_db),
    settings: Settings = Depends(dep_settings),
) -> User:
    import logging
    logger = logging.getLogger(__name__)
    
    auth_mgr = AuthManager(db, settings)

    # Try JWT bearer token
    if credentials:
        logger.info("Attempting JWT auth with token: %s...", credentials.credentials[:20])
        payload = auth_mgr.verify_token(credentials.credentials)
        if payload:
            user_id = int(payload.sub)
            logger.info("JWT valid, user_id: %s", user_id)
            user = auth_mgr.get_user_by_id(user_id)
            if user and user.is_active:
                request.state.user_id = user.id
                return user
            else:
                logger.warning("User not found or inactive: %s", user_id)
        else:
            logger.warning("JWT verification failed")

    # Try API key from header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        key_payload = auth_mgr.verify_api_key(api_key)
        if key_payload:
            user = auth_mgr.get_user_by_id(key_payload.user_id)
            if user and user.is_active:
                request.state.user_id = user.id
                return user

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing credentials")


def require_roles(*roles: UserRole):
    """Factory that returns a dependency checking the current user has one of the given roles."""

    async def _check(
        current_user: User = Depends(dep_current_user),
        sec: SecurityManager = Depends(dep_security_manager),
    ) -> User:
        user_role = UserRole(current_user.role)
        if not sec.has_any_role(user_role, list(roles)):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user

    return _check
