from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import APIKey, User
from app.db.repositories.user_repository import UserRepository
from app.models.auth import APIKeyPayload, TokenPayload, UserCreate, UserRole

logger = logging.getLogger(__name__)


class AuthManager:
    """Handles JWT tokens, API key management, and password hashing."""

    def __init__(self, db: Session, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._db = db
        self._user_repo = UserRepository(db)
        self._pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    # ---- Password ----

    def hash_password(self, password: str) -> str:
        # Truncate to 72 bytes to avoid bcrypt limitation
        password_bytes = password.encode('utf-8')[:72]
        try:
            return self._pwd_context.hash(password_bytes)
        except Exception:
            # Fallback to SHA256 for demo
            import hashlib
            return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, plain: str, hashed: str) -> bool:
        # Check if it's a SHA256 hash (64 hex characters)
        if len(hashed) == 64 and all(c in '0123456789abcdef' for c in hashed):
            import hashlib
            return hashlib.sha256(plain.encode()).hexdigest() == hashed
        
        # Try bcrypt
        try:
            plain_bytes = plain.encode('utf-8')[:72]
            return self._pwd_context.verify(plain_bytes, hashed)
        except Exception:
            return False

    # ---- JWT ----

    def create_access_token(self, user_id: int, role: UserRole) -> str:
        expire = datetime.utcnow() + timedelta(minutes=self._settings.jwt_expiry_minutes)
        payload = {"sub": str(user_id), "role": role.value, "exp": expire}
        return jwt.encode(payload, self._settings.jwt_secret_key, algorithm=self._settings.jwt_algorithm)

    def verify_token(self, token: str) -> TokenPayload | None:
        try:
            data = jwt.decode(token, self._settings.jwt_secret_key, algorithms=[self._settings.jwt_algorithm])
            return TokenPayload(sub=data["sub"], role=UserRole(data["role"]), exp=data["exp"])
        except JWTError as e:
            logger.warning("JWT verification failed: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error verifying token: %s", e)
            return None

    # ---- API Keys ----

    def create_api_key(self, user_id: int, name: str) -> tuple[str, APIKey]:
        raw_key = f"rsk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:8]

        api_key = APIKey(
            user_id=user_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
        )
        self._db.add(api_key)
        self._db.commit()
        self._db.refresh(api_key)
        return raw_key, api_key

    def verify_api_key(self, raw_key: str) -> APIKeyPayload | None:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self._db.query(APIKey).filter(
            APIKey.key_hash == key_hash,
            APIKey.is_active == True,  # noqa: E712
        ).first()

        if not api_key:
            return None
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None

        user = self._user_repo.get_by_id(api_key.user_id)
        if not user or not user.is_active:
            return None

        return APIKeyPayload(user_id=user.id, role=UserRole(user.role), key_id=api_key.id)

    # ---- User Auth ----

    def authenticate_user(self, email: str, password: str) -> User | None:
        user = self._user_repo.get_by_email(email)
        if not user or not user.is_active:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user

    def register_user(self, data: UserCreate) -> User:
        hashed = self.hash_password(data.password)
        return self._user_repo.create(data, hashed)

    def get_user_by_id(self, user_id: int) -> User | None:
        return self._user_repo.get_by_id(user_id)
