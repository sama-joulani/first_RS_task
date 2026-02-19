from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import dep_auth_manager, dep_current_user, require_roles
from app.db.models import User
from app.models.auth import (
    APIKeyCreate,
    APIKeyResponse,
    TokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
    UserRole,
)
from app.security.auth_manager import AuthManager

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    data: UserCreate,
    _admin: User = Depends(require_roles(UserRole.ADMIN)),
    auth: AuthManager = Depends(dep_auth_manager),
):
    existing = auth._user_repo.get_by_email(data.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user = auth.register_user(data)
    return UserResponse.model_validate(user)


@router.post("/login", response_model=TokenResponse)
def login(data: TokenRequest, auth: AuthManager = Depends(dep_auth_manager)):
    user = auth.authenticate_user(data.email, data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = auth.create_access_token(user.id, UserRole(user.role))
    return TokenResponse(
        access_token=token,
        expires_in=auth._settings.jwt_expiry_minutes * 60,
    )


@router.post("/api-keys", response_model=dict)
def create_api_key(
    data: APIKeyCreate,
    current_user: User = Depends(dep_current_user),
    auth: AuthManager = Depends(dep_auth_manager),
):
    raw_key, api_key = auth.create_api_key(current_user.id, data.name)
    return {
        "key": raw_key,
        "id": api_key.id,
        "name": api_key.name,
        "key_prefix": api_key.key_prefix,
        "message": "Store this key securely. It cannot be retrieved again.",
    }


@router.get("/api-keys", response_model=list[APIKeyResponse])
def list_api_keys(current_user: User = Depends(dep_current_user), auth: AuthManager = Depends(dep_auth_manager)):
    from app.db.models import APIKey

    keys = auth._db.query(APIKey).filter(APIKey.user_id == current_user.id, APIKey.is_active == True).all()  # noqa: E712
    return [APIKeyResponse.model_validate(k) for k in keys]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_api_key(
    key_id: int,
    current_user: User = Depends(dep_current_user),
    auth: AuthManager = Depends(dep_auth_manager),
):
    from app.db.models import APIKey

    key = auth._db.query(APIKey).filter(APIKey.id == key_id, APIKey.user_id == current_user.id).first()
    if not key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    key.is_active = False
    auth._db.commit()


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(dep_current_user)):
    return UserResponse.model_validate(current_user)
