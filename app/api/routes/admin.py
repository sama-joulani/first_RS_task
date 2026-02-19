from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import dep_db, require_roles
from app.db.models import User
from app.db.repositories.document_repository import DocumentRepository
from app.db.repositories.user_repository import UserRepository
from app.models.auth import UserResponse, UserRole
from app.models.common import PaginatedResponse

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/users", response_model=PaginatedResponse[UserResponse])
def list_users(
    offset: int = 0,
    limit: int = 20,
    _admin: User = Depends(require_roles(UserRole.ADMIN)),
    db: Session = Depends(dep_db),
):
    repo = UserRepository(db)
    users = repo.list_all(offset, limit)
    total = repo.count()
    items = [UserResponse.model_validate(u) for u in users]
    return PaginatedResponse(items=items, total=total, offset=offset, limit=limit)


@router.put("/users/{user_id}/role", response_model=UserResponse)
def update_user_role(
    user_id: int,
    role: UserRole,
    _admin: User = Depends(require_roles(UserRole.ADMIN)),
    db: Session = Depends(dep_db),
):
    repo = UserRepository(db)
    user = repo.update_role(user_id, role)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse.model_validate(user)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    admin: User = Depends(require_roles(UserRole.ADMIN)),
    db: Session = Depends(dep_db),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    repo = UserRepository(db)
    if not repo.delete(user_id):
        raise HTTPException(status_code=404, detail="User not found")


@router.get("/stats")
def system_stats(
    _admin: User = Depends(require_roles(UserRole.ADMIN)),
    db: Session = Depends(dep_db),
):
    user_repo = UserRepository(db)
    doc_repo = DocumentRepository(db)
    return {
        "total_users": user_repo.count(),
        "total_documents": doc_repo.count(),
    }
