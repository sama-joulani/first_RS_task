from __future__ import annotations

from app.models.auth import UserRole

ROLE_PERMISSIONS: dict[UserRole, set[str]] = {
    UserRole.VIEWER: {"search", "chat", "view_documents"},
    UserRole.CONTRIBUTOR: {"search", "chat", "view_documents", "upload_documents", "delete_own_documents"},
    UserRole.ADMIN: {
        "search", "chat", "view_documents", "upload_documents",
        "delete_own_documents", "delete_any_documents", "manage_users",
        "manage_all_documents", "view_admin",
    },
}


class SecurityManager:
    """Enforces RBAC by mapping roles to permission sets."""

    def __init__(self):
        self._permissions = ROLE_PERMISSIONS

    def check_permission(self, role: UserRole, permission: str) -> bool:
        allowed = self._permissions.get(role, set())
        return permission in allowed

    def get_permissions(self, role: UserRole) -> set[str]:
        return self._permissions.get(role, set())

    def has_any_role(self, user_role: UserRole, required_roles: list[UserRole]) -> bool:
        return user_role in required_roles
