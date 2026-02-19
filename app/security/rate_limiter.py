from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import Settings, get_settings
from app.models.auth import UserRole


ROLE_RATE_LIMITS: dict[UserRole, str] = {
    UserRole.VIEWER: "30/minute",
    UserRole.CONTRIBUTOR: "60/minute",
    UserRole.ADMIN: "120/minute",
}


class RateLimiter:
    """Wraps slowapi Limiter with role-based rate limits."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[self._settings.rate_limit_default],
        )

    @property
    def limiter(self) -> Limiter:
        return self._limiter

    def get_limit_for_role(self, role: UserRole) -> str:
        return ROLE_RATE_LIMITS.get(role, self._settings.rate_limit_default)
