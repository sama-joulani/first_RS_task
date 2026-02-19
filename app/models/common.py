from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    offset: int
    limit: int


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict | None = None


class SuccessResponse(BaseModel):
    message: str


class MetadataFilter(BaseModel):
    field: str
    operator: str = "eq"  # eq, ne, gt, gte, lt, lte, in
    value: str | int | float | list[str]
