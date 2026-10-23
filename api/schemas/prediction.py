"""Validated API requests for existing customers or complete feature records."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """An existing customer ID, optionally with a complete feature override."""

    model_config = ConfigDict(extra="forbid")
    customer_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    features: dict[str, float | str] | None = None
