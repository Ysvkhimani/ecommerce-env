"""Pydantic models for the HTTP API and environment."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class EcommerceAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["add_item", "apply_coupon", "checkout", "pay"] = Field(
        ...,
        description="Cart action to execute",
    )
    metadata: dict = Field(default_factory=dict)


class EcommerceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cart: List[str] = Field(default_factory=list)
    total: float = 0.0
    coupon_applied: bool = False
    payment_done: bool = False
    order_status: str = "incomplete"
    reward: float | None = None
    done: bool = False
    metadata: dict = Field(default_factory=dict)


class EcommerceEnvState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: Optional[str] = None
    step_count: int = 0
    cart: List[str] = Field(default_factory=list)
    total: float = 0.0
    coupon_applied: bool = False
    payment_done: bool = False
    order_status: str = "incomplete"
    history: List[str] = Field(default_factory=list)


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: dict
