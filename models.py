"""Pydantic models for the HTTP API and environment.

EcommerceAction and EcommerceObservation extend the openenv.core base types so
the server passes openenv validate and create_app() works correctly.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class EcommerceAction(Action):
    """Cart action — one of four discrete steps in the purchase flow."""

    action: Literal["add_item", "apply_coupon", "checkout", "pay"] = Field(
        ...,
        description="Cart action to execute",
    )
    # `metadata` is inherited from Action


class EcommerceObservation(Observation):
    """Observation returned after each step."""

    cart: List[str] = Field(default_factory=list, description="Items in the cart")
    total: float = Field(default=0.0, description="Cart total (after any coupon)")
    coupon_applied: bool = Field(default=False, description="Whether a coupon is applied")
    payment_done: bool = Field(default=False, description="Whether payment is complete")
    order_status: str = Field(default="incomplete", description="Order status")
    # `done`, `reward`, `metadata` are inherited from Observation


class EcommerceEnvState(State):
    """Full internal state, richer than the public observation."""

    cart: List[str] = Field(default_factory=list)
    total: float = 0.0
    coupon_applied: bool = False
    payment_done: bool = False
    order_status: str = "incomplete"
    history: List[str] = Field(default_factory=list)
    # `episode_id`, `step_count` are inherited from State


class StepRequest(Action):
    """Generic step request wrapper (kept for backward compatibility)."""

    action: dict = Field(default_factory=dict)
