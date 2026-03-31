"""Pydantic models for the Customer Support Agent environment."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class SupportAction(Action):
    """One of eight discrete actions the support agent can take."""

    action: Literal[
        "acknowledge",
        "investigate",
        "offer_refund",
        "offer_exchange",
        "apply_discount",
        "escalate",
        "request_info",
        "resolve",
    ] = Field(..., description="Support action to execute")
    # `metadata` inherited from Action


class SupportObservation(Observation):
    """Observation returned after each support action."""

    # Ticket context (static throughout episode)
    ticket_type: str = Field(default="damaged_item", description="Type of customer issue")
    ticket_subject: str = Field(default="", description="Ticket subject line")
    ticket_description: str = Field(default="", description="Full customer message")
    customer_name: str = Field(default="", description="Customer's name")
    customer_tier: str = Field(default="regular", description="Customer tier: vip/regular/new")
    order_value: float = Field(default=0.0, description="Value of the affected order ($)")

    # Mutable episode state
    sentiment: float = Field(default=0.3, description="Customer sentiment 0.0 (angry) → 1.0 (happy)")
    investigated: bool = Field(default=False, description="Whether order was looked up")
    refund_offered: bool = Field(default=False, description="Whether a full refund was offered")
    exchange_offered: bool = Field(default=False, description="Whether a replacement was offered")
    discount_applied: bool = Field(default=False, description="Whether a goodwill discount was given")
    escalated: bool = Field(default=False, description="Whether ticket was escalated")
    resolved: bool = Field(default=False, description="Whether ticket is closed")
    satisfaction_score: float = Field(default=0.0, description="Final satisfaction score (0.0–1.0)")
    # `done`, `reward`, `metadata` inherited from Observation


class SupportEnvState(State):
    """Full internal state, richer than the observation (includes history)."""

    ticket_type: str = "damaged_item"
    ticket_subject: str = ""
    ticket_description: str = ""
    customer_name: str = ""
    customer_tier: str = "regular"
    order_value: float = 0.0
    sentiment: float = 0.3
    investigated: bool = False
    refund_offered: bool = False
    exchange_offered: bool = False
    discount_applied: bool = False
    escalated: bool = False
    resolved: bool = False
    satisfaction_score: float = 0.0
    history: List[str] = Field(default_factory=list)
    # `episode_id`, `step_count` inherited from State


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (used by server/app.py create_app call)
# ---------------------------------------------------------------------------
EcommerceAction = SupportAction
EcommerceObservation = SupportObservation
EcommerceEnvState = SupportEnvState
