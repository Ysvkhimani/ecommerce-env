"""Pydantic models for the Customer Support Agent environment."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class SupportAction(Action):
    """One of nine discrete actions the support agent can take."""

    action: Literal[
        "acknowledge",
        "investigate",
        "offer_refund",
        "offer_exchange",
        "apply_discount",
        "send_update",
        "escalate",
        "request_info",
        "resolve",
    ] = Field(..., description="Support action to execute")


class SupportObservation(Observation):
    """Full observation after each action — includes ticket context and live state."""

    # Ticket context
    ticket_id: str = Field(default="", description="Unique ticket identifier")
    ticket_type: str = Field(default="", description="damaged_item / wrong_item / missing_item / late_delivery / billing_issue")
    ticket_subject: str = Field(default="", description="Ticket subject line")
    ticket_description: str = Field(default="", description="Full customer message")
    customer_name: str = Field(default="", description="Customer name")
    customer_tier: str = Field(default="regular", description="vip / regular / new")
    order_value: float = Field(default=0.0, description="Affected order value ($)")
    correct_resolutions: List[str] = Field(default_factory=list, description="Action(s) that correctly resolve this ticket type")

    # Live episode state
    sentiment: float = Field(default=0.3, description="Customer sentiment 0.0 (angry) → 1.0 (happy)")
    investigated: bool = Field(default=False)
    refund_offered: bool = Field(default=False)
    exchange_offered: bool = Field(default=False)
    discount_applied: bool = Field(default=False)
    update_sent: bool = Field(default=False)
    escalated: bool = Field(default=False)
    resolved: bool = Field(default=False)
    satisfaction_score: float = Field(default=0.0)
    correct_resolution_used: bool = Field(default=False, description="Whether the right resolution action was used for this ticket type")

    # Conversation
    customer_response: str = Field(default="", description="Customer's latest response message")

    # done, reward, metadata inherited from Observation


class SupportEnvState(State):
    """Full internal state including action history."""

    ticket_id: str = ""
    ticket_type: str = ""
    ticket_subject: str = ""
    ticket_description: str = ""
    customer_name: str = ""
    customer_tier: str = "regular"
    order_value: float = 0.0
    correct_resolutions: List[str] = Field(default_factory=list)
    sentiment: float = 0.3
    investigated: bool = False
    refund_offered: bool = False
    exchange_offered: bool = False
    discount_applied: bool = False
    update_sent: bool = False
    escalated: bool = False
    resolved: bool = False
    satisfaction_score: float = 0.0
    correct_resolution_used: bool = False
    customer_response: str = ""
    history: List[str] = Field(default_factory=list)


# Backward-compat aliases
EcommerceAction = SupportAction
EcommerceObservation = SupportObservation
EcommerceEnvState = SupportEnvState
