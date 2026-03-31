"""Customer support ticket simulator — shared by environment, grader, and scripts.

Scenario: An AI agent acts as a customer service representative for an e-commerce store.
A customer has received a damaged item and submitted a support ticket. The agent must
resolve it efficiently while maximising customer satisfaction.

The episode always starts with the same ticket so grader scores are fully reproducible.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
ALLOWED_ACTIONS = frozenset({
    "acknowledge",    # Acknowledge the customer's issue            (+sentiment, +reward)
    "investigate",    # Look up the order details                   (+sentiment, +reward, unlocks better options)
    "offer_refund",   # Issue a full refund                         (big +sentiment/reward if investigated)
    "offer_exchange", # Send a replacement item                     (good alternative to refund)
    "apply_discount", # Give 10% off next order as goodwill        (+small sentiment/reward)
    "escalate",       # Transfer to senior agent                    (-sentiment, -reward)
    "request_info",   # Ask customer for more info (they gave it)  (-sentiment, -reward)
    "resolve",        # Close the ticket                            (reward = final sentiment, done=True)
})

# ---------------------------------------------------------------------------
# Ticket scenario (fixed for reproducibility)
# ---------------------------------------------------------------------------
TICKET = {
    "ticket_id": "TKT-2024-001",
    "type": "damaged_item",
    "subject": "My laptop arrived with a cracked screen",
    "description": (
        "I ordered a laptop (Order #ORD-98765, $999) but it arrived with a cracked screen. "
        "I have photos as proof. I'd like a refund or replacement please."
    ),
    "customer_name": "Alex",
    "customer_tier": "regular",   # regular | vip | new
    "order_value": 999.0,
    "initial_sentiment": 0.3,     # 0.0 = very angry, 1.0 = very happy
}

# Optimal action sequence for this ticket:
#   acknowledge → investigate → offer_refund → resolve
#   → satisfaction = 0.3 + 0.15 + 0.05 + 0.40 = 0.90  (hard task scores 1.0)
OPTIMAL_POLICY = ["acknowledge", "investigate", "offer_refund", "resolve"]


class InvalidActionError(ValueError):
    def __init__(self, action: str) -> None:
        self.action = action
        super().__init__(f"Invalid action {action!r}. Allowed: {sorted(ALLOWED_ACTIONS)}")


class CustomerSupportSimulator:
    """Single-episode customer support simulator with reward and done signals."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.history: List[str] = []
        self.episode_id: Optional[str] = None

    def reset(self) -> Dict[str, Any]:
        # Update IN PLACE so module-level aliases (grader.py) stay valid.
        self.state.clear()
        self.state.update({
            "ticket_type": TICKET["type"],
            "ticket_subject": TICKET["subject"],
            "ticket_description": TICKET["description"],
            "customer_name": TICKET["customer_name"],
            "customer_tier": TICKET["customer_tier"],
            "order_value": TICKET["order_value"],
            "sentiment": TICKET["initial_sentiment"],
            "investigated": False,
            "refund_offered": False,
            "exchange_offered": False,
            "discount_applied": False,
            "escalated": False,
            "resolved": False,
            "satisfaction_score": 0.0,
        })
        self.history.clear()
        self.episode_id = str(uuid4())
        return copy.deepcopy(self.state)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool]:
        if action not in ALLOWED_ACTIONS:
            raise InvalidActionError(action)

        s = self.state
        reward = 0.0
        done = False

        if action == "acknowledge":
            if "acknowledge" not in self.history:
                s["sentiment"] = min(1.0, s["sentiment"] + 0.15)
                reward = 0.1
            else:
                reward = -0.05  # penalise repeating

        elif action == "investigate":
            if not s["investigated"]:
                s["investigated"] = True
                s["sentiment"] = min(1.0, s["sentiment"] + 0.05)
                reward = 0.1
            else:
                reward = -0.05

        elif action == "offer_refund":
            if not s["refund_offered"]:
                s["refund_offered"] = True
                if s["investigated"]:
                    # Investigated first → confident, proper resolution
                    s["sentiment"] = min(1.0, s["sentiment"] + 0.40)
                    reward = 0.5
                else:
                    # Blind refund — still helps, but less optimal
                    s["sentiment"] = min(1.0, s["sentiment"] + 0.20)
                    reward = 0.2
            else:
                reward = -0.1

        elif action == "offer_exchange":
            if not s["exchange_offered"]:
                s["exchange_offered"] = True
                if s["investigated"]:
                    s["sentiment"] = min(1.0, s["sentiment"] + 0.30)
                    reward = 0.4
                else:
                    s["sentiment"] = min(1.0, s["sentiment"] + 0.10)
                    reward = 0.1
            else:
                reward = -0.1

        elif action == "apply_discount":
            if not s["discount_applied"]:
                s["discount_applied"] = True
                s["sentiment"] = min(1.0, s["sentiment"] + 0.10)
                reward = 0.1
            else:
                reward = -0.05

        elif action == "escalate":
            if not s["escalated"]:
                s["escalated"] = True
                # Escalation frustrates the customer (they wanted direct help)
                s["sentiment"] = max(0.0, s["sentiment"] - 0.10)
                reward = -0.3
            else:
                reward = -0.2

        elif action == "request_info":
            # Customer already provided all info in the ticket — this is unhelpful
            s["sentiment"] = max(0.0, s["sentiment"] - 0.10)
            reward = -0.1

        elif action == "resolve":
            s["resolved"] = True
            s["satisfaction_score"] = round(s["sentiment"], 4)
            reward = s["sentiment"]   # 0.0–1.0 based on how happy the customer is
            done = True

        self.history.append(action)
        return copy.deepcopy(s), reward, done


# ---------------------------------------------------------------------------
# Process-wide shared simulator (one store per Space process)
# ---------------------------------------------------------------------------
_sim = CustomerSupportSimulator()


def get_simulator() -> CustomerSupportSimulator:
    return _sim


# Module-level aliases used by grader.py
state: Dict[str, Any] = _sim.state
history: List[str] = _sim.history


def reset() -> Dict[str, Any]:
    return _sim.reset()


def step(action: str) -> Tuple[Dict[str, Any], float, bool]:
    return _sim.step(action)
