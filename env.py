"""Customer support simulator with 5 ticket scenarios and live customer responses.

Each episode randomly selects one of 5 real-world ticket types. The agent must
read the ticket, infer the correct resolution type, and execute it efficiently.

Ticket scenarios:
  1. damaged_item   → correct resolution: offer_refund / offer_exchange
  2. wrong_item     → correct resolution: offer_exchange / offer_refund
  3. missing_item   → correct resolution: offer_refund / offer_exchange
  4. late_delivery  → correct resolution: send_update (+ apply_discount)
  5. billing_issue  → correct resolution: investigate then resolve (no item action)

This forces the LLM agent to actually read and understand the ticket before acting.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
ALLOWED_ACTIONS = frozenset({
    "acknowledge",    # Acknowledge the issue
    "investigate",    # Look up order/account details
    "offer_refund",   # Issue full refund (best for damaged/missing/wrong items)
    "offer_exchange", # Send replacement (good for wrong/damaged items)
    "apply_discount", # 10% off next order as goodwill
    "send_update",    # Send delivery status update (best for late_delivery)
    "escalate",       # Transfer to senior agent — penalised
    "request_info",   # Ask for info already in ticket — penalised
    "resolve",        # Close the ticket
})

# ---------------------------------------------------------------------------
# Ticket scenarios — seeded random ensures reproducible episode sequences
# ---------------------------------------------------------------------------
_RNG = random.Random(42)  # deterministic across runs

TICKET_SCENARIOS = [
    {
        "ticket_id": "TKT-001",
        "type": "damaged_item",
        "subject": "My laptop arrived with a cracked screen",
        "description": (
            "I ordered a laptop (Order #ORD-98765, $999) but it arrived "
            "with a cracked screen. I have photos. I'd like a refund or replacement."
        ),
        "customer_name": "Alex",
        "customer_tier": "regular",
        "order_value": 999.0,
        "initial_sentiment": 0.3,
        "correct_resolutions": {"offer_refund", "offer_exchange"},
        "opening_message": (
            "Hi, I received my laptop today but the screen is completely cracked. "
            "This is unacceptable — I paid $999 for this. Please help."
        ),
        "responses": {
            "acknowledge":    "Thank you for getting back to me so quickly.",
            "investigate":    "My order is #ORD-98765, purchased last Tuesday.",
            "offer_refund":   "A full refund sounds great. How long will it take?",
            "offer_exchange": "I'd accept a replacement but please check it first.",
            "apply_discount": "I appreciate it, but I still need the main issue fixed.",
            "send_update":    "The delivery already happened — the problem is the item is damaged.",
            "escalate":       "Why am I being passed around? This is really frustrating!",
            "request_info":   "I already gave you all the information in my ticket!",
            "resolve":        "Thank you, I'll update my review once I receive the refund. ⭐⭐⭐⭐",
        },
    },
    {
        "ticket_id": "TKT-002",
        "type": "wrong_item",
        "subject": "I received the wrong product",
        "description": (
            "I ordered a blue smartwatch (Order #ORD-55432, $249) but received "
            "a red fitness band instead. Please send the correct item."
        ),
        "customer_name": "Jordan",
        "customer_tier": "vip",
        "order_value": 249.0,
        "initial_sentiment": 0.35,
        "correct_resolutions": {"offer_exchange", "offer_refund"},
        "opening_message": (
            "This is completely wrong — I ordered a smartwatch and got a fitness band. "
            "As a VIP customer I expect better. Fix this immediately."
        ),
        "responses": {
            "acknowledge":    "Finally a response. I've been waiting since yesterday.",
            "investigate":    "Order #ORD-55432. I ordered the BLUE smartwatch, model SW-X200.",
            "offer_refund":   "I'd prefer the correct item but a refund is acceptable.",
            "offer_exchange": "Yes! Please send the correct blue smartwatch. Thank you.",
            "apply_discount": "A discount is nice but I really just want the right product.",
            "send_update":    "The delivery already came — it just had the wrong item inside.",
            "escalate":       "I'm a VIP customer and I'm being transferred? Unbelievable.",
            "request_info":   "It's all in the ticket. Order #ORD-55432. Blue smartwatch.",
            "resolve":        "Great, I'll watch for the replacement. Thanks for fixing this. ⭐⭐⭐⭐⭐",
        },
    },
    {
        "ticket_id": "TKT-003",
        "type": "missing_item",
        "subject": "My package was marked delivered but never arrived",
        "description": (
            "Order #ORD-77201 ($149) shows delivered 2 days ago but I never received it. "
            "I checked with neighbours and the front desk. It's simply not here."
        ),
        "customer_name": "Sam",
        "customer_tier": "new",
        "order_value": 149.0,
        "initial_sentiment": 0.25,
        "correct_resolutions": {"offer_refund", "offer_exchange"},
        "opening_message": (
            "My package says delivered but it never arrived. "
            "This is my first order here and I'm very disappointed."
        ),
        "responses": {
            "acknowledge":    "Thank you. I'm hoping you can sort this out.",
            "investigate":    "Order #ORD-77201. Tracking says delivered at 2:14 PM on Monday.",
            "offer_refund":   "A refund would be fine. I just want my money back.",
            "offer_exchange": "Yes please re-send! I still want the item.",
            "apply_discount": "Thanks but I'd rather have the refund or a resend.",
            "send_update":    "I know it was marked as delivered, but it's not here.",
            "escalate":       "Please just sort this out, I don't want to be transferred again.",
            "request_info":   "I've told you everything already. It simply wasn't there.",
            "resolve":        "Thank you. I'll give the store another chance. ⭐⭐⭐",
        },
    },
    {
        "ticket_id": "TKT-004",
        "type": "late_delivery",
        "subject": "Order still hasn't arrived — it's been 10 days",
        "description": (
            "I ordered headphones (Order #ORD-34876, $89) 10 days ago with "
            "5-day delivery. Still nothing. I need them for a work event tomorrow."
        ),
        "customer_name": "Casey",
        "customer_tier": "regular",
        "order_value": 89.0,
        "initial_sentiment": 0.2,
        "correct_resolutions": {"send_update", "apply_discount"},
        "opening_message": (
            "It's been 10 days and my order hasn't arrived. "
            "I paid for 5-day delivery. This is completely unacceptable."
        ),
        "responses": {
            "acknowledge":    "I hope you can tell me where my package actually is.",
            "investigate":    "Order #ORD-34876. Placed 10 days ago, 5-day delivery promised.",
            "offer_refund":   "If the item isn't coming, a refund would be okay I suppose.",
            "offer_exchange": "Sending a replacement makes sense if the original is lost.",
            "apply_discount": "That helps a bit — thanks. But please also tell me where it is.",
            "send_update":    "Thank you for checking! That's really helpful to know.",
            "escalate":       "I just want to know where my order is — why escalate?",
            "request_info":   "I gave you the order number. What more do you need?",
            "resolve":        "Thanks for the update and discount. I'll wait a bit longer. ⭐⭐⭐⭐",
        },
    },
    {
        "ticket_id": "TKT-005",
        "type": "billing_issue",
        "subject": "I was charged twice for the same order",
        "description": (
            "I see two identical charges of $199 on my card for Order #ORD-21098. "
            "I only placed one order. Please reverse the duplicate charge immediately."
        ),
        "customer_name": "Riley",
        "customer_tier": "vip",
        "order_value": 199.0,
        "initial_sentiment": 0.15,
        "correct_resolutions": {"investigate", "offer_refund"},  # Investigate to confirm, then refund the duplicate charge
        "opening_message": (
            "I was double-charged $199 on my card. This is a serious billing error. "
            "I need this reversed TODAY."
        ),
        "responses": {
            "acknowledge":    "Thank you for responding. I need this resolved urgently.",
            "investigate":    "Order #ORD-21098. Both charges appeared on Tuesday at 3:47 PM.",
            "offer_refund":   "Thank you! Please process the reversal of the duplicate $199 charge as soon as possible.",
            "offer_exchange": "I don't need an exchange — I need a billing correction.",
            "apply_discount": "I don't want a discount, I want my money back from the error.",
            "send_update":    "This isn't a delivery issue — it's a billing problem.",
            "escalate":       "Fine, but whoever takes over please fix the double charge.",
            "request_info":   "I already told you — two charges of $199 on Tuesday.",
            "resolve":        "Thank you for sorting out the billing. I'll monitor my statement. ⭐⭐⭐⭐",
        },
    },
]

# Optimal policies per ticket type (for baseline script)
OPTIMAL_POLICIES: Dict[str, List[str]] = {
    "damaged_item":  ["acknowledge", "investigate", "offer_refund", "resolve"],
    "wrong_item":    ["acknowledge", "investigate", "offer_exchange", "resolve"],
    "missing_item":  ["acknowledge", "investigate", "offer_refund", "resolve"],
    "late_delivery": ["acknowledge", "investigate", "send_update", "apply_discount", "resolve"],
    "billing_issue": ["acknowledge", "investigate", "offer_refund", "resolve"],
}

# Default for the baseline script (always reproducible)
OPTIMAL_POLICY = OPTIMAL_POLICIES["damaged_item"]


class InvalidActionError(ValueError):
    def __init__(self, action: str) -> None:
        self.action = action
        super().__init__(f"Invalid action {action!r}. Allowed: {sorted(ALLOWED_ACTIONS)}")


class CustomerSupportSimulator:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.history: List[str] = []
        self.episode_id: Optional[str] = None
        self._scenario: Dict[str, Any] = TICKET_SCENARIOS[0]

    def reset(self) -> Dict[str, Any]:
        self._scenario = _RNG.choice(TICKET_SCENARIOS)
        sc = self._scenario

        # Update IN PLACE so module-level aliases (grader.py) stay valid.
        self.state.clear()
        self.state.update({
            "ticket_id":           sc["ticket_id"],
            "ticket_type":         sc["type"],
            "ticket_subject":      sc["subject"],
            "ticket_description":  sc["description"],
            "customer_name":       sc["customer_name"],
            "customer_tier":       sc["customer_tier"],
            "order_value":         sc["order_value"],
            "correct_resolutions": list(sc["correct_resolutions"]),
            "opening_message":     sc["opening_message"],

            # Mutable episode state
            "sentiment":           sc["initial_sentiment"],
            "investigated":        False,
            "refund_offered":      False,
            "exchange_offered":    False,
            "discount_applied":    False,
            "update_sent":         False,
            "escalated":           False,
            "resolved":            False,
            "satisfaction_score":  0.0,
            "correct_resolution_used": False,
            "customer_response":   sc["opening_message"],
        })
        self.history.clear()
        self.episode_id = str(uuid4())
        return copy.deepcopy(self.state)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool]:
        if action not in ALLOWED_ACTIONS:
            raise InvalidActionError(action)

        s = self.state
        sc = self._scenario
        reward = 0.0
        done = False

        correct_res = sc["correct_resolutions"]

        if action == "acknowledge":
            if "acknowledge" not in self.history:
                s["sentiment"] = min(1.0, s["sentiment"] + 0.15)
                reward = 0.1
            else:
                reward = -0.05

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
                if "offer_refund" in correct_res:
                    s["correct_resolution_used"] = True
                    boost = 0.40 if s["investigated"] else 0.20
                    reward = 0.5 if s["investigated"] else 0.2
                else:
                    boost = 0.10   # not the best action for this ticket
                    reward = 0.1
                s["sentiment"] = min(1.0, s["sentiment"] + boost)
            else:
                reward = -0.1

        elif action == "offer_exchange":
            if not s["exchange_offered"]:
                s["exchange_offered"] = True
                if "offer_exchange" in correct_res:
                    s["correct_resolution_used"] = True
                    boost = 0.35 if s["investigated"] else 0.15
                    reward = 0.45 if s["investigated"] else 0.15
                else:
                    boost = 0.10
                    reward = 0.1
                s["sentiment"] = min(1.0, s["sentiment"] + boost)
            else:
                reward = -0.1

        elif action == "send_update":
            if not s["update_sent"]:
                s["update_sent"] = True
                if "send_update" in correct_res:
                    s["correct_resolution_used"] = True
                    s["sentiment"] = min(1.0, s["sentiment"] + 0.30)
                    reward = 0.4
                else:
                    # Wrong action for this ticket type
                    s["sentiment"] = max(0.0, s["sentiment"] - 0.05)
                    reward = -0.1
            else:
                reward = -0.05

        elif action == "apply_discount":
            if not s["discount_applied"]:
                s["discount_applied"] = True
                if "apply_discount" in correct_res:
                    s["correct_resolution_used"] = True
                    s["sentiment"] = min(1.0, s["sentiment"] + 0.20)
                    reward = 0.3
                else:
                    s["sentiment"] = min(1.0, s["sentiment"] + 0.08)
                    reward = 0.08
            else:
                reward = -0.05

        elif action == "escalate":
            # VIP customers react more negatively to escalation
            vip = s.get("customer_tier") == "vip"
            if not s["escalated"]:
                s["escalated"] = True
                penalty = 0.50 if vip else 0.30
                s["sentiment"] = max(0.0, s["sentiment"] - (0.15 if vip else 0.10))
                reward = -penalty
            else:
                reward = -0.4 if vip else -0.2

        elif action == "request_info":
            s["sentiment"] = max(0.0, s["sentiment"] - 0.10)
            reward = -0.1

        elif action == "resolve":
            # For billing_issue the correct resolution is just investigate+resolve
            if s["ticket_type"] == "billing_issue" and s["investigated"]:
                s["correct_resolution_used"] = True
            s["resolved"] = True
            s["satisfaction_score"] = round(s["sentiment"], 4)
            reward = s["sentiment"]
            done = True

        # Small per-step cost encourages efficient resolution
        reward -= 0.01

        # Update customer response
        s["customer_response"] = sc["responses"].get(action, "...")
        self.history.append(action)
        return copy.deepcopy(s), reward, done


# ---------------------------------------------------------------------------
# Process-wide shared simulator
# ---------------------------------------------------------------------------
_sim = CustomerSupportSimulator()


def get_simulator() -> CustomerSupportSimulator:
    return _sim


state: Dict[str, Any] = _sim.state
history: List[str] = _sim.history


def reset() -> Dict[str, Any]:
    return _sim.reset()


def step(action: str) -> Tuple[Dict[str, Any], float, bool]:
    return _sim.step(action)
