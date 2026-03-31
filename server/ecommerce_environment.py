"""OpenEnv-compliant CustomerSupportEnv wrapping the shared simulator."""

from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportAction, SupportObservation
    from ..env import get_simulator, InvalidActionError
except ImportError:
    from models import SupportAction, SupportObservation
    from env import get_simulator, InvalidActionError


class CustomerSupportEnv(Environment):
    """Customer support ticket environment (openenv.core compliant)."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        self._sim = get_simulator()

    def reset(self) -> SupportObservation:
        self._sim.reset()
        return self._obs(reward=0.0, done=False)

    def step(self, action: SupportAction) -> SupportObservation:  # type: ignore[override]
        try:
            _state, reward, done = self._sim.step(action.action)
        except InvalidActionError:
            raise
        return self._obs(reward=reward, done=done)

    @property
    def state(self) -> State:
        return State(
            episode_id=self._sim.episode_id,
            step_count=len(self._sim.history),
        )

    def _obs(self, reward: float, done: bool) -> SupportObservation:
        s = self._sim.state
        return SupportObservation(
            ticket_type=str(s.get("ticket_type", "damaged_item")),
            ticket_subject=str(s.get("ticket_subject", "")),
            ticket_description=str(s.get("ticket_description", "")),
            customer_name=str(s.get("customer_name", "")),
            customer_tier=str(s.get("customer_tier", "regular")),
            order_value=float(s.get("order_value", 0.0)),
            sentiment=float(s.get("sentiment", 0.3)),
            investigated=bool(s.get("investigated", False)),
            refund_offered=bool(s.get("refund_offered", False)),
            exchange_offered=bool(s.get("exchange_offered", False)),
            discount_applied=bool(s.get("discount_applied", False)),
            escalated=bool(s.get("escalated", False)),
            resolved=bool(s.get("resolved", False)),
            satisfaction_score=float(s.get("satisfaction_score", 0.0)),
            reward=reward,
            done=done,
        )


# Alias used by server/app.py
EcommerceEnv = CustomerSupportEnv
