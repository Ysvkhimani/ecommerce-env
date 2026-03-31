"""Customer Support environment — wraps the shared simulator for the HTTP API."""

from __future__ import annotations

import logging
from typing import Optional

from env import InvalidActionError, get_simulator
from models import SupportAction, SupportEnvState, SupportObservation

logger = logging.getLogger(__name__)


class CustomerSupportEnvironment:
    """Support ticket RL environment: reset / step / state."""

    def __init__(self) -> None:
        self._sim = get_simulator()
        self._state = self._state_from_sim()

    def _state_from_sim(self) -> SupportEnvState:
        s = self._sim.state
        return SupportEnvState(
            episode_id=self._sim.episode_id or None,
            step_count=len(self._sim.history),
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
            history=list(self._sim.history),
        )

    def reset(self, **kwargs: object) -> SupportObservation:
        try:
            self._sim.reset()
        except Exception as e:
            raise RuntimeError("Failed to reset environment") from e
        self._state = self._state_from_sim()
        return self._observation(reward=0.0, done=False)

    def step(self, action: SupportAction, **kwargs: object) -> SupportObservation:
        try:
            _s, reward, done = self._sim.step(action.action)
        except InvalidActionError:
            raise
        except Exception as e:
            raise RuntimeError("Environment step failed") from e
        self._state = self._state_from_sim()
        return self._observation(reward=reward, done=done)

    @property
    def state(self) -> SupportEnvState:
        self._state = self._state_from_sim()
        return self._state

    def _observation(self, reward: float, done: bool) -> SupportObservation:
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


# Backward-compat alias
EcommerceEnvironment = CustomerSupportEnvironment
