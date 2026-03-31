"""Customer Support environment — wraps the shared simulator."""

from __future__ import annotations

import logging
from typing import Optional

from env import InvalidActionError, get_simulator
from models import SupportAction, SupportEnvState, SupportObservation

logger = logging.getLogger(__name__)


class CustomerSupportEnvironment:
    def __init__(self) -> None:
        self._sim = get_simulator()

    def reset(self, **kwargs: object) -> SupportObservation:
        try:
            self._sim.reset()
        except Exception as e:
            raise RuntimeError("Failed to reset environment") from e
        return self._observation(reward=0.0, done=False)

    def step(self, action: SupportAction, **kwargs: object) -> SupportObservation:
        try:
            _s, reward, done = self._sim.step(action.action)
        except InvalidActionError:
            raise
        except Exception as e:
            raise RuntimeError("Environment step failed") from e
        return self._observation(reward=reward, done=done)

    @property
    def state(self) -> SupportEnvState:
        s = self._sim.state
        return SupportEnvState(
            episode_id=self._sim.episode_id,
            step_count=len(self._sim.history),
            ticket_id=str(s.get("ticket_id", "")),
            ticket_type=str(s.get("ticket_type", "")),
            ticket_subject=str(s.get("ticket_subject", "")),
            ticket_description=str(s.get("ticket_description", "")),
            customer_name=str(s.get("customer_name", "")),
            customer_tier=str(s.get("customer_tier", "regular")),
            order_value=float(s.get("order_value", 0.0)),
            correct_resolutions=list(s.get("correct_resolutions", [])),
            sentiment=float(s.get("sentiment", 0.3)),
            investigated=bool(s.get("investigated", False)),
            refund_offered=bool(s.get("refund_offered", False)),
            exchange_offered=bool(s.get("exchange_offered", False)),
            discount_applied=bool(s.get("discount_applied", False)),
            update_sent=bool(s.get("update_sent", False)),
            escalated=bool(s.get("escalated", False)),
            resolved=bool(s.get("resolved", False)),
            satisfaction_score=float(s.get("satisfaction_score", 0.0)),
            correct_resolution_used=bool(s.get("correct_resolution_used", False)),
            customer_response=str(s.get("customer_response", "")),
            history=list(self._sim.history),
        )

    def _observation(self, reward: float, done: bool) -> SupportObservation:
        s = self._sim.state
        return SupportObservation(
            ticket_id=str(s.get("ticket_id", "")),
            ticket_type=str(s.get("ticket_type", "")),
            ticket_subject=str(s.get("ticket_subject", "")),
            ticket_description=str(s.get("ticket_description", "")),
            customer_name=str(s.get("customer_name", "")),
            customer_tier=str(s.get("customer_tier", "regular")),
            order_value=float(s.get("order_value", 0.0)),
            correct_resolutions=list(s.get("correct_resolutions", [])),
            sentiment=float(s.get("sentiment", 0.3)),
            investigated=bool(s.get("investigated", False)),
            refund_offered=bool(s.get("refund_offered", False)),
            exchange_offered=bool(s.get("exchange_offered", False)),
            discount_applied=bool(s.get("discount_applied", False)),
            update_sent=bool(s.get("update_sent", False)),
            escalated=bool(s.get("escalated", False)),
            resolved=bool(s.get("resolved", False)),
            satisfaction_score=float(s.get("satisfaction_score", 0.0)),
            correct_resolution_used=bool(s.get("correct_resolution_used", False)),
            customer_response=str(s.get("customer_response", "")),
            reward=reward,
            done=done,
        )


EcommerceEnvironment = CustomerSupportEnvironment
