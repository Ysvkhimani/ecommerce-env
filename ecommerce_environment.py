"""E-commerce environment backed by the shared process simulator."""

from __future__ import annotations

import logging
from typing import Optional

from env import InvalidActionError, get_simulator
from models import EcommerceAction, EcommerceEnvState, EcommerceObservation

logger = logging.getLogger(__name__)


class EcommerceEnvironment:
    """Cart RL-style env: reset / step / state (Gradio + scripts use the same instance)."""

    def __init__(self) -> None:
        self._sim = get_simulator()
        self._state = self._state_from_sim()

    def _state_from_sim(self) -> EcommerceEnvState:
        sid = self._sim.episode_id or ""
        s = self._sim.state
        return EcommerceEnvState(
            episode_id=sid or None,
            step_count=len(self._sim.history),
            cart=list(s.get("cart", [])),
            total=float(s.get("total", 0)),
            coupon_applied=bool(s.get("coupon_applied", False)),
            payment_done=bool(s.get("payment_done", False)),
            order_status=str(s.get("order_status", "incomplete")),
            history=list(self._sim.history),
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: object,
    ) -> EcommerceObservation:
        try:
            self._sim.reset()
        except Exception as e:
            logger.exception("reset failed")
            raise RuntimeError("Failed to reset environment") from e
        self._state = self._state_from_sim()
        return self._observation(reward=0.0, done=False)

    def step(
        self,
        action: EcommerceAction,
        timeout_s: Optional[float] = None,
        **kwargs: object,
    ) -> EcommerceObservation:
        try:
            _s, reward, done = self._sim.step(action.action)
        except InvalidActionError:
            raise
        except Exception as e:
            logger.exception("step failed for action=%s", getattr(action, "action", action))
            raise RuntimeError("Environment step failed") from e
        self._state = self._state_from_sim()
        return self._observation(reward=reward, done=done)

    @property
    def state(self) -> EcommerceEnvState:
        self._state = self._state_from_sim()
        return self._state

    def _observation(self, reward: float, done: bool) -> EcommerceObservation:
        s = self._sim.state
        return EcommerceObservation(
            cart=list(s.get("cart", [])),
            total=float(s.get("total", 0)),
            coupon_applied=bool(s.get("coupon_applied", False)),
            payment_done=bool(s.get("payment_done", False)),
            order_status=str(s.get("order_status", "incomplete")),
            reward=reward,
            done=done,
        )
