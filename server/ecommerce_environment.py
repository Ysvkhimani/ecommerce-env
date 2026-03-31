"""OpenEnv-compliant EcommerceEnv wrapping the shared cart simulator."""

from __future__ import annotations

import sys
import os

# Make the project root importable when running as `uvicorn server.app:app`
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EcommerceAction, EcommerceObservation
    from ..env import get_simulator, InvalidActionError
except ImportError:
    from models import EcommerceAction, EcommerceObservation
    from env import get_simulator, InvalidActionError


class EcommerceEnv(Environment):
    """E-commerce cart environment backed by the shared process-level simulator."""

    # All instances share the global _sim, so concurrent sessions are not safe.
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        self._sim = get_simulator()

    def reset(self) -> EcommerceObservation:
        self._sim.reset()
        return self._obs(reward=0.0, done=False)

    def step(self, action: EcommerceAction) -> EcommerceObservation:  # type: ignore[override]
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

    def _obs(self, reward: float, done: bool) -> EcommerceObservation:
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
