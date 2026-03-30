"""E-commerce cart simulation (shared by HTTP env, client, and grader)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


class EcommerceSimulator:
    """Single-episode cart simulator with reward and done signals."""

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {}
        self.history: List[str] = []
        self.episode_id: Optional[str] = None

    def reset(self) -> Dict[str, Any]:
        self.state.clear()
        self.state.update(
            {
                "cart": [],
                "total": 0,
                "coupon_applied": False,
                "payment_done": False,
                "order_status": "incomplete",
            }
        )
        self.history.clear()
        self.episode_id = str(uuid4())
        return dict(self.state)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool]:
        reward = 0.0
        done = False

        if action == "add_item":
            self.state["cart"].append("item")
            self.state["total"] += 100
            reward = 0.2

        elif action == "apply_coupon":
            if not self.state["coupon_applied"]:
                self.state["total"] *= 0.9
                self.state["coupon_applied"] = True
                reward = 0.3
            else:
                reward = -0.1

        elif action == "checkout":
            reward = 0.3 if self.state["cart"] else -0.5

        elif action == "pay":
            if self.state["cart"]:
                self.state["payment_done"] = True
                self.state["order_status"] = "completed"
                reward = 1.0
                done = True
            else:
                reward = -1.0

        self.history.append(action)
        return dict(self.state), reward, done


# Single process-wide simulator: shared by OpenEnv `EcommerceEnvironment`, grader, and scripts.
_sim = EcommerceSimulator()


def get_simulator() -> EcommerceSimulator:
    """Return the shared cart simulator (one episode store per Space process)."""
    return _sim


# Grader and callers use these names; they alias the live simulator containers.
state: Dict[str, Any] = _sim.state
history: List[str] = _sim.history


def reset() -> Dict[str, Any]:
    return _sim.reset()


def step(action: str) -> Tuple[Dict[str, Any], float, bool]:
    return _sim.step(action)
