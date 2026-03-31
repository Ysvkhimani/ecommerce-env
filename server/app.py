"""
OpenEnv-compliant FastAPI server — E-commerce Customer Support Agent Environment.

Standard endpoints (openenv.core.create_app):
    POST /reset     — start new episode
    POST /step      — execute action
    GET  /state     — current state
    GET  /schema    — action/observation schema
    WS   /ws        — WebSocket session

Additional hackathon endpoints:
    GET  /          — health check
    GET  /tasks     — task list + action schema
    GET  /grader    — grader scores (easy / medium / hard)
    POST /baseline  — run optimal policy, return episode + scores

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import sys
from typing import Any

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from openenv.core.env_server.http_server import create_app

try:
    from .ecommerce_environment import CustomerSupportEnv
    from ..models import SupportAction, SupportObservation
    from ..grader import grade_easy, grade_hard, grade_medium
    from ..env import OPTIMAL_POLICY
except ImportError:
    from server.ecommerce_environment import CustomerSupportEnv
    from models import SupportAction, SupportObservation
    from grader import grade_easy, grade_hard, grade_medium
    from env import OPTIMAL_POLICY

# ---------------------------------------------------------------------------
# OpenEnv base app
# ---------------------------------------------------------------------------
app = create_app(
    CustomerSupportEnv,
    SupportAction,
    SupportObservation,
    env_name="ecommerce-support-env",
    max_concurrent_envs=1,
)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS = [
    {
        "id": "easy",
        "description": "Resolve the customer support ticket — any resolution counts",
        "difficulty": "easy",
        "scoring": "1.0 if resolved, else 0.0",
    },
    {
        "id": "medium",
        "description": "Resolve the ticket with meaningful customer satisfaction",
        "difficulty": "medium",
        "scoring": "satisfaction_score (0.0–1.0) if resolved, partial credit for low satisfaction",
    },
    {
        "id": "hard",
        "description": (
            "Resolve efficiently: satisfaction ≥ 0.8, ≤ 5 steps, no escalation. "
            "Optimal: acknowledge → investigate → offer_refund → resolve"
        ),
        "difficulty": "hard",
        "scoring": "1.0 (efficient+satisfied), 0.7 (satisfied, no escalation), 0.5 (resolved, clean), 0.2 (escalated)",
    },
]

ACTION_SCHEMA = {
    "action": {
        "type": "string",
        "enum": [
            "acknowledge",
            "investigate",
            "offer_refund",
            "offer_exchange",
            "apply_discount",
            "escalate",
            "request_info",
            "resolve",
        ],
        "description": "Support action to execute",
    },
    "metadata": {"type": "object", "description": "Optional metadata", "default": {}},
}

# ---------------------------------------------------------------------------
# Hackathon-required endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": "ecommerce-customer-support", "docs": "/docs"}


@app.get("/tasks")
async def get_tasks() -> dict[str, Any]:
    return {"tasks": TASKS, "action_schema": ACTION_SCHEMA}


@app.get("/grader")
async def get_grader() -> dict[str, float]:
    return {
        "easy": grade_easy(),
        "medium": grade_medium(),
        "hard": grade_hard(),
    }


@app.post("/baseline")
async def run_baseline() -> dict[str, Any]:
    """Run the optimal policy and return episode + per-task scores."""
    env = CustomerSupportEnv()
    env.reset()
    episode: list[dict[str, Any]] = []
    for action in OPTIMAL_POLICY:
        act = SupportAction(action=action)
        obs = env.step(act)
        episode.append({"action": action, "sentiment": obs.sentiment, "reward": obs.reward, "done": obs.done})

    return {
        "policy": " → ".join(OPTIMAL_POLICY),
        "episode": episode,
        "scores": {
            "easy": grade_easy(),
            "medium": grade_medium(),
            "hard": grade_hard(),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.host == "0.0.0.0" and args.port == 7860:
        main()
    else:
        main(host=args.host, port=args.port)
