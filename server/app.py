"""
OpenEnv-compliant FastAPI server for the Ecommerce environment.

Standard endpoints (provided by openenv.core.create_app):
    POST /reset     — reset episode
    POST /step      — execute action
    GET  /state     — current state
    GET  /schema    — action/observation schema
    WS   /ws        — WebSocket session

Additional endpoints required by hackathon judges:
    GET  /tasks     — task list + action schema
    GET  /grader    — grader scores (easy / medium / hard)
    POST /baseline  — run optimal policy, return episode + scores

Gradio web UI is served at /web by openenv.core (built-in).

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
    # then open http://localhost:7860/web
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Make project root importable when running via `uvicorn server.app:app`
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from openenv.core.env_server.http_server import create_app

try:
    from .ecommerce_environment import EcommerceEnv
    from ..models import EcommerceAction, EcommerceObservation
    from ..grader import grade_easy, grade_hard, grade_medium
except ImportError:
    from server.ecommerce_environment import EcommerceEnv
    from models import EcommerceAction, EcommerceObservation
    from grader import grade_easy, grade_hard, grade_medium

# ---------------------------------------------------------------------------
# Create the base OpenEnv app (provides /reset /step /state /schema /ws /web)
# ---------------------------------------------------------------------------
app = create_app(
    EcommerceEnv,
    EcommerceAction,
    EcommerceObservation,
    env_name="ecommerce-env",
    max_concurrent_envs=1,
)

# ---------------------------------------------------------------------------
# Root health check (judges ping the Space URL and expect 200)
# ---------------------------------------------------------------------------


@app.get("/")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": "ecommerce-env", "ui": "/web", "docs": "/docs"}


# ---------------------------------------------------------------------------
# Task definitions (static metadata)
# ---------------------------------------------------------------------------
TASKS = [
    {
        "id": "easy",
        "description": "Complete a purchase: add at least one item and pay",
        "difficulty": "easy",
        "scoring": "1.0 if payment_done, else 0.0",
    },
    {
        "id": "medium",
        "description": "Complete a purchase with a coupon applied before paying",
        "difficulty": "medium",
        "scoring": "1.0 if paid+coupon, 0.5 if paid without coupon, else 0.0",
    },
    {
        "id": "hard",
        "description": "Complete purchase in exact order: add_item → apply_coupon → checkout → pay",
        "difficulty": "hard",
        "scoring": "1.0 if exact sequence, 0.5 if paid (wrong order), else 0.0",
    },
]

ACTION_SCHEMA = {
    "action": {
        "type": "string",
        "enum": ["add_item", "apply_coupon", "checkout", "pay"],
        "description": "Cart action to execute",
    },
    "metadata": {
        "type": "object",
        "description": "Optional extra metadata",
        "default": {},
    },
}

# ---------------------------------------------------------------------------
# Additional judge-required endpoints
# ---------------------------------------------------------------------------


@app.get("/tasks")
async def get_tasks() -> dict[str, Any]:
    """Return the task list and action schema."""
    return {"tasks": TASKS, "action_schema": ACTION_SCHEMA}


@app.get("/grader")
async def get_grader() -> dict[str, float]:
    """Return grader scores for the current episode state."""
    return {
        "easy": grade_easy(),
        "medium": grade_medium(),
        "hard": grade_hard(),
    }


@app.post("/baseline")
async def run_baseline() -> dict[str, Any]:
    """Run the optimal policy and return per-task scores."""
    env = EcommerceEnv()
    env.reset()
    policy = ["add_item", "apply_coupon", "checkout", "pay"]
    episode: list[dict[str, Any]] = []
    for action in policy:
        act = EcommerceAction(action=action)
        obs = env.step(act)
        episode.append({"action": action, "reward": obs.reward, "done": obs.done})

    return {
        "policy": "optimal (add_item → apply_coupon → checkout → pay)",
        "episode": episode,
        "scores": {
            "easy": grade_easy(),
            "medium": grade_medium(),
            "hard": grade_hard(),
        },
    }


# ---------------------------------------------------------------------------
# Entry point for `uv run server` or direct execution
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
