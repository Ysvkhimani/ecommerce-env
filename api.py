"""FastAPI application: OpenEnv HTTP API + Gradio UI (mounted at /ui).

Endpoints required by the hackathon judges:
  GET  /         -> health check (200 OK)
  GET  /tasks    -> task list + action schema
  POST /reset    -> reset episode, return initial observation
  POST /step     -> execute one action, return observation + reward + done
  GET  /state    -> current environment state
  GET  /grader   -> grader scores (easy / medium / hard)
  POST /baseline -> run optimal policy, return episode + scores

Gradio UI is mounted at /ui for human interaction.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("ecommerce-env")

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ValidationError

from ecommerce_environment import EcommerceEnvironment
from env import InvalidActionError
from grader import grade_easy, grade_hard, grade_medium
from models import EcommerceAction

# ---------------------------------------------------------------------------
# Shared environment instance (used by both FastAPI routes and Gradio UI)
# ---------------------------------------------------------------------------
_env = EcommerceEnvironment()

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
        "scoring": "1.0 if exact sequence matched, 0.5 if paid, else 0.0",
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
        "description": "Optional extra metadata (ignored by simulator)",
        "default": {},
    },
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
fastapi_app = FastAPI(
    title="Ecommerce OpenEnv",
    description="E-commerce cart simulation — OpenEnv standard interface",
    version="1.0.0",
)


@fastapi_app.get("/")
async def root():
    return {
        "status": "ok",
        "space_id": os.environ.get("SPACE_ID", "local"),
        "ui": "/ui",
        "docs": "/docs",
    }


@fastapi_app.get("/tasks")
async def get_tasks():
    """Return task list and action schema (required by judges)."""
    return {"tasks": TASKS, "action_schema": ACTION_SCHEMA}


@fastapi_app.post("/reset")
async def reset_env():
    """Reset the environment and return the initial observation."""
    obs = _env.reset()
    return obs.model_dump()


class _StepBody(BaseModel):
    action: str
    metadata: dict = {}


@fastapi_app.post("/step")
async def step_env(body: _StepBody):
    """Execute one action and return observation, reward, and done flag."""
    try:
        act = EcommerceAction(action=body.action, metadata=body.metadata)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"error": "Invalid action", "detail": exc.errors()})
    try:
        obs = _env.step(act)
    except InvalidActionError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    raw = obs.model_dump()
    return {
        "observation": {k: v for k, v in raw.items() if k not in ("reward", "done", "metadata")},
        "reward": obs.reward,
        "done": obs.done,
    }


@fastapi_app.get("/state")
async def get_state():
    """Return full environment state (step count, cart, history, …)."""
    return _env.state.model_dump()


@fastapi_app.get("/grader")
async def get_grader():
    """Return grader scores for the current episode state."""
    return {
        "easy": grade_easy(),
        "medium": grade_medium(),
        "hard": grade_hard(),
    }


@fastapi_app.post("/baseline")
async def run_baseline():
    """Run the optimal baseline policy and return per-task scores."""
    _env.reset()
    policy = ["add_item", "apply_coupon", "checkout", "pay"]
    episode: list[dict[str, Any]] = []
    for action in policy:
        act = EcommerceAction(action=action)
        obs = _env.step(act)
        episode.append({"action": action, "reward": obs.reward, "done": obs.done})

    scores = {
        "easy": grade_easy(),
        "medium": grade_medium(),
        "hard": grade_hard(),
    }
    return {
        "policy": "optimal (add_item → apply_coupon → checkout → pay)",
        "episode": episode,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Gradio UI (mounted at /ui)
# ---------------------------------------------------------------------------
def _serialize_obs(obs: Any) -> dict[str, Any]:
    full = obs.model_dump()
    reward = full.pop("reward", None)
    done = full.pop("done", False)
    full.pop("metadata", None)
    return {"observation": full, "reward": reward, "done": done}


def _fmt_md(data: dict[str, Any]) -> str:
    if "error" in data:
        return f"**Error:** {data['error']}"
    obs = data.get("observation") or {}
    lines = [
        f"**Cart:** `{obs.get('cart', [])}`",
        f"**Total:** `{obs.get('total')}`",
        f"**Coupon:** `{obs.get('coupon_applied')}` | **Paid:** `{obs.get('payment_done')}` | **Status:** `{obs.get('order_status')}`",
        f"**Reward:** `{data.get('reward')}` | **Done:** `{data.get('done')}`",
    ]
    return "\n\n".join(lines)


def _ui_reset() -> tuple[str, str]:
    obs = _env.reset()
    data = _serialize_obs(obs)
    return _fmt_md(data), json.dumps(data, indent=2)


def _ui_step(action_name: str) -> tuple[str, str]:
    if not action_name:
        warn = "Choose an action from the list before stepping."
        return f"**Warning:** {warn}", json.dumps({"error": warn}, indent=2)
    try:
        act = EcommerceAction(action=action_name)
        obs = _env.step(act)
        data = _serialize_obs(obs)
        return _fmt_md(data), json.dumps(data, indent=2)
    except (InvalidActionError, ValidationError) as exc:
        return f"**Error:** {exc}", json.dumps({"error": str(exc)}, indent=2)


def _ui_state() -> str:
    return json.dumps(_env.state.model_dump(), indent=2)


def _ui_grades() -> str:
    return json.dumps(
        {"easy": grade_easy(), "medium": grade_medium(), "hard": grade_hard()},
        indent=2,
    )


_space_id = os.environ.get("SPACE_ID", "(local)")

with gr.Blocks(title="Ecommerce OpenEnv", analytics_enabled=False) as demo:
    gr.Markdown(
        f"# 🛒 Ecommerce OpenEnv\n"
        f"**SPACE_ID:** `{_space_id}`\n\n"
        "Interact manually below, or use the HTTP API:\n"
        "`GET /tasks` · `POST /reset` · `POST /step` · `GET /state` · `GET /grader` · `POST /baseline`"
    )
    with gr.Row():
        reset_btn = gr.Button("Reset")
        step_btn = gr.Button("Step", variant="primary")
        state_btn = gr.Button("Get state")
        grades_btn = gr.Button("Run grader")
    action_in = gr.Dropdown(
        choices=["add_item", "apply_coupon", "checkout", "pay"],
        value="add_item",
        label="Action",
    )
    obs_md = gr.Markdown(value="_Initialising…_", label="Observation")
    raw = gr.Code(value="{}", label="JSON", language="json")
    grades_out = gr.Code(value="{}", label="Grader scores (easy / medium / hard)", language="json")

    demo.load(_ui_reset, outputs=[obs_md, raw])
    reset_btn.click(_ui_reset, outputs=[obs_md, raw])
    step_btn.click(_ui_step, inputs=[action_in], outputs=[obs_md, raw])
    state_btn.click(_ui_state, outputs=[raw])
    grades_btn.click(_ui_grades, outputs=[grades_out])

demo.queue()

# Mount Gradio inside FastAPI — all /ui/* routes go to Gradio, rest to FastAPI.
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

logger.info("api.py loaded; SPACE_ID=%s", _space_id)
