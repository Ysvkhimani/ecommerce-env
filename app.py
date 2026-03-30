"""Gradio Space entrypoint (sdk: gradio). No Docker — fast Hub builds."""

from __future__ import annotations

import json
import logging
from typing import Any

import gradio as gr
from pydantic import ValidationError

from ecommerce_environment import EcommerceEnvironment
from env import InvalidActionError
from grader import grade_easy, grade_hard, grade_medium
from models import EcommerceAction, EcommerceObservation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_env = EcommerceEnvironment()


def serialize_observation(obs: EcommerceObservation) -> dict[str, Any]:
    full = obs.model_dump()
    reward = full.pop("reward", None)
    done = full.pop("done", False)
    full.pop("metadata", None)
    return {"observation": full, "reward": reward, "done": done}


def _format_obs_md(data: dict[str, Any]) -> str:
    if "error" in data:
        return f"**Error:** {data.get('error', 'Unknown error')}"
    obs = data.get("observation") or {}
    if not isinstance(obs, dict):
        obs = {}
    lines = [
        f"**Cart:** `{obs.get('cart', [])}`",
        f"**Total:** `{obs.get('total')}`",
        f"**Coupon:** `{obs.get('coupon_applied')}` | **Paid:** `{obs.get('payment_done')}` | **Status:** `{obs.get('order_status')}`",
        f"**Reward:** `{data.get('reward')}` | **Done:** `{data.get('done')}`",
    ]
    return "\n\n".join(lines)


def _json_error(message: str, exc: BaseException | None = None) -> str:
    payload: dict[str, Any] = {"error": message}
    if exc is not None:
        payload["type"] = type(exc).__name__
        logger.warning("%s: %s", type(exc).__name__, exc, exc_info=exc)
    return json.dumps(payload, indent=2)


def _gradio_reset() -> tuple[str, str]:
    try:
        obs = _env.reset()
        data = serialize_observation(obs)
        return _format_obs_md(data), json.dumps(data, indent=2)
    except Exception as e:
        logger.exception("reset failed")
        msg = "Reset failed. Please try again."
        return f"**Error:** {msg}", _json_error(msg, e)


def _gradio_step(action_name: str) -> tuple[str, str]:
    if not action_name:
        warn = "Choose an action from the list before stepping."
        return f"**Warning:** {warn}", _json_error(warn)
    try:
        act = EcommerceAction(action=action_name)  # type: ignore[arg-type]
    except ValidationError as e:
        logger.info("Invalid action payload: %s", e)
        return "**Validation error** (check action value).", json.dumps(e.errors(), indent=2)
    try:
        obs = _env.step(act)
        data = serialize_observation(obs)
        return _format_obs_md(data), json.dumps(data, indent=2)
    except InvalidActionError as e:
        return f"**Invalid action:** {e}", _json_error(str(e), e)
    except RuntimeError as e:
        return f"**Runtime error:** {e}", _json_error(str(e), e)
    except Exception as e:
        logger.exception("unexpected step error")
        return "**Unexpected error** during step.", _json_error("Unexpected error during step.", e)


def _gradio_state() -> str:
    try:
        return json.dumps(_env.state.model_dump(), indent=2)
    except Exception as e:
        logger.exception("get state failed")
        return _json_error("Could not read environment state.", e)


def _gradio_grades() -> str:
    try:
        return json.dumps(
            {
                "easy": grade_easy(),
                "medium": grade_medium(),
                "hard": grade_hard(),
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("grader failed")
        return _json_error("Grader failed.", e)


with gr.Blocks(title="Ecommerce OpenEnv") as demo:
    gr.Markdown("# Ecommerce OpenEnv\nClick **Reset**, choose an action, then **Step**.")
    with gr.Row():
        reset_btn = gr.Button("Reset")
        step_btn = gr.Button("Step", variant="primary")
        state_btn = gr.Button("Get state")
        grades_btn = gr.Button("Run grader")
    action_in = gr.Dropdown(
        choices=["add_item", "apply_coupon", "checkout", "pay"],
        label="Action",
    )
    obs_md = gr.Markdown(label="Observation")
    raw = gr.Code(label="JSON", language="json")
    grades_out = gr.Code(label="Grader (easy / medium / hard)", language="json", visible=True)

    reset_btn.click(_gradio_reset, outputs=[obs_md, raw])
    step_btn.click(_gradio_step, inputs=[action_in], outputs=[obs_md, raw])
    state_btn.click(_gradio_state, outputs=[raw])
    grades_btn.click(_gradio_grades, outputs=[grades_out])
