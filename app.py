"""Gradio Space entrypoint (sdk: gradio). No Docker — fast Hub builds."""

from __future__ import annotations

import json
from typing import Any

import gradio as gr
from pydantic import ValidationError

from ecommerce_environment import EcommerceEnvironment
from grader import grade_easy, grade_hard, grade_medium
from models import EcommerceAction, EcommerceObservation

_env = EcommerceEnvironment()


def serialize_observation(obs: EcommerceObservation) -> dict[str, Any]:
    full = obs.model_dump()
    reward = full.pop("reward", None)
    done = full.pop("done", False)
    full.pop("metadata", None)
    return {"observation": full, "reward": reward, "done": done}


def _format_obs_md(data: dict[str, Any]) -> str:
    obs = data.get("observation", {})
    lines = [
        f"**Cart:** `{obs.get('cart', [])}`",
        f"**Total:** `{obs.get('total')}`",
        f"**Coupon:** `{obs.get('coupon_applied')}` | **Paid:** `{obs.get('payment_done')}` | **Status:** `{obs.get('order_status')}`",
        f"**Reward:** `{data.get('reward')}` | **Done:** `{data.get('done')}`",
    ]
    return "\n\n".join(lines)


def _gradio_reset() -> tuple[str, str]:
    obs = _env.reset()
    data = serialize_observation(obs)
    return _format_obs_md(data), json.dumps(data, indent=2)


def _gradio_step(action_name: str) -> tuple[str, str]:
    if not action_name:
        return "", "Choose an action."
    try:
        act = EcommerceAction(action=action_name)  # type: ignore[arg-type]
    except ValidationError as e:
        return "", json.dumps(e.errors(), indent=2)
    obs = _env.step(act)
    data = serialize_observation(obs)
    return _format_obs_md(data), json.dumps(data, indent=2)


def _gradio_state() -> str:
    return json.dumps(_env.state.model_dump(), indent=2)


def _gradio_grades() -> str:
    return json.dumps(
        {
            "easy": grade_easy(),
            "medium": grade_medium(),
            "hard": grade_hard(),
        },
        indent=2,
    )


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
