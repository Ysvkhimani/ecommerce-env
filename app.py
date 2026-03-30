"""FastAPI + Gradio: lightweight Space image (avoids openenv-core's huge dependency tree on HF builds)."""

from __future__ import annotations

import json
from typing import Any

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from ecommerce_environment import EcommerceEnvironment
from grader import grade_easy, grade_hard, grade_medium
from models import EcommerceAction, EcommerceObservation, EcommerceEnvState, StepRequest

# Single env for REST + /web playground (shared cart with grader via env.get_simulator()).
_env = EcommerceEnvironment()

app = FastAPI(
    title="Ecommerce OpenEnv",
    version="0.1.0",
    description="E-commerce cart simulation with Gradio UI at `/web`.",
)


def serialize_observation(obs: EcommerceObservation) -> dict[str, Any]:
    """OpenEnv-style envelope: observation fields without reward/done at top level."""
    full = obs.model_dump()
    reward = full.pop("reward", None)
    done = full.pop("done", False)
    full.pop("metadata", None)
    return {"observation": full, "reward": reward, "done": done}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset")
def reset_episode() -> dict[str, Any]:
    obs = _env.reset()
    return serialize_observation(obs)


@app.post("/step")
def step_episode(body: StepRequest) -> dict[str, Any]:
    try:
        action = EcommerceAction.model_validate(body.action)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    obs = _env.step(action)
    return serialize_observation(obs)


@app.get("/state", response_model=EcommerceEnvState)
def get_state() -> EcommerceEnvState:
    return _env.state


@app.get("/schema")
def get_schema() -> dict[str, Any]:
    return {
        "action": EcommerceAction.model_json_schema(),
        "observation": EcommerceObservation.model_json_schema(),
        "state": EcommerceEnvState.model_json_schema(),
    }


@app.get("/tasks")
def tasks() -> dict:
    return {
        "tasks": ["easy", "medium", "hard"],
        "actions": ["add_item", "apply_coupon", "checkout", "pay"],
    }


@app.get("/grader")
def get_grades() -> dict:
    return {
        "easy": grade_easy(),
        "medium": grade_medium(),
        "hard": grade_hard(),
    }


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


def build_gradio() -> gr.Blocks:
    with gr.Blocks(title="Ecommerce OpenEnv") as demo:
        gr.Markdown("# Ecommerce OpenEnv\nUse **Reset**, then **Step** with an action.")
        with gr.Row():
            reset_btn = gr.Button("Reset")
            step_btn = gr.Button("Step", variant="primary")
            state_btn = gr.Button("Get state")
        action_in = gr.Dropdown(
            choices=["add_item", "apply_coupon", "checkout", "pay"],
            label="Action",
        )
        obs_md = gr.Markdown(label="Observation")
        raw = gr.Code(label="JSON", language="json")
        reset_btn.click(_gradio_reset, outputs=[obs_md, raw])
        step_btn.click(_gradio_step, inputs=[action_in], outputs=[obs_md, raw])
        state_btn.click(lambda: _gradio_state(), outputs=[raw])
    return demo


app = gr.mount_gradio_app(app, build_gradio(), path="/web")
