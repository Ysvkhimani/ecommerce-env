"""
Inference Script — Ecommerce OpenEnv
=====================================
Runs an LLM agent (via OpenAI-compatible API) against the ecommerce environment
for each of the 3 tasks and reports baseline scores.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     The model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face / API key

Usage:
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration (read from environment — mandatory per submission rules)
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 30

# ---------------------------------------------------------------------------
# Environment and graders (imported directly — no server needed)
# ---------------------------------------------------------------------------
from ecommerce_environment import EcommerceEnvironment  # noqa: E402
from grader import grade_easy, grade_hard, grade_medium  # noqa: E402
from models import EcommerceAction  # noqa: E402

VALID_ACTIONS = ["add_item", "apply_coupon", "checkout", "pay"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI agent operating an e-commerce shopping cart.
    At each step you receive the current cart state and must choose exactly one action.
    Available actions: add_item, apply_coupon, checkout, pay

    Rules:
    - add_item   : adds a $100 item to the cart
    - apply_coupon: applies a 10% discount (only once per episode)
    - checkout   : moves to the checkout stage
    - pay        : completes the payment and closes the order

    Reply with ONLY the action name — no explanation, no punctuation.
    """
).strip()

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS = [
    {
        "id": "easy",
        "description": "Add at least one item to the cart and complete the payment.",
    },
    {
        "id": "medium",
        "description": (
            "Apply a coupon discount AND complete the payment. "
            "Hint: add an item first, then apply the coupon, then pay."
        ),
    },
    {
        "id": "hard",
        "description": (
            "Complete the purchase following the EXACT sequence: "
            "add_item → apply_coupon → checkout → pay."
        ),
    },
]


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _extract_action(text: str) -> str:
    """Return the first valid action found in the model response, else 'add_item'."""
    cleaned = text.strip().lower()
    for action in VALID_ACTIONS:
        if action in cleaned:
            return action
    return "add_item"


def run_task(client: OpenAI, task_id: str, task_description: str) -> float:
    """Run one episode for the given task; return the grader score."""
    env = EcommerceEnvironment()
    obs = env.reset()

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + f"\n\nYour current task: {task_description}",
        }
    ]

    for step_num in range(1, MAX_STEPS + 1):
        user_content = (
            f"Step {step_num}/{MAX_STEPS}\n"
            f"Cart: {obs.cart}\n"
            f"Total: ${obs.total:.2f}\n"
            f"Coupon applied: {obs.coupon_applied}\n"
            f"Payment done: {obs.payment_done}\n"
            f"Order status: {obs.order_status}\n\n"
            "Which action do you take? (add_item / apply_coupon / checkout / pay)"
        )
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        action_text: str = response.choices[0].message.content or ""
        chosen_action = _extract_action(action_text)

        messages.append({"role": "assistant", "content": chosen_action})
        print(f"    step {step_num:2d}: {chosen_action}")

        act = EcommerceAction(action=chosen_action)
        obs = env.step(act)

        if obs.done:
            break

    # Grade against the task that was requested
    scores: dict[str, float] = {
        "easy": grade_easy(),
        "medium": grade_medium(),
        "hard": grade_hard(),
    }
    return scores[task_id]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict[str, float]:
    if not API_KEY:
        print(
            "WARNING: HF_TOKEN / API_KEY not set — LLM calls will fail.\n"
            "Set the environment variables before running this script.",
            file=sys.stderr,
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results: dict[str, float] = {}
    for task in TASKS:
        print(f"\n[Task: {task['id']}] {task['description']}")
        score = run_task(client, task["id"], task["description"])
        results[task["id"]] = score
        print(f"  -> score: {score:.2f}")

    print("\n" + "=" * 40)
    print("Baseline scores:")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
