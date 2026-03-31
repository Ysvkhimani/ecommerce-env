"""
Inference Script — E-commerce Customer Support Agent
=====================================================
An LLM agent (via OpenAI-compatible API) acts as a customer support representative.
It reads a customer ticket and chooses actions to resolve it.

MANDATORY environment variables:
    API_BASE_URL   LLM endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model to use  (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key

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

from openai import OpenAI

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 30

from ecommerce_environment import CustomerSupportEnvironment
from grader import grade_easy, grade_hard, grade_medium
from models import SupportAction

VALID_ACTIONS = [
    "acknowledge", "investigate", "offer_refund", "offer_exchange",
    "apply_discount", "escalate", "request_info", "resolve",
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI customer support agent for an e-commerce store.
    A customer has submitted a support ticket. Your goal is to resolve it
    efficiently while keeping the customer satisfied.

    Available actions:
      acknowledge   — Acknowledge the customer's issue (builds trust)
      investigate   — Look up the order details (required before offering solutions)
      offer_refund  — Issue a full refund (best for damaged/wrong items)
      offer_exchange — Send a replacement item (alternative to refund)
      apply_discount — Give 10%% off next order as goodwill gesture
      escalate      — Transfer to senior agent (use only if necessary)
      request_info  — Ask customer for more info (use only if missing details)
      resolve       — Close the ticket (do this after offering a solution)

    Reply with ONLY the action name — nothing else.
    Best practice: acknowledge → investigate → offer_refund/offer_exchange → resolve
""").strip()

TASKS = [
    ("easy",   "Resolve the customer support ticket — any resolution counts."),
    ("medium", "Resolve the ticket with high customer satisfaction."),
    ("hard",   "Resolve efficiently: satisfaction ≥ 0.8, ≤ 5 steps, no escalation."),
]


def _extract_action(text: str) -> str:
    cleaned = text.strip().lower()
    for a in VALID_ACTIONS:
        if a in cleaned:
            return a
    return "acknowledge"


def run_task(client: OpenAI, task_id: str, task_desc: str) -> float:
    env = CustomerSupportEnvironment()
    obs = env.reset()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + f"\n\nYour current objective: {task_desc}",
        }
    ]

    for step_num in range(1, MAX_STEPS + 1):
        user_content = (
            f"Step {step_num}/{MAX_STEPS}\n"
            f"Ticket: {obs.ticket_subject}\n"
            f"Customer message: {obs.ticket_description}\n"
            f"Customer: {obs.customer_name} ({obs.customer_tier} tier)\n"
            f"Current sentiment: {obs.sentiment:.2f} (0=very angry, 1=very happy)\n"
            f"Investigated: {obs.investigated} | Refund offered: {obs.refund_offered} | "
            f"Exchange offered: {obs.exchange_offered} | Resolved: {obs.resolved}\n\n"
            f"What action do you take? ({' / '.join(VALID_ACTIONS)})"
        )
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        action_text = response.choices[0].message.content or ""
        chosen = _extract_action(action_text)
        messages.append({"role": "assistant", "content": chosen})
        print(f"    step {step_num:2d}: {chosen:20s}  sentiment={obs.sentiment:.2f}")

        act = SupportAction(action=chosen)
        obs = env.step(act)

        if obs.done:
            break

    scores = {"easy": grade_easy(), "medium": grade_medium(), "hard": grade_hard()}
    return scores[task_id]


def main() -> dict[str, float]:
    if not API_KEY:
        print("WARNING: HF_TOKEN / API_KEY not set.", file=sys.stderr)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results: dict[str, float] = {}

    for task_id, task_desc in TASKS:
        print(f"\n[Task: {task_id}] {task_desc}")
        score = run_task(client, task_id, task_desc)
        results[task_id] = score
        print(f"  -> score: {score:.2f}")

    print("\n" + "=" * 40)
    print("Baseline scores:")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
