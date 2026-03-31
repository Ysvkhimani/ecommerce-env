"""
Inference Script — E-commerce Customer Support Agent
=====================================================
An LLM agent resolves customer support tickets by reading the ticket type
and choosing the appropriate actions. The agent must generalise across
5 different ticket scenarios.

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

MAX_STEPS = 12
TEMPERATURE = 0.0
MAX_TOKENS = 30

from ecommerce_environment import CustomerSupportEnvironment
from grader import grade_easy, grade_expert, grade_hard, grade_medium
from models import SupportAction

VALID_ACTIONS = [
    "acknowledge", "investigate", "offer_refund", "offer_exchange",
    "apply_discount", "send_update", "escalate", "request_info", "resolve",
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI customer support agent for an e-commerce store.
    Each episode you receive a NEW customer ticket. Read it carefully —
    the ticket type determines the best resolution action.

    Available actions:
      acknowledge   — Acknowledge the issue (always a good first step)
      investigate   — Look up order/account details (do this before offering solutions)
      offer_refund  — Issue full refund (best for: damaged, missing, wrong items)
      offer_exchange — Send replacement (best for: wrong item, damaged item)
      apply_discount — Give 10%% off next order (best for: late delivery)
      send_update   — Send delivery status update (best for: late delivery)
      escalate      — Transfer to senior agent (avoid — penalised)
      request_info  — Ask for more info (avoid if info already given — penalised)
      resolve       — Close the ticket (do AFTER offering appropriate solution)

    IMPORTANT: Match your resolution to the ticket type!
    - damaged_item  → offer_refund or offer_exchange, then resolve
    - wrong_item    → offer_exchange or offer_refund, then resolve
    - missing_item  → offer_refund or offer_exchange, then resolve
    - late_delivery → send_update, apply_discount, then resolve
    - billing_issue → investigate (to confirm charge), then resolve

    Reply with ONLY the action name.
""").strip()

TASKS = [
    ("easy",   "Resolve the customer support ticket."),
    ("medium", "Resolve the ticket with high customer satisfaction."),
    ("hard",   "Resolve correctly: use the right action for the ticket type, ≤6 steps, no escalation."),
    ("expert", "Near-perfect resolution: correct action for ticket type, satisfaction ≥ 0.8, ≤ 4 steps, no escalation."),
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
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\nYour objective: {task_desc}"}
    ]

    for step_num in range(1, MAX_STEPS + 1):
        user_content = (
            f"Step {step_num}/{MAX_STEPS}\n"
            f"Ticket type: {obs.ticket_type}\n"
            f"Subject: {obs.ticket_subject}\n"
            f"Customer ({obs.customer_name}, {obs.customer_tier}): {obs.ticket_description}\n"
            f"Customer's latest message: \"{obs.customer_response}\"\n"
            f"Sentiment: {obs.sentiment:.2f} | Investigated: {obs.investigated} | "
            f"Refund: {obs.refund_offered} | Exchange: {obs.exchange_offered} | "
            f"Update sent: {obs.update_sent} | Resolved: {obs.resolved}\n\n"
            f"Choose action: ({' / '.join(VALID_ACTIONS)})"
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
        print(f"    step {step_num:2d}: {chosen:20s}  sentiment={obs.sentiment:.2f}  customer: \"{obs.customer_response[:50]}\"")

        act = SupportAction(action=chosen)
        obs = env.step(act)
        if obs.done:
            break

    scores = {"easy": grade_easy(), "medium": grade_medium(), "hard": grade_hard(), "expert": grade_expert()}
    return scores[task_id]


def main() -> dict[str, float]:
    if not API_KEY:
        print("WARNING: HF_TOKEN not set.", file=sys.stderr)
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results: dict[str, float] = {}
    for task_id, task_desc in TASKS:
        print(f"\n[Task: {task_id}] {task_desc}")
        score = run_task(client, task_id, task_desc)
        results[task_id] = score
        print(f"  -> score: {score:.2f}")
    print("\n" + "=" * 40)
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
