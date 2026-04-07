"""
Inference Script — E-commerce Customer Support Agent
=====================================================
MANDATORY environment variables:
    API_BASE_URL   LLM endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model name    (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key

STDOUT FORMAT (exact — any deviation causes incorrect scoring):
    [START] task=<name> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

# Ensure repo root is always on path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY") or "dummy"
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_NAME: str = "ecommerce-customer-support"

MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 30

from ecommerce_environment import CustomerSupportEnvironment
from grader import grade_easy, grade_expert, grade_hard, grade_medium
from models import SupportAction

VALID_ACTIONS = [
    "acknowledge", "investigate", "offer_refund", "offer_exchange",
    "apply_discount", "send_update", "escalate", "request_info", "resolve",
]

# Deterministic fallback if LLM unavailable
FALLBACK_POLICY = ["acknowledge", "investigate", "offer_refund", "resolve"]

GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "expert": grade_expert,
}

TASKS = [
    ("easy",   "Resolve the customer support ticket."),
    ("medium", "Resolve the ticket with high customer satisfaction."),
    ("hard",   "Resolve correctly: use the right action for the ticket type, ≤6 steps, no escalation."),
    ("expert", "Near-perfect: correct action for ticket type, satisfaction ≥ 0.8, ≤4 steps, no escalation."),
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI customer support agent for an e-commerce store.
    Read the ticket carefully — the ticket type determines the correct resolution.

    Actions:
      acknowledge    — Acknowledge the issue (good first step)
      investigate    — Look up order details (do before offering solutions)
      offer_refund   — Full refund (best for: damaged, missing, wrong, billing)
      offer_exchange — Send replacement (best for: wrong item, damaged item)
      apply_discount — 10% off next order (best for: late delivery)
      send_update    — Delivery status update (best for: late delivery ONLY)
      escalate       — Transfer to senior agent (avoid — penalised)
      request_info   — Ask for info (avoid if already provided — penalised)
      resolve        — Close the ticket (do AFTER offering appropriate solution)

    Ticket type → correct resolution:
    - damaged_item  → offer_refund or offer_exchange, then resolve
    - wrong_item    → offer_exchange or offer_refund, then resolve
    - missing_item  → offer_refund or offer_exchange, then resolve
    - late_delivery → send_update, apply_discount, then resolve
    - billing_issue → investigate, offer_refund, then resolve

    Reply with ONLY the action name. Nothing else.
""").strip()


def _extract_action(text: str) -> str:
    cleaned = text.strip().lower()
    for a in VALID_ACTIONS:
        if a in cleaned:
            return a
    return "acknowledge"


def _call_llm(client: OpenAI, messages: list) -> str | None:
    """Returns action string or None on failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=30,
        )
        return _extract_action(response.choices[0].message.content or "")
    except Exception as e:
        print(f"# LLM error: {e}", file=sys.stderr, flush=True)
        return None


def run_task(client: OpenAI, task_name: str, task_desc: str) -> float:
    """Run one task episode and emit structured stdout. Always emits [END]."""
    env = CustomerSupportEnvironment()
    rewards: list[float] = []
    step_num = 0
    success = False

    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\nObjective: {task_desc}"}
    ]
    use_fallback = False

    try:
        obs = env.reset()

        for step_num in range(1, MAX_STEPS + 1):
            # Choose action via LLM or fallback
            if not use_fallback:
                user_content = (
                    f"Step {step_num} | ticket={obs.ticket_type} | "
                    f"customer={obs.customer_name} ({obs.customer_tier})\n"
                    f"Subject: {obs.ticket_subject}\n"
                    f"Message: {obs.ticket_description}\n"
                    f"Customer says: \"{obs.customer_response}\"\n"
                    f"sentiment={obs.sentiment:.2f} investigated={obs.investigated} "
                    f"refund={obs.refund_offered} exchange={obs.exchange_offered} "
                    f"update_sent={obs.update_sent} resolved={obs.resolved}\n"
                    f"Choose one action: {' / '.join(VALID_ACTIONS)}"
                )
                messages.append({"role": "user", "content": user_content})
                chosen = _call_llm(client, messages)
                if chosen is None:
                    use_fallback = True

            if use_fallback:
                idx = step_num - 1
                chosen = FALLBACK_POLICY[idx] if idx < len(FALLBACK_POLICY) else "resolve"

            if not use_fallback:
                messages.append({"role": "assistant", "content": chosen})

            # Step environment
            obs = env.step(SupportAction(action=chosen))
            reward = float(obs.reward) if obs.reward is not None else 0.0
            done = bool(obs.done)
            rewards.append(reward)

            print(
                f"[STEP] step={step_num} action={chosen} reward={reward:.2f} "
                f"done={'true' if done else 'false'} error=null",
                flush=True,
            )

            if done:
                success = True
                break

    except Exception as e:
        print(f"# Episode error: {e}", file=sys.stderr, flush=True)
        # Ensure at least one [STEP] exists
        if not rewards:
            rewards = [0.0]
            print(f"[STEP] step=1 action=acknowledge reward=0.00 done=false error={str(e)}", flush=True)

    # Always emit [END]
    score = GRADERS[task_name]()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={max(step_num,1)} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    return score


def main() -> dict[str, float]:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results: dict[str, float] = {}

    for task_name, task_desc in TASKS:
        try:
            score = run_task(client, task_name, task_desc)
        except Exception as e:
            print(f"# Fatal error for task {task_name}: {e}", file=sys.stderr, flush=True)
            # Guaranteed output even on total failure
            print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}", flush=True)
            print(f"[STEP] step=1 action=acknowledge reward=0.00 done=false error={str(e)}", flush=True)
            print(f"[END] success=false steps=1 score=0.00 rewards=0.00", flush=True)
            score = 0.0
        results[task_name] = score

    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == "__main__":
    main()
