"""Baseline inference script.

Runs the optimal policy (add_item → apply_coupon → checkout → pay) and
reports grader scores for all three tasks (easy / medium / hard).

Usage:
    python baseline.py

Expected output:
    easy:   1.0
    medium: 1.0
    hard:   1.0
"""

from __future__ import annotations

import json

from env import reset, step
from grader import grade_easy, grade_hard, grade_medium


def run_baseline() -> dict[str, float]:
    """Execute optimal policy and return grader scores."""
    reset()
    policy = ["add_item", "apply_coupon", "checkout", "pay"]

    print("Running baseline policy:")
    for action in policy:
        state, reward, done = step(action)
        print(f"  action={action!r:15s}  reward={reward:+.2f}  done={done}")

    scores = {
        "easy": grade_easy(),
        "medium": grade_medium(),
        "hard": grade_hard(),
    }
    return scores


if __name__ == "__main__":
    scores = run_baseline()
    print("\nBaseline scores:")
    print(json.dumps(scores, indent=2))
