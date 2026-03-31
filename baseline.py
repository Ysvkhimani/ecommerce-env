"""Baseline script — runs the optimal policy and prints grader scores.

Optimal policy: acknowledge → investigate → offer_refund → resolve
Expected scores: easy=1.0, medium≈0.90, hard=1.0

Usage:
    python baseline.py
"""

from __future__ import annotations

import json
from env import OPTIMAL_POLICY, reset, step
from grader import grade_easy, grade_hard, grade_medium


def run_baseline() -> dict[str, float]:
    reset()
    print("Running optimal policy:")
    for action in OPTIMAL_POLICY:
        state, reward, done = step(action)
        print(f"  {action:20s}  sentiment={state['sentiment']:.2f}  reward={reward:+.2f}  done={done}")

    scores = {"easy": grade_easy(), "medium": grade_medium(), "hard": grade_hard()}
    return scores


if __name__ == "__main__":
    scores = run_baseline()
    print("\nBaseline scores:")
    print(json.dumps(scores, indent=2))
