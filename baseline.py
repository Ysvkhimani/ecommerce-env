"""Baseline — runs the optimal policy for each ticket type and prints scores."""

from __future__ import annotations

import json
from env import OPTIMAL_POLICIES, TICKET_SCENARIOS, reset, step
from grader import grade_easy, grade_hard, grade_medium


def run_episode(ticket_index: int) -> dict:
    """Run optimal policy for ticket at given index."""
    from env import _sim, _RNG
    # Force a specific ticket for reproducibility
    _sim._scenario = TICKET_SCENARIOS[ticket_index]
    ticket_type = TICKET_SCENARIOS[ticket_index]["type"]
    policy = OPTIMAL_POLICIES[ticket_type]

    # Reset using the forced scenario
    _sim.state.clear()
    sc = _sim._scenario
    _sim.state.update({
        "ticket_id": sc["ticket_id"], "ticket_type": sc["type"],
        "ticket_subject": sc["subject"], "ticket_description": sc["description"],
        "customer_name": sc["customer_name"], "customer_tier": sc["customer_tier"],
        "order_value": sc["order_value"], "correct_resolutions": list(sc["correct_resolutions"]),
        "opening_message": sc["opening_message"], "sentiment": sc["initial_sentiment"],
        "investigated": False, "refund_offered": False, "exchange_offered": False,
        "discount_applied": False, "update_sent": False, "escalated": False,
        "resolved": False, "satisfaction_score": 0.0, "correct_resolution_used": False,
        "customer_response": sc["opening_message"],
    })
    _sim.history.clear()
    from uuid import uuid4
    _sim.episode_id = str(uuid4())

    print(f"\n  [{ticket_type}] policy: {' → '.join(policy)}")
    for action in policy:
        s, reward, done = step(action)
        print(f"    {action:20s}  sentiment={s['sentiment']:.2f}  reward={reward:+.2f}")

    return {"easy": grade_easy(), "medium": grade_medium(), "hard": grade_hard()}


if __name__ == "__main__":
    print("Running baseline for all 5 ticket types:\n")
    all_scores = {}
    for i, scenario in enumerate(TICKET_SCENARIOS):
        scores = run_episode(i)
        all_scores[scenario["type"]] = scores
        print(f"  Scores: easy={scores['easy']:.2f}  medium={scores['medium']:.2f}  hard={scores['hard']:.2f}")

    print("\n" + "=" * 50)
    print("All baseline scores:")
    print(json.dumps(all_scores, indent=2))
