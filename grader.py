"""Grading helpers — evaluates the quality of ticket resolution."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _read_state() -> Dict[str, Any]:
    from env import state as sim_state
    return sim_state if isinstance(sim_state, dict) else {}


def _read_history() -> List[str]:
    from env import history as sim_history
    return sim_history if isinstance(sim_history, list) else []


def grade_easy() -> float:
    """Task: Resolve the ticket — any resolution counts.
    Score: 1.0 if resolved, else 0.0
    """
    try:
        return 1.0 if _read_state().get("resolved") else 0.0
    except Exception:
        logger.exception("grade_easy failed")
        return 0.0


def grade_medium() -> float:
    """Task: Resolve with meaningful customer satisfaction.
    Score: satisfaction_score if resolved (minimum 0.3 for any resolution).
    """
    try:
        s = _read_state()
        if not s.get("resolved"):
            return 0.0
        score = float(s.get("satisfaction_score", 0.0))
        return max(0.3, score)
    except Exception:
        logger.exception("grade_medium failed")
        return 0.0


def grade_hard() -> float:
    """Task: Resolve correctly and efficiently.

    Requires:
      - Ticket resolved
      - Correct resolution action used for the ticket type
      - No escalation
      - Satisfaction ≥ 0.7
      - Steps ≤ 6

    Scoring:
      1.0  → all conditions met
      0.7  → resolved + correct action + satisfaction ≥ 0.6 + no escalation
      0.5  → resolved + correct action + any satisfaction
      0.3  → resolved but wrong action or escalated
      0.0  → not resolved
    """
    try:
        s = _read_state()
        h = _read_history()
        if not s.get("resolved"):
            return 0.0
        satisfaction = float(s.get("satisfaction_score", 0.0))
        steps = len(h)
        escalated = bool(s.get("escalated", False))
        correct = bool(s.get("correct_resolution_used", False))

        if correct and satisfaction >= 0.7 and steps <= 6 and not escalated:
            return 1.0
        if correct and satisfaction >= 0.6 and not escalated:
            return 0.7
        if correct and not escalated:
            return 0.5
        return 0.3
    except Exception:
        logger.exception("grade_hard failed")
        return 0.0
