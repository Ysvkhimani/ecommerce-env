"""Grading helpers — evaluates the quality of ticket resolution."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _clamp(score: float) -> float:
    """Ensure score is strictly within (0, 1) — validator requirement."""
    return max(0.01, min(0.99, score))


def _read_state() -> Dict[str, Any]:
    from env import state as sim_state
    return sim_state if isinstance(sim_state, dict) else {}


def _read_history() -> List[str]:
    from env import history as sim_history
    return sim_history if isinstance(sim_history, list) else []


def grade_easy() -> float:
    """Task: Resolve the ticket — any resolution counts.
    Score: 0.99 if resolved, else 0.01
    """
    try:
        return _clamp(1.0 if _read_state().get("resolved") else 0.0)
    except Exception:
        logger.exception("grade_easy failed")
        return 0.01


def grade_medium() -> float:
    """Task: Resolve with meaningful customer satisfaction.
    Score: satisfaction_score if resolved (minimum 0.30 for any resolution).
    """
    try:
        s = _read_state()
        if not s.get("resolved"):
            return _clamp(0.0)
        score = float(s.get("satisfaction_score", 0.0))
        return _clamp(max(0.30, score))
    except Exception:
        logger.exception("grade_medium failed")
        return 0.01


def grade_expert() -> float:
    """Expert task: near-perfect resolution — correct action, minimum steps, maximum satisfaction.

    This is designed to challenge frontier models. Requirements:
      - Ticket resolved without customer hanging up
      - Correct resolution action for the ticket type
      - No escalation
      - Satisfaction ≥ 0.8 AND steps ≤ 4 for full score

    Scoring (clamped to open interval):
      0.95 → correct + satisfaction ≥ 0.8 + steps ≤ 4 + no escalation
      0.60 → correct + satisfaction ≥ 0.7 + steps ≤ 5 + no escalation
      0.30 → correct + no escalation (any satisfaction/steps)
      0.05 → not resolved, wrong resolution, or customer hung up
    """
    try:
        s = _read_state()
        h = _read_history()
        if not s.get("resolved"):
            return _clamp(0.0)
        satisfaction = float(s.get("satisfaction_score", 0.0))
        steps = len(h)
        escalated = bool(s.get("escalated", False))
        correct = bool(s.get("correct_resolution_used", False))

        if correct and satisfaction >= 0.8 and steps <= 4 and not escalated:
            return _clamp(1.0)
        if correct and satisfaction >= 0.7 and steps <= 5 and not escalated:
            return _clamp(0.6)
        if correct and not escalated:
            return _clamp(0.3)
        return _clamp(0.0)
    except Exception:
        logger.exception("grade_expert failed")
        return 0.01


def grade_hard() -> float:
    """Task: Resolve correctly and efficiently.

    Requires:
      - Ticket resolved
      - Correct resolution action used for the ticket type
      - No escalation
      - Satisfaction ≥ 0.7
      - Steps ≤ 6

    Scoring (clamped to open interval):
      0.95 → all conditions met
      0.70 → resolved + correct action + satisfaction ≥ 0.6 + no escalation
      0.50 → resolved + correct action + any satisfaction
      0.30 → resolved but wrong action or escalated
      0.05 → not resolved
    """
    try:
        s = _read_state()
        h = _read_history()
        if not s.get("resolved"):
            return _clamp(0.0)
        satisfaction = float(s.get("satisfaction_score", 0.0))
        steps = len(h)
        escalated = bool(s.get("escalated", False))
        correct = bool(s.get("correct_resolution_used", False))

        if correct and satisfaction >= 0.7 and steps <= 6 and not escalated:
            return _clamp(1.0)
        if correct and satisfaction >= 0.6 and not escalated:
            return _clamp(0.7)
        if correct and not escalated:
            return _clamp(0.5)
        return _clamp(0.3)
    except Exception:
        logger.exception("grade_hard failed")
        return 0.01
