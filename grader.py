"""Grading helpers — read the shared simulator state after an episode."""

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

    Score: 1.0 if ticket is resolved, else 0.0
    """
    try:
        return 1.0 if _read_state().get("resolved") else 0.0
    except Exception:
        logger.exception("grade_easy failed")
        return 0.0


def grade_medium() -> float:
    """Task: Resolve with meaningful customer satisfaction.

    Score: satisfaction_score (0.0–1.0) if resolved, else 0.0
    Partial credit at 0.5 for resolving but with low satisfaction (< 0.5).
    """
    try:
        s = _read_state()
        if not s.get("resolved"):
            return 0.0
        score = float(s.get("satisfaction_score", 0.0))
        if score >= 0.7:
            return score            # e.g. 0.90 sentiment → 0.90 score
        return max(0.3, score)      # still some credit for resolving
    except Exception:
        logger.exception("grade_medium failed")
        return 0.0


def grade_hard() -> float:
    """Task: Resolve efficiently — high satisfaction, ≤ 5 steps, no escalation.

    Scoring rubric:
      1.0  — resolved + satisfaction ≥ 0.8 + steps ≤ 5 + no escalation
      0.7  — resolved + satisfaction ≥ 0.7 + no escalation
      0.5  — resolved + satisfaction ≥ 0.5 + no escalation
      0.2  — resolved but escalated or too many steps
      0.0  — not resolved
    """
    try:
        s = _read_state()
        h = _read_history()
        if not s.get("resolved"):
            return 0.0
        satisfaction = float(s.get("satisfaction_score", 0.0))
        steps = len(h)
        escalated = bool(s.get("escalated", False))
        if satisfaction >= 0.8 and steps <= 5 and not escalated:
            return 1.0
        if satisfaction >= 0.7 and not escalated:
            return 0.7
        if satisfaction >= 0.5 and not escalated:
            return 0.5
        return 0.2  # resolved but poorly
    except Exception:
        logger.exception("grade_hard failed")
        return 0.0
