"""Grading helpers (read shared simulator state; defensive against missing keys)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _read_state() -> Dict[str, Any]:
    from env import state as cart_state

    if not isinstance(cart_state, dict):
        logger.error("env.state is not a dict")
        return {}
    return cart_state


def _read_history() -> List[str]:
    from env import history as cart_history

    if not isinstance(cart_history, list):
        logger.error("env.history is not a list")
        return []
    return cart_history


def grade_easy() -> float:
    try:
        st = _read_state()
        return 1.0 if st.get("payment_done") else 0.0
    except Exception as e:
        logger.exception("grade_easy failed")
        return 0.0


def grade_medium() -> float:
    try:
        st = _read_state()
        if st.get("payment_done") and st.get("coupon_applied"):
            return 1.0
        if st.get("payment_done"):
            return 0.5
        return 0.0
    except Exception as e:
        logger.exception("grade_medium failed")
        return 0.0


def grade_hard() -> float:
    try:
        hist = _read_history()
        correct_flow = ["add_item", "apply_coupon", "checkout", "pay"]
        if hist == correct_flow:
            return 1.0
        if "pay" in hist:
            return 0.5
        return 0.0
    except Exception as e:
        logger.exception("grade_hard failed")
        return 0.0
