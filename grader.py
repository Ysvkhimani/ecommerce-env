from env import state, history


def grade_easy():
    return 1.0 if state["payment_done"] else 0.0


def grade_medium():
    if state["payment_done"] and state["coupon_applied"]:
        return 1.0
    elif state["payment_done"]:
        return 0.5
    return 0.0


def grade_hard():
    correct_flow = ["add_item", "apply_coupon", "checkout", "pay"]

    if history == correct_flow:
        return 1.0
    elif "pay" in history:
        return 0.5
    return 0.0