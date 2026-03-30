"""Example script: run the default purchase flow."""

from __future__ import annotations

from env import InvalidActionError, reset, step


def run():
    state = reset()

    actions = ["add_item", "apply_coupon", "checkout", "pay"]

    for action in actions:
        try:
            state, reward, done = step(action)
        except InvalidActionError as e:
            print(f"Invalid action skipped: {e}")
            raise
    return state


if __name__ == "__main__":
    result = run()
    print("Final State:", result)
