from env import reset, step

def run():
    state = reset()

    actions = ["add_item", "apply_coupon", "checkout", "pay"]

    for action in actions:
        state, reward, done = step(action)

    return state

if __name__ == "__main__":
    result = run()
    print("Final State:", result)