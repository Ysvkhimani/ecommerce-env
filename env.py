state = {}
history = []

def reset():
    global state, history

    state.clear()   # 🔥 IMPORTANT
    state.update({
        "cart": [],
        "total": 0,
        "coupon_applied": False,
        "payment_done": False,
        "order_status": "incomplete"
    })

    history.clear()  # 🔥 IMPORTANT

    return state

def step(action):
    global state, history

    reward = 0
    done = False

    if action == "add_item":
        state["cart"].append("item")
        state["total"] += 100
        reward = 0.2

    elif action == "apply_coupon":
        if not state["coupon_applied"]:
            state["total"] *= 0.9
            state["coupon_applied"] = True
            reward = 0.3
        else:
            reward = -0.1

    elif action == "checkout":
        if state["cart"]:
            reward = 0.3
        else:
            reward = -0.5

    elif action == "pay":
        if state["cart"]:
            state["payment_done"] = True
            state["order_status"] = "completed"
            reward = 1.0
            done = True
        else:
            reward = -1.0

    history.append(action)

    return state, reward, done