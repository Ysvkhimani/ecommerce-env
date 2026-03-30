from fastapi import FastAPI
from env import reset, step, state as env_state
import grader

app = FastAPI()


# ✅ Root endpoint 
@app.get("/")
def home():
    return {
        "message": "Ecommerce OpenEnv is running",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"]
    }


# ✅ Reset environment
@app.get("/reset")
def reset_env():
    current_state = reset()
    return {"state": current_state}


# ✅ Take action
@app.post("/step")
def take_step(action: str):
    s, reward, done = step(action)
    return {
        "state": s,
        "reward": reward,
        "done": done
    }


# ✅ Get current state
@app.get("/state")
def get_state():
    return {"state": env_state}


# ✅ Tasks info
@app.get("/tasks")
def tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "actions": ["add_item", "apply_coupon", "checkout", "pay"]
    }


# ✅ Grader scores
@app.get("/grader")
def get_grades():
    return {
        "easy": grader.grade_easy(),
        "medium": grader.grade_medium(),
        "hard": grader.grade_hard()
    }


# ✅ Baseline agent (important for evaluation)
@app.get("/baseline")
def run_baseline():
    reset()

    actions = ["add_item", "apply_coupon", "checkout", "pay"]

    for action in actions:
        step(action)

    return {
        "scores": {
            "easy": grader.grade_easy(),
            "medium": grader.grade_medium(),
            "hard": grader.grade_hard()
        },
        "final_state": env_state
    }


# ✅ Local run support (important for Docker/HF)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)