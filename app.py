from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env import reset, step, state as env_state
import grader

app = FastAPI()

print("🔥 APP STARTED SUCCESSFULLY 🔥")


@app.get("/")
def root():
    return {"message": "Ecommerce API is LIVE 🚀"}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <h1>🛒 Ecommerce OpenEnv</h1>
    <p>Working UI ✅</p>
    <a href="/docs">Go to Swagger</a>
    """


@app.get("/reset")
def reset_env():
    return {"state": reset()}


@app.post("/step")
def take_step(action: str):
    s, reward, done = step(action)
    return {"state": s, "reward": reward, "done": done}


@app.get("/state")
def get_state():
    return {"state": env_state}


@app.get("/tasks")
def tasks():
    return {
        "tasks": ["easy", "medium", "hard"],
        "actions": ["add_item", "apply_coupon", "checkout", "pay"]
    }


@app.get("/grader")
def get_grades():
    return {
        "easy": grader.grade_easy(),
        "medium": grader.grade_medium(),
        "hard": grader.grade_hard()
    }