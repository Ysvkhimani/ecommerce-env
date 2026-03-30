from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env import reset, step, state as env_state
import grader

app = FastAPI(title="Ecommerce OpenEnv API")


# UI HOME PAGE
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Ecommerce OpenEnv</title>
        </head>
        <body style="font-family: Arial; text-align:center; margin-top:50px;">
            <h1>🛒 Ecommerce OpenEnv</h1>
            <p>Your AI environment is running successfully ✅</p>

            <h3>Available Endpoints</h3>
            <ul style="list-style:none;">
                <li><a href="/reset">/reset</a></li>
                <li><a href="/state">/state</a></li>
                <li><a href="/tasks">/tasks</a></li>
                <li><a href="/grader">/grader</a></li>
                <li><a href="/baseline">/baseline</a></li>
                <li><a href="/docs">Swagger Docs</a></li>
            </ul>

            <p>💡 Use /docs to test APIs interactively</p>
        </body>
    </html>
    """



@app.get("/web", response_class=HTMLResponse)
def web_ui():
    return home()


# Health
@app.get("/health")
def health():
    return {"status": "ok"}


# REST APIs (unchanged)
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


@app.get("/baseline")
def run_baseline():
    reset()
    for action in ["add_item", "apply_coupon", "checkout", "pay"]:
        step(action)

    return {
        "scores": {
            "easy": grader.grade_easy(),
            "medium": grader.grade_medium(),
            "hard": grader.grade_hard()
        },
        "final_state": env_state
    }


# run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)