"""
OpenEnv-compliant FastAPI server — E-commerce Customer Support Agent Environment.

Standard endpoints (openenv.core.create_app):
    POST /reset     — start new episode
    POST /step      — execute action
    GET  /state     — current state
    GET  /schema    — action/observation schema
    WS   /ws        — WebSocket session

Additional hackathon endpoints:
    GET  /          — health check
    GET  /tasks     — task list + action schema
    GET  /grader    — grader scores (easy / medium / hard)
    POST /baseline  — run optimal policy, return episode + scores

Run locally:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
import sys
from typing import Any

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app

try:
    from .ecommerce_environment import CustomerSupportEnv
    from ..models import SupportAction, SupportObservation
    from ..grader import grade_easy, grade_hard, grade_medium
    from ..env import OPTIMAL_POLICY
except ImportError:
    from server.ecommerce_environment import CustomerSupportEnv
    from models import SupportAction, SupportObservation
    from grader import grade_easy, grade_hard, grade_medium
    from env import OPTIMAL_POLICY

# ---------------------------------------------------------------------------
# OpenEnv base app
# ---------------------------------------------------------------------------
app = create_app(
    CustomerSupportEnv,
    SupportAction,
    SupportObservation,
    env_name="ecommerce-support-env",
    max_concurrent_envs=1,
)

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS = [
    {
        "id": "easy",
        "description": "Resolve the customer support ticket — any resolution counts",
        "difficulty": "easy",
        "scoring": "1.0 if resolved, else 0.0",
    },
    {
        "id": "medium",
        "description": "Resolve the ticket with meaningful customer satisfaction",
        "difficulty": "medium",
        "scoring": "satisfaction_score (0.0–1.0) if resolved, partial credit for low satisfaction",
    },
    {
        "id": "hard",
        "description": (
            "Resolve efficiently: satisfaction ≥ 0.8, ≤ 5 steps, no escalation. "
            "Optimal: acknowledge → investigate → offer_refund → resolve"
        ),
        "difficulty": "hard",
        "scoring": "1.0 (efficient+satisfied), 0.7 (satisfied, no escalation), 0.5 (resolved, clean), 0.2 (escalated)",
    },
]

ACTION_SCHEMA = {
    "action": {
        "type": "string",
        "enum": [
            "acknowledge",
            "investigate",
            "offer_refund",
            "offer_exchange",
            "apply_discount",
            "escalate",
            "request_info",
            "resolve",
        ],
        "description": "Support action to execute",
    },
    "metadata": {"type": "object", "description": "Optional metadata", "default": {}},
}

# ---------------------------------------------------------------------------
# Hackathon-required endpoints
# ---------------------------------------------------------------------------


_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E-commerce Customer Support Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1e3a5f 0%, #0f2044 100%); padding: 24px 32px; border-bottom: 1px solid #334155; }
  .header h1 { font-size: 1.6rem; color: #93c5fd; }
  .header p  { color: #94a3b8; margin-top: 6px; font-size: 0.9rem; }
  .badge { display: inline-block; background: #1e40af; color: #bfdbfe; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; margin-left: 10px; }
  .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 24px 32px; max-width: 1200px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; }
  .card h2 { color: #93c5fd; font-size: 1rem; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
  .ticket-box { background: #0f172a; border: 1px solid #475569; border-radius: 8px; padding: 14px; font-size: 0.85rem; line-height: 1.6; }
  .ticket-box .label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .ticket-box .value { color: #e2e8f0; margin-bottom: 10px; }
  .actions-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .btn { padding: 10px 14px; border: none; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: all 0.15s; }
  .btn-action { background: #1e40af; color: #bfdbfe; }
  .btn-action:hover { background: #2563eb; transform: translateY(-1px); }
  .btn-resolve { background: #065f46; color: #6ee7b7; }
  .btn-resolve:hover { background: #059669; }
  .btn-bad    { background: #7f1d1d; color: #fca5a5; }
  .btn-bad:hover { background: #b91c1c; }
  .btn-reset  { background: #1e293b; color: #94a3b8; border: 1px solid #475569; width: 100%; margin-top: 10px; }
  .btn-reset:hover { background: #334155; }
  .state-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #1e293b; font-size: 0.85rem; }
  .state-row:last-child { border-bottom: none; }
  .state-key  { color: #64748b; }
  .state-val  { color: #e2e8f0; font-weight: 600; }
  .sentiment-bar { height: 8px; background: #1e293b; border-radius: 999px; margin-top: 10px; overflow: hidden; }
  .sentiment-fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
  .score-row  { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #1e293b; }
  .score-row:last-child { border-bottom: none; }
  .score-badge { padding: 3px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; }
  .history-tag { display: inline-block; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; margin: 2px; color: #94a3b8; }
  .reward-toast { position: fixed; top: 20px; right: 20px; padding: 10px 18px; border-radius: 8px; font-weight: 700; font-size: 0.9rem; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
  .full-width { grid-column: 1 / -1; }
  a.docs-link { color: #60a5fa; text-decoration: none; font-size: 0.85rem; }
  a.docs-link:hover { text-decoration: underline; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
  .tag-easy   { background: #166534; color: #86efac; }
  .tag-medium { background: #854d0e; color: #fde68a; }
  .tag-hard   { background: #7f1d1d; color: #fca5a5; }
</style>
</head>
<body>
<div class="header">
  <h1>🎧 E-commerce Customer Support Agent <span class="badge">OpenEnv</span></h1>
  <p>An AI agent resolves customer support tickets by choosing the right sequence of actions. Try it live below, or use the <a class="docs-link" href="/docs">API docs ↗</a></p>
</div>
<div class="container">

  <!-- Ticket -->
  <div class="card">
    <h2>📋 Customer Ticket</h2>
    <div class="ticket-box">
      <div class="label">Subject</div>
      <div class="value">My laptop arrived with a cracked screen</div>
      <div class="label">Customer</div>
      <div class="value">Alex &nbsp;·&nbsp; Regular tier &nbsp;·&nbsp; Order $999</div>
      <div class="label">Message</div>
      <div class="value">I ordered a laptop (Order #ORD-98765, $999) but it arrived with a cracked screen. I have photos as proof. I'd like a refund or replacement please.</div>
    </div>
    <div style="margin-top:14px; font-size:0.8rem; color:#64748b;">
      💡 Optimal: <code style="color:#93c5fd">acknowledge → investigate → offer_refund → resolve</code>
    </div>
  </div>

  <!-- Actions -->
  <div class="card">
    <h2>⚡ Actions</h2>
    <div class="actions-grid">
      <button class="btn btn-action" onclick="doStep('acknowledge')">👋 acknowledge</button>
      <button class="btn btn-action" onclick="doStep('investigate')">🔍 investigate</button>
      <button class="btn btn-action" onclick="doStep('offer_refund')">💰 offer_refund</button>
      <button class="btn btn-action" onclick="doStep('offer_exchange')">🔄 offer_exchange</button>
      <button class="btn btn-action" onclick="doStep('apply_discount')">🏷️ apply_discount</button>
      <button class="btn btn-bad"    onclick="doStep('escalate')">⬆️ escalate</button>
      <button class="btn btn-bad"    onclick="doStep('request_info')">❓ request_info</button>
      <button class="btn btn-resolve" onclick="doStep('resolve')">✅ resolve</button>
    </div>
    <button class="btn btn-reset" onclick="doReset()">🔄 Reset Episode</button>
  </div>

  <!-- State -->
  <div class="card">
    <h2>📊 Current State</h2>
    <div>
      <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="color:#64748b;font-size:0.8rem;">Customer Sentiment</span>
        <span id="sentimentVal" style="color:#e2e8f0;font-weight:700;font-size:0.8rem;">0.30</span>
      </div>
      <div class="sentiment-bar"><div id="sentimentFill" class="sentiment-fill" style="width:30%;background:#ef4444;"></div></div>
    </div>
    <div style="margin-top:14px;">
      <div class="state-row"><span class="state-key">Investigated</span><span id="sInv" class="state-val">❌</span></div>
      <div class="state-row"><span class="state-key">Refund offered</span><span id="sRef" class="state-val">❌</span></div>
      <div class="state-row"><span class="state-key">Exchange offered</span><span id="sExc" class="state-val">❌</span></div>
      <div class="state-row"><span class="state-key">Escalated</span><span id="sEsc" class="state-val">❌</span></div>
      <div class="state-row"><span class="state-key">Resolved</span><span id="sRes" class="state-val">❌</span></div>
      <div class="state-row"><span class="state-key">Satisfaction</span><span id="sSat" class="state-val">—</span></div>
      <div class="state-row"><span class="state-key">Steps</span><span id="sSteps" class="state-val">0</span></div>
    </div>
    <div style="margin-top:12px;">
      <div style="color:#64748b;font-size:0.75rem;margin-bottom:6px;">ACTION HISTORY</div>
      <div id="history"></div>
    </div>
    <div style="margin-top:10px;font-size:0.8rem;color:#64748b;">Last reward: <span id="lastReward" style="color:#fbbf24;font-weight:700;">—</span></div>
  </div>

  <!-- Grader -->
  <div class="card">
    <h2>🏆 Grader Scores</h2>
    <div class="score-row">
      <span><span class="tag tag-easy">EASY</span> &nbsp; Resolve the ticket</span>
      <span id="scoreEasy" class="score-badge" style="background:#166534;color:#86efac;">0.00</span>
    </div>
    <div class="score-row">
      <span><span class="tag tag-medium">MEDIUM</span> &nbsp; High satisfaction</span>
      <span id="scoreMedium" class="score-badge" style="background:#854d0e;color:#fde68a;">0.00</span>
    </div>
    <div class="score-row">
      <span><span class="tag tag-hard">HARD</span> &nbsp; ≤5 steps, no escalation</span>
      <span id="scoreHard" class="score-badge" style="background:#7f1d1d;color:#fca5a5;">0.00</span>
    </div>
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid #334155;">
      <div style="color:#64748b;font-size:0.75rem;margin-bottom:8px;">BASELINE (optimal policy)</div>
      <div style="font-size:0.8rem;color:#94a3b8;">acknowledge → investigate → offer_refund → resolve</div>
      <div style="font-size:0.8rem;color:#86efac;margin-top:4px;">→ easy: 1.0 &nbsp;|&nbsp; medium: 0.90 &nbsp;|&nbsp; hard: 1.0</div>
    </div>
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid #334155;font-size:0.8rem;color:#64748b;">
      API: <a class="docs-link" href="/docs">/docs</a> &nbsp;·&nbsp;
      <a class="docs-link" href="/tasks">/tasks</a> &nbsp;·&nbsp;
      <a class="docs-link" href="/schema">/schema</a> &nbsp;·&nbsp;
      <a class="docs-link" href="/grader">/grader</a>
    </div>
  </div>

</div>

<div id="toast" class="reward-toast"></div>

<script>
let steps = 0;

function showToast(msg, color) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.background = color;
  t.style.color = '#fff';
  t.style.opacity = '1';
  setTimeout(() => t.style.opacity = '0', 1800);
}

function sentimentColor(v) {
  if (v >= 0.7) return '#22c55e';
  if (v >= 0.5) return '#f59e0b';
  return '#ef4444';
}

function updateState(obs, reward) {
  const s = obs.observation || obs;
  const sentiment = s.sentiment || 0;
  document.getElementById('sentimentVal').textContent = sentiment.toFixed(2);
  document.getElementById('sentimentFill').style.width = (sentiment * 100) + '%';
  document.getElementById('sentimentFill').style.background = sentimentColor(sentiment);
  document.getElementById('sInv').textContent  = s.investigated   ? '✅' : '❌';
  document.getElementById('sRef').textContent  = s.refund_offered  ? '✅' : '❌';
  document.getElementById('sExc').textContent  = s.exchange_offered? '✅' : '❌';
  document.getElementById('sEsc').textContent  = s.escalated       ? '⚠️' : '❌';
  document.getElementById('sRes').textContent  = s.resolved        ? '✅' : '❌';
  document.getElementById('sSat').textContent  = s.resolved ? s.satisfaction_score.toFixed(2) : '—';
  document.getElementById('sSteps').textContent = steps;
  if (reward !== undefined) {
    const r = obs.reward !== undefined ? obs.reward : reward;
    document.getElementById('lastReward').textContent = (r >= 0 ? '+' : '') + Number(r).toFixed(2);
  }
}

function addHistory(action) {
  const el = document.getElementById('history');
  const tag = document.createElement('span');
  tag.className = 'history-tag';
  tag.textContent = action;
  el.appendChild(tag);
}

async function doReset() {
  steps = 0;
  document.getElementById('history').innerHTML = '';
  document.getElementById('lastReward').textContent = '—';
  const r = await fetch('/reset', {method:'POST'});
  const data = await r.json();
  updateState(data, 0);
  await refreshGrader();
  showToast('Episode reset', '#1e40af');
}

async function doStep(action) {
  steps++;
  addHistory(action);
  const r = await fetch('/step', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action: {action: action}})
  });
  const data = await r.json();
  updateState(data);
  const reward = data.reward;
  showToast(action + '  ' + (reward >= 0 ? '+' : '') + Number(reward).toFixed(2), reward >= 0 ? '#065f46' : '#7f1d1d');
  await refreshGrader();
  if (data.done) showToast('🎉 Episode complete! Score: ' + (data.observation||data).satisfaction_score?.toFixed(2), '#1e40af');
}

async function refreshGrader() {
  const r = await fetch('/grader');
  const g = await r.json();
  document.getElementById('scoreEasy').textContent   = g.easy.toFixed(2);
  document.getElementById('scoreMedium').textContent = g.medium.toFixed(2);
  document.getElementById('scoreHard').textContent   = g.hard.toFixed(2);
}

// Init
doReset();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def ui() -> HTMLResponse:
    return HTMLResponse(content=_UI_HTML)


@app.get("/tasks")
async def get_tasks() -> dict[str, Any]:
    return {"tasks": TASKS, "action_schema": ACTION_SCHEMA}


@app.get("/grader")
async def get_grader() -> dict[str, float]:
    return {
        "easy": grade_easy(),
        "medium": grade_medium(),
        "hard": grade_hard(),
    }


@app.post("/baseline")
async def run_baseline() -> dict[str, Any]:
    """Run the optimal policy and return episode + per-task scores."""
    env = CustomerSupportEnv()
    env.reset()
    episode: list[dict[str, Any]] = []
    for action in OPTIMAL_POLICY:
        act = SupportAction(action=action)
        obs = env.step(act)
        episode.append({"action": action, "sentiment": obs.sentiment, "reward": obs.reward, "done": obs.done})

    return {
        "policy": " → ".join(OPTIMAL_POLICY),
        "episode": episode,
        "scores": {
            "easy": grade_easy(),
            "medium": grade_medium(),
            "hard": grade_hard(),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.host == "0.0.0.0" and args.port == 7860:
        main()
    else:
        main(host=args.host, port=args.port)
