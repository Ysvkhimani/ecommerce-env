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
    from ..grader import grade_easy, grade_hard, grade_medium, grade_expert
    from ..env import OPTIMAL_POLICY, OPTIMAL_POLICIES, TICKET_SCENARIOS
except ImportError:
    from server.ecommerce_environment import CustomerSupportEnv
    from models import SupportAction, SupportObservation
    from grader import grade_easy, grade_hard, grade_medium, grade_expert
    from env import OPTIMAL_POLICY, OPTIMAL_POLICIES, TICKET_SCENARIOS

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
        "description": "Resolve with meaningful customer satisfaction",
        "difficulty": "medium",
        "scoring": "satisfaction_score if resolved (min 0.3 for any resolution)",
    },
    {
        "id": "hard",
        "description": (
            "Resolve correctly: use the right action for the ticket type, "
            "satisfaction ≥ 0.7, ≤ 6 steps, no escalation. "
            "Agent must read the ticket and infer the correct resolution."
        ),
        "difficulty": "hard",
        "scoring": "1.0 (correct+efficient), 0.7 (correct+satisfied), 0.5 (correct), 0.3 (wrong action)",
    },
    {
        "id": "expert",
        "description": (
            "Near-perfect resolution: correct resolution for ticket type, "
            "satisfaction ≥ 0.8, ≤ 4 steps, no escalation, no customer hang-up. "
            "Forces the agent to act decisively with maximum efficiency."
        ),
        "difficulty": "expert",
        "scoring": "1.0 (perfect), 0.6 (correct+satisfied+≤5 steps), 0.3 (correct but slow), 0.0 (wrong/unresolved)",
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
            "send_update",
            "escalate",
            "request_info",
            "resolve",
        ],
        "description": "Support action to execute",
    },
    "metadata": {"type": "object", "description": "Optional metadata", "default": {}},
}

TICKET_TYPE_LABELS = {
    "damaged_item":  "📦 Damaged Item",
    "wrong_item":    "🔀 Wrong Item",
    "missing_item":  "🔍 Missing Item",
    "late_delivery": "🕐 Late Delivery",
    "billing_issue": "💳 Billing Issue",
}

# ---------------------------------------------------------------------------
# Hackathon-required endpoints
# ---------------------------------------------------------------------------


_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E-commerce Customer Support Agent — OpenEnv</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.hdr{background:linear-gradient(135deg,#1e3a5f,#0f2044);padding:14px 20px;border-bottom:1px solid #334155;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px}
.hdr h1{font-size:1.1rem;color:#93c5fd}
.hdr p{color:#94a3b8;font-size:0.78rem;margin-top:3px}
.badge{background:#1e40af;color:#bfdbfe;padding:2px 9px;border-radius:999px;font-size:0.7rem;margin-left:8px}
.hdr-links a{color:#60a5fa;font-size:0.78rem;text-decoration:none;margin-left:10px}
.wrap{display:grid;grid-template-columns:1fr 360px;gap:14px;padding:14px 20px;max-width:1300px}
@media(max-width:800px){
  .wrap{grid-template-columns:1fr;padding:10px 12px}
  .hdr{padding:12px 14px}
  .hdr h1{font-size:1rem}
  .hdr-links{width:100%;display:flex;flex-wrap:wrap;gap:6px}
  .hdr-links a{margin-left:0}
  .actions-grid{grid-template-columns:1fr 1fr}
  .conv{max-height:220px}
  .card{padding:12px}
}
.card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px}
.card h2{color:#93c5fd;font-size:0.9rem;margin-bottom:12px}
/* Ticket header */
.ticket-hdr{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.type-pill{padding:3px 10px;border-radius:999px;font-size:0.72rem;font-weight:700}
.tier-vip{background:#713f12;color:#fde68a} .tier-regular{background:#1e3a5f;color:#bfdbfe} .tier-new{background:#14532d;color:#86efac}
.ticket-subject{font-size:1rem;font-weight:700;color:#e2e8f0;margin-bottom:6px}
.ticket-meta{font-size:0.78rem;color:#64748b;margin-bottom:10px}
/* Conversation */
.conv{max-height:320px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding:4px 0}
.msg{max-width:85%;padding:9px 13px;border-radius:12px;font-size:0.83rem;line-height:1.5}
.msg-customer{background:#1e3a5f;color:#bfdbfe;align-self:flex-start;border-radius:12px 12px 12px 3px}
.msg-agent{background:#14532d;color:#86efac;align-self:flex-end;border-radius:12px 12px 3px 12px}
.msg-label{font-size:0.68rem;opacity:0.7;margin-bottom:2px}
/* Actions */
.actions-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}
.btn{padding:9px 12px;border:none;border-radius:7px;cursor:pointer;font-size:0.8rem;font-weight:600;transition:all 0.15s;text-align:left}
.btn-good{background:#1e3a5f;color:#93c5fd}.btn-good:hover{background:#1e40af;transform:translateY(-1px)}
.btn-resolve{background:#064e3b;color:#6ee7b7}.btn-resolve:hover{background:#065f46}
.btn-bad{background:#450a0a;color:#fca5a5}.btn-bad:hover{background:#7f1d1d}
.btn-delivery{background:#3b2100;color:#fcd34d}.btn-delivery:hover{background:#78350f}
.btn-reset{background:transparent;color:#64748b;border:1px solid #334155;width:100%;margin-top:9px;text-align:center}
.btn-reset:hover{background:#334155;color:#e2e8f0}
/* Sentiment */
.sent-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}
.sent-bar{height:7px;background:#0f172a;border-radius:999px;overflow:hidden;margin-bottom:12px}
.sent-fill{height:100%;border-radius:999px;transition:width 0.4s,background 0.4s}
/* State rows */
.sr{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #0f172a;font-size:0.82rem}
.sr:last-child{border-bottom:none}
.sk{color:#64748b}.sv{font-weight:700}
/* Scores */
.score-row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #0f172a;font-size:0.83rem}
.score-row:last-child{border-bottom:none}
.sbadge{padding:2px 11px;border-radius:999px;font-size:0.8rem;font-weight:700}
.tag{display:inline-block;padding:1px 7px;border-radius:4px;font-size:0.7rem;font-weight:700}
.tag-e{background:#166534;color:#86efac}.tag-m{background:#854d0e;color:#fde68a}.tag-h{background:#7f1d1d;color:#fca5a5}
.htag{display:inline-block;background:#0f172a;border:1px solid #334155;border-radius:5px;padding:1px 7px;font-size:0.72rem;margin:2px;color:#94a3b8}
.toast{position:fixed;top:18px;right:18px;padding:9px 16px;border-radius:7px;font-weight:700;font-size:0.88rem;opacity:0;transition:opacity 0.3s;pointer-events:none;z-index:999}
.hint{font-size:0.75rem;color:#475569;margin-top:8px;line-height:1.5}
code{color:#93c5fd;font-family:monospace}
.correct-badge{display:inline-block;background:#065f46;color:#6ee7b7;padding:2px 9px;border-radius:999px;font-size:0.72rem;margin-left:6px}
.wrong-badge{display:inline-block;background:#7f1d1d;color:#fca5a5;padding:2px 9px;border-radius:999px;font-size:0.72rem;margin-left:6px}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1>🎧 E-commerce Customer Support Agent <span class="badge">OpenEnv</span></h1>
    <p>AI agent resolves real customer tickets — 5 ticket types, live conversation, graded performance</p>
  </div>
  <div class="hdr-links">
    <a href="/docs">API Docs ↗</a>
    <a href="/tasks">/tasks</a>
    <a href="/grader">/grader</a>
    <a href="/baseline">/baseline</a>
  </div>
</div>

<div class="wrap">
  <!-- LEFT: Ticket + Conversation + Actions -->
  <div style="display:flex;flex-direction:column;gap:14px">

    <!-- Ticket card -->
    <div class="card">
      <h2>📋 Customer Ticket</h2>
      <div class="ticket-hdr">
        <span id="typePill" class="type-pill tier-regular">damaged_item</span>
        <span id="tierPill" class="type-pill tier-regular">regular</span>
        <span id="ticketId" style="font-size:0.75rem;color:#475569">TKT-001</span>
      </div>
      <div id="ticketSubject" class="ticket-subject">—</div>
      <div id="ticketMeta" class="ticket-meta">—</div>
      <div id="ticketDesc" style="font-size:0.83rem;color:#94a3b8;line-height:1.6">—</div>
      <div id="hintBox" class="hint"></div>
    </div>

    <!-- Conversation thread -->
    <div class="card">
      <h2>💬 Conversation Thread</h2>
      <div id="conv" class="conv"></div>
    </div>

    <!-- Actions -->
    <div class="card">
      <h2>⚡ Agent Actions</h2>
      <div class="actions-grid">
        <button class="btn btn-good"     onclick="doStep('acknowledge')">👋 acknowledge</button>
        <button class="btn btn-good"     onclick="doStep('investigate')">🔍 investigate</button>
        <button class="btn btn-good"     onclick="doStep('offer_refund')">💰 offer_refund</button>
        <button class="btn btn-good"     onclick="doStep('offer_exchange')">🔄 offer_exchange</button>
        <button class="btn btn-good"     onclick="doStep('apply_discount')">🏷️ apply_discount</button>
        <button class="btn btn-delivery" onclick="doStep('send_update')">📦 send_update</button>
        <button class="btn btn-bad"      onclick="doStep('escalate')">⬆️ escalate</button>
        <button class="btn btn-bad"      onclick="doStep('request_info')">❓ request_info</button>
      </div>
      <button class="btn btn-resolve" onclick="doStep('resolve')" style="width:100%;margin-top:8px;text-align:center">✅ resolve ticket</button>
      <button class="btn btn-reset"   onclick="doReset()">🔄 New Episode (random ticket)</button>
    </div>
  </div>

  <!-- RIGHT: State + Scores -->
  <div style="display:flex;flex-direction:column;gap:14px">

    <!-- Sentiment -->
    <div class="card">
      <h2>😊 Customer Sentiment</h2>
      <div class="sent-row">
        <span style="font-size:0.78rem;color:#64748b">0 = very angry</span>
        <span id="sentVal" style="font-size:1.3rem;font-weight:700;color:#e2e8f0">0.30</span>
        <span style="font-size:0.78rem;color:#64748b">1 = very happy</span>
      </div>
      <div class="sent-bar"><div id="sentFill" class="sent-fill" style="width:30%;background:#ef4444"></div></div>
      <div class="sr"><span class="sk">Investigated</span><span id="sInv" class="sv">❌</span></div>
      <div class="sr"><span class="sk">Refund offered</span><span id="sRef" class="sv">❌</span></div>
      <div class="sr"><span class="sk">Exchange offered</span><span id="sExc" class="sv">❌</span></div>
      <div class="sr"><span class="sk">Update sent</span><span id="sUpd" class="sv">❌</span></div>
      <div class="sr"><span class="sk">Escalated</span><span id="sEsc" class="sv">❌</span></div>
      <div class="sr"><span class="sk">✓ Correct resolution</span><span id="sCorr" class="sv">—</span></div>
      <div class="sr"><span class="sk">Resolved</span><span id="sRes" class="sv">❌</span></div>
      <div class="sr"><span class="sk">Satisfaction score</span><span id="sSat" class="sv">—</span></div>
      <div class="sr"><span class="sk">Steps</span><span id="sSteps" class="sv">0</span></div>
      <div class="sr"><span class="sk">Last reward</span><span id="sRew" class="sv" style="color:#fbbf24">—</span></div>
      <div style="margin-top:10px;font-size:0.75rem;color:#475569">History:</div>
      <div id="hist" style="margin-top:4px"></div>
    </div>

    <!-- Grader -->
    <div class="card">
      <h2>🏆 Grader Scores</h2>
        <div class="score-row"><span><span class="tag tag-e">EASY</span> resolve ticket</span><span id="gEasy" class="sbadge" style="background:#166534;color:#86efac">0.00</span></div>
        <div class="score-row"><span><span class="tag tag-m">MEDIUM</span> satisfaction score</span><span id="gMed" class="sbadge" style="background:#854d0e;color:#fde68a">0.00</span></div>
        <div class="score-row"><span><span class="tag tag-h">HARD</span> correct + efficient</span><span id="gHard" class="sbadge" style="background:#7f1d1d;color:#fca5a5">0.00</span></div>
        <div class="score-row"><span><span class="tag" style="background:#3b0764;color:#e9d5ff">EXPERT</span> ≤4 steps + sat≥0.8</span><span id="gExpert" class="sbadge" style="background:#3b0764;color:#e9d5ff">0.00</span></div>
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid #334155;font-size:0.75rem;color:#64748b">
        <div style="margin-bottom:6px;font-weight:600;color:#94a3b8">CORRECT ACTION PER TICKET TYPE</div>
        <div>📦 damaged_item → <code>offer_refund</code> / <code>offer_exchange</code></div>
        <div>🔀 wrong_item → <code>offer_exchange</code> / <code>offer_refund</code></div>
        <div>🔍 missing_item → <code>offer_refund</code> / <code>offer_exchange</code></div>
        <div>🕐 late_delivery → <code>send_update</code> + <code>apply_discount</code></div>
        <div>💳 billing_issue → <code>investigate</code> then resolve</div>
      </div>
    </div>

  </div>
</div>

<div id="toast" class="toast"></div>

<script>
let steps=0, ticketType='';

const TIER_CLASS={vip:'tier-vip',regular:'tier-regular',new:'tier-new'};
const TYPE_EMOJI={damaged_item:'📦',wrong_item:'🔀',missing_item:'🔍',late_delivery:'🕐',billing_issue:'💳'};

function sc(v){return v>=0.7?'#22c55e':v>=0.5?'#f59e0b':'#ef4444'}

function toast(msg,col){
  const t=document.getElementById('toast');
  t.textContent=msg;t.style.background=col;t.style.color='#fff';t.style.opacity='1';
  setTimeout(()=>t.style.opacity='0',2000);
}

function addMsg(role, action, text){
  const c=document.getElementById('conv');
  const d=document.createElement('div');
  d.className='msg '+(role==='agent'?'msg-agent':'msg-customer');
  d.innerHTML=`<div class="msg-label">${role==='agent'?'🤖 Agent ['+action+']':'👤 '+text.split(' ')[0].replace(':','')}</div>${text}`;
  if(role==='agent') d.children[0].textContent='🤖 Agent ['+action+']';
  c.appendChild(d);
  c.scrollTop=c.scrollHeight;
}

function updatePanel(obs, reward){
  const s=obs.observation||obs;
  const sent=s.sentiment||0;
  document.getElementById('sentVal').textContent=sent.toFixed(2);
  document.getElementById('sentFill').style.width=(sent*100)+'%';
  document.getElementById('sentFill').style.background=sc(sent);
  document.getElementById('sInv').textContent=s.investigated?'✅':'❌';
  document.getElementById('sRef').textContent=s.refund_offered?'✅':'❌';
  document.getElementById('sExc').textContent=s.exchange_offered?'✅':'❌';
  document.getElementById('sUpd').textContent=s.update_sent?'✅':'❌';
  document.getElementById('sEsc').textContent=s.escalated?'⚠️':'❌';
  document.getElementById('sRes').textContent=s.resolved?'✅':'❌';
  document.getElementById('sSat').textContent=s.resolved?s.satisfaction_score.toFixed(2):'—';
  document.getElementById('sSteps').textContent=steps;
  const corr=s.correct_resolution_used;
  document.getElementById('sCorr').textContent=corr?'✅':'—';
  const r=obs.reward??reward;
  if(r!==undefined) document.getElementById('sRew').textContent=(r>=0?'+':'')+Number(r).toFixed(2);
}

async function refreshGrader(){
  const g=await(await fetch('/grader')).json();
  document.getElementById('gEasy').textContent=g.easy.toFixed(2);
  document.getElementById('gMed').textContent=g.medium.toFixed(2);
  document.getElementById('gHard').textContent=g.hard.toFixed(2);
  document.getElementById('gExpert').textContent=(g.expert||0).toFixed(2);
}

async function doReset(){
  steps=0;
  document.getElementById('conv').innerHTML='';
  document.getElementById('hist').innerHTML='';
  document.getElementById('sRew').textContent='—';
  const data=await(await fetch('/reset',{method:'POST'})).json();
  const s=data.observation||data;
  ticketType=s.ticket_type||'';

  // Update ticket card
  const emoji=TYPE_EMOJI[ticketType]||'🎫';
  document.getElementById('typePill').textContent=emoji+' '+ticketType;
  document.getElementById('typePill').className='type-pill tier-regular';
  document.getElementById('tierPill').textContent=s.customer_tier;
  document.getElementById('tierPill').className='type-pill '+(TIER_CLASS[s.customer_tier]||'tier-regular');
  document.getElementById('ticketId').textContent=s.ticket_id||'';
  document.getElementById('ticketSubject').textContent=s.ticket_subject||'';
  document.getElementById('ticketMeta').textContent=(s.customer_name||'')+'  ·  Order $'+(s.order_value||0)+'  ·  Correct action: '+(s.correct_resolutions||[]).join(' / ');
  document.getElementById('ticketDesc').textContent=s.ticket_description||'';

  // Opening message from customer
  addMsg('customer','',s.customer_response||s.opening_message||'...');
  updatePanel(data,0);
  await refreshGrader();
  toast('New episode: '+ticketType,'#1e40af');
}

async function doStep(action){
  steps++;
  // Agent message
  addMsg('agent',action,'Taking action: '+action+'...');
  const resp=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:{action}})});
  const data=await resp.json();
  const s=data.observation||data;
  const reward=data.reward;

  // Replace last agent message with result
  const msgs=document.getElementById('conv').querySelectorAll('.msg-agent');
  if(msgs.length) msgs[msgs.length-1].innerHTML='<div class="msg-label">🤖 Agent ['+action+']</div>Executed: <strong>'+action+'</strong>';

  // Customer response
  if(s.customer_response) addMsg('customer','',s.customer_response);

  // History tag
  const h=document.getElementById('hist');
  const tag=document.createElement('span');
  tag.className='htag';tag.textContent=action;h.appendChild(tag);

  updatePanel(data,reward);
  await refreshGrader();
  const col=reward>=0?'#065f46':'#7f1d1d';
  toast(action+'  '+(reward>=0?'+':'')+Number(reward).toFixed(2),col);
  if(data.done){
    const sat=s.satisfaction_score||0;
    if(s.resolved){
      const stars='⭐'.repeat(Math.round(sat*5));
      addMsg('customer','','[TICKET CLOSED] Satisfaction: '+sat.toFixed(2)+' '+stars);
      toast('🎉 Resolved! Score: '+sat.toFixed(2),'#1e40af');
    } else {
      addMsg('customer','','☎️ [CUSTOMER HUNG UP] They filed a chargeback. Score: 0.00');
      toast('💔 Customer hung up — episode over','#7f1d1d');
    }
  }
}

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
        "expert": grade_expert(),
    }


@app.post("/baseline")
async def run_baseline() -> dict[str, Any]:
    """Run optimal policy for each of the 5 ticket types and return all scores."""
    results = []
    for scenario in TICKET_SCENARIOS:
        ticket_type = scenario["type"]
        policy = OPTIMAL_POLICIES[ticket_type]

        env = CustomerSupportEnv()
        # Force the specific scenario
        env._sim._scenario = scenario
        sim = env._sim
        sim.state.clear()
        sim.state.update({
            "ticket_id": scenario["ticket_id"], "ticket_type": scenario["type"],
            "ticket_subject": scenario["subject"], "ticket_description": scenario["description"],
            "customer_name": scenario["customer_name"], "customer_tier": scenario["customer_tier"],
            "order_value": scenario["order_value"], "correct_resolutions": list(scenario["correct_resolutions"]),
            "opening_message": scenario["opening_message"], "sentiment": scenario["initial_sentiment"],
            "investigated": False, "refund_offered": False, "exchange_offered": False,
            "discount_applied": False, "update_sent": False, "escalated": False,
            "resolved": False, "satisfaction_score": 0.0, "correct_resolution_used": False,
            "customer_response": scenario["opening_message"],
        })
        sim.history.clear()
        from uuid import uuid4
        sim.episode_id = str(uuid4())

        episode: list[dict[str, Any]] = []
        for action in policy:
            act = SupportAction(action=action)
            obs = env.step(act)
            episode.append({"action": action, "sentiment": round(obs.sentiment, 2), "reward": round(obs.reward or 0, 2), "done": obs.done})

        results.append({
            "ticket_type": ticket_type,
            "policy": " → ".join(policy),
            "episode": episode,
            "scores": {"easy": grade_easy(), "medium": grade_medium(), "hard": grade_hard(), "expert": grade_expert()},
        })

    return {"scenarios": results}


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
