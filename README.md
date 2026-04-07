---
title: Ecommerce Env
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.11"
startup_duration_timeout: 30m
tags:
  - openenv
---

# E-commerce Customer Support Agent — OpenEnv Environment

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Space-blue)](https://y0120-ecommerce-env.hf.space)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Hackathon%202026-orange)](https://huggingface.co/spaces/Y0120/ecommerce-env)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)

> 🚀 **[Try it live →](https://y0120-ecommerce-env.hf.space)**

An **OpenEnv**-compliant RL environment where an AI agent plays the role of a **customer support representative** for an e-commerce store. Each episode the agent receives a **random real-world customer ticket**, must read it carefully to understand the issue, and then choose the right sequence of actions to resolve it efficiently.

> **Why this matters:** AI-powered customer support is one of the fastest-growing applications of LLM agents. This environment tests whether an agent can *understand context*, *match resolution to problem type*, and *manage customer sentiment* — skills that directly transfer to production support systems.

---

## 5 Ticket Scenarios (randomly sampled per episode)

| Ticket ID | Type | Customer | Tier | Order Value | Optimal Resolution |
|---|---|---|---|---|---|
| TKT-001 | `damaged_item` | Alex | regular | $999 | `offer_refund` / `offer_exchange` |
| TKT-002 | `wrong_item` | Jordan | **VIP** | $249 | `offer_exchange` / `offer_refund` |
| TKT-003 | `missing_item` | Sam | new | $149 | `offer_refund` / `offer_exchange` |
| TKT-004 | `late_delivery` | Casey | regular | $89 | `send_update` + `apply_discount` |
| TKT-005 | `billing_issue` | Riley | **VIP** | $199 | `investigate` → `offer_refund` |

Each ticket has a unique **opening message** and **customer responses per action** — making every episode feel like a real conversation.

The agent **must read the ticket type** to choose the correct resolution. Using `offer_refund` on a `late_delivery` ticket gives reduced reward; using `send_update` on a `billing_issue` gives a negative reward. This tests genuine understanding, not memorisation.

---

## Action Space (9 discrete actions)

| Action | Description | Base Reward | Notes |
|---|---|---|---|
| `acknowledge` | Acknowledge the issue — builds trust | `+0.10` | Penalised if repeated |
| `investigate` | Look up order/account details | `+0.10` | **Unlocks full resolution rewards** |
| `offer_refund` | Issue full refund | `+0.50` (after investigate) | ✅ Correct for: damaged, missing, wrong, billing |
| `offer_exchange` | Send replacement item | `+0.45` (after investigate) | ✅ Correct for: wrong, damaged |
| `apply_discount` | 10% off next order | `+0.30` (correct) / `+0.08` | ✅ Correct for: late_delivery |
| `send_update` | Send delivery status update | `+0.40` (correct) / `−0.10` | ✅ Correct for: late_delivery only |
| `escalate` | Transfer to senior agent | `−0.30` (regular) / `−0.50` (**VIP**) | Frustrates customer; extra penalty for VIP |
| `request_info` | Ask for info already provided | `−0.10` | Penalises agent for lazy behaviour |
| `resolve` | Close the ticket | `= final sentiment` | Only meaningful after addressing root cause |

**Step cost:** `−0.01` per step — encourages efficient resolution over stalling.

---

## Observation Space

```json
{
  "ticket_id": "TKT-004",
  "ticket_type": "late_delivery",
  "ticket_subject": "Order still hasn't arrived — it's been 10 days",
  "ticket_description": "I ordered headphones (Order #ORD-34876, $89) 10 days ago with 5-day delivery...",
  "customer_name": "Casey",
  "customer_tier": "regular",
  "order_value": 89.0,
  "correct_resolutions": ["send_update", "apply_discount"],
  "sentiment": 0.55,
  "investigated": true,
  "refund_offered": false,
  "exchange_offered": false,
  "discount_applied": false,
  "update_sent": true,
  "escalated": false,
  "resolved": false,
  "satisfaction_score": 0.0,
  "correct_resolution_used": true,
  "customer_response": "Thank you for checking! That's really helpful to know.",
  "reward": 0.40,
  "done": false
}
```

The observation includes the **full ticket text**, **customer's latest response**, and **which resolutions are correct for this ticket type** — giving the agent everything it needs to act intelligently.

---

## Tasks & Graders (4 levels)

| ID | Objective | Scoring |
|---|---|---|
| `easy` | Resolve the ticket — any resolution counts | `1.0` if resolved, `0.0` otherwise |
| `medium` | Resolve with meaningful customer satisfaction | `satisfaction_score` if resolved (min 0.3) |
| `hard` | Resolve correctly and efficiently | See below |
| `expert` | Near-perfect: correct action + max satisfaction + minimum steps | See below |

**Hard task scoring:**

| Conditions | Score |
|---|---|
| Correct resolution + satisfaction ≥ 0.7 + steps ≤ 6 + no escalation | **1.0** |
| Correct resolution + satisfaction ≥ 0.6 + no escalation | **0.7** |
| Correct resolution used + no escalation | **0.5** |
| Resolved but wrong resolution or escalated | **0.3** |
| Not resolved | **0.0** |

**Expert task scoring:**

| Conditions | Score |
|---|---|
| Correct resolution + satisfaction ≥ 0.8 + steps ≤ 4 + no escalation | **1.0** |
| Correct resolution + satisfaction ≥ 0.7 + steps ≤ 5 + no escalation | **0.6** |
| Correct resolution + no escalation (any satisfaction/steps) | **0.3** |
| Wrong resolution, unresolved, or customer hung up | **0.0** |

The expert task is genuinely hard: `late_delivery` needs 5 steps (maximum 0.6), and `billing_issue` starts at very low sentiment (0.15). Even the deterministic optimal policy scores 0.6 on these — frontier models must figure out the context and act with maximum precision.

---

## Baseline Scores (Optimal Policy)

| Ticket Type | Optimal Policy | Easy | Medium | Hard | Expert |
|---|---|---|---|---|---|
| `damaged_item` | ack → investigate → offer_refund → resolve | 1.0 | 0.90 | 1.0 | **1.0** |
| `wrong_item` | ack → investigate → offer_exchange → resolve | 1.0 | 0.90 | 1.0 | **1.0** |
| `missing_item` | ack → investigate → offer_refund → resolve | 1.0 | 0.85 | 1.0 | **1.0** |
| `late_delivery` | ack → investigate → send_update → apply_discount → resolve | 1.0 | 0.90 | 1.0 | **0.6** |
| `billing_issue` | ack → investigate → offer_refund → resolve | 1.0 | 0.75 | 1.0 | **0.6** |

`late_delivery` requires 5 steps by design (cannot achieve expert=1.0). `billing_issue` starts at very low sentiment (0.15), making satisfaction ≥ 0.8 unreachable in 4 steps — the expert task is *genuinely hard*.

---

## Reward Design

Rewards are shaped across the **full trajectory** — not just at episode end:

1. **Per-action partial rewards** — `acknowledge` and `investigate` give immediate signal even early in the episode.
2. **Context-dependent bonus** — `offer_refund` gives `+0.50` only if `investigated` is `True`, otherwise `+0.20`. Forces the agent to build context before acting.
3. **Ticket-type matching** — correct resolutions give full reward; wrong resolutions give partial or negative reward. Agent must reason about ticket content.
4. **Escalation penalty** — `−0.30` for regular customers, `−0.50` for VIP customers. Teaches sensitivity to customer tier.
5. **Repetition penalty** — repeating the same action gives negative reward (`−0.05` to `−0.10`).
6. **Step cost** — `−0.01` per step discourages padding and rewards efficient resolution.
7. **Terminal reward** — `resolve` returns the current sentiment score (0.0–1.0) as a dense continuous signal.
8. **Impatience decay** — After step 5, customer sentiment decreases `−0.05` per step. The longer the agent stalls, the harder it becomes to achieve high satisfaction.
9. **Customer hang-up** — If sentiment drops to ≤ 0.05 (from escalating VIP customers, repeated bad actions, or impatience), the customer files a chargeback and the episode ends with `done=True` and a `−0.50` penalty. Models must learn not to compound mistakes.

This design ensures gradients flow throughout the episode, making the environment suitable for policy gradient RL methods, not just supervised/imitation learning.

---

## HTTP API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Interactive UI — playable in browser |
| `GET` | `/tasks` | Task list + full action schema |
| `POST` | `/reset` | Start new episode (samples random ticket) |
| `POST` | `/step` | Execute `{"action": {"action": "send_update"}}` |
| `GET` | `/state` | Current episode state |
| `GET` | `/grader` | Live grader scores (easy / medium / hard) |
| `POST` | `/baseline` | Run optimal policy for all 5 ticket types |
| `GET` | `/docs` | Swagger UI |

---

## Setup

### Docker
```bash
docker build -t ecommerce-support-env .
docker run -p 7860:7860 ecommerce-support-env
# → http://localhost:7860
```

### Python (dev)
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Baseline (deterministic, no LLM)
```bash
python baseline.py
# Runs optimal policy for all 5 ticket types
# → easy: 1.0, medium: 0.75–0.90, hard: 1.0 (all scenarios)
```

### Inference (LLM agent)
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
# Runs 3 tasks; prints per-task scores
```

---

## Project Structure

```
ecommerce-env/
├── env.py                      # Simulator: 5 ticket scenarios, reward logic, RNG
├── models.py                   # Pydantic types: SupportAction, SupportObservation, SupportEnvState
├── grader.py                   # Graders: grade_easy(), grade_medium(), grade_hard()
├── ecommerce_environment.py    # Top-level environment wrapper
├── baseline.py                 # Deterministic optimal-policy baseline
├── inference.py                # LLM agent (OpenAI client)
├── server/
│   ├── app.py                  # FastAPI app: all endpoints + interactive HTML UI
│   └── ecommerce_environment.py # OpenEnv-compliant Environment subclass
├── Dockerfile
├── openenv.yaml
├── pyproject.toml
└── requirements.txt
```
