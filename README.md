---
title: Ecommerce Env
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
python_version: "3.11"
startup_duration_timeout: 30m
tags:
  - openenv
---

# E-commerce Customer Support Agent Environment

An **OpenEnv**-compliant environment where an AI agent acts as a **customer service representative** for an e-commerce store. The agent receives a real customer support ticket and must resolve it by choosing the right sequence of actions — balancing empathy, investigation, and resolution quality.

> **Why this matters:** AI-powered customer support is one of the fastest-growing applications of LLM agents. This environment lets you train and evaluate agents on a realistic, consequential task that humans do every day.

---

## The Scenario

A customer purchased a laptop (Order #ORD-98765, $999). It arrived with a **cracked screen**. They've submitted a ticket demanding resolution.

The agent starts with a frustrated customer (sentiment = 0.3 / 10) and must bring them to satisfaction through careful, empathetic action.

---

## Action Space (8 discrete actions)

| Action | Effect | Reward |
|---|---|---|
| `acknowledge` | Acknowledge the issue — builds immediate trust | +0.1 |
| `investigate` | Look up order details — **unlocks better resolution options** | +0.1 |
| `offer_refund` | Full refund — best outcome if investigated first | +0.5 (investigated) / +0.2 |
| `offer_exchange` | Send replacement — good alternative to refund | +0.4 (investigated) / +0.1 |
| `apply_discount` | Goodwill 10% off next order | +0.1 |
| `escalate` | Transfer to senior agent — frustrates customer | −0.3 |
| `request_info` | Ask for info already in the ticket — annoys customer | −0.1 |
| `resolve` | Close ticket — reward = final customer sentiment (0.0–1.0) | 0.0–1.0 |

**Optimal policy:** `acknowledge → investigate → offer_refund → resolve`
→ Final sentiment: **0.90** | Steps: **4** | Score: **1.0 / 1.0 / 1.0**

---

## Observation Space

```json
{
  "ticket_type": "damaged_item",
  "ticket_subject": "My laptop arrived with a cracked screen",
  "ticket_description": "...",
  "customer_name": "Alex",
  "customer_tier": "regular",
  "order_value": 999.0,
  "sentiment": 0.75,
  "investigated": true,
  "refund_offered": false,
  "exchange_offered": false,
  "discount_applied": false,
  "escalated": false,
  "resolved": false,
  "satisfaction_score": 0.0,
  "reward": 0.1,
  "done": false
}
```

---

## Tasks (Easy → Hard)

| ID | Description | Scoring |
|---|---|---|
| `easy` | Resolve the ticket — any resolution counts | 1.0 if resolved |
| `medium` | Resolve with meaningful customer satisfaction | satisfaction_score if resolved |
| `hard` | Resolve efficiently: satisfaction ≥ 0.8, ≤ 5 steps, no escalation | 1.0 / 0.7 / 0.5 / 0.2 |

---

## HTTP API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | Task list + action schema |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Execute `{"action": {"action": "acknowledge"}}` |
| `GET` | `/state` | Current episode state |
| `GET` | `/grader` | Live grader scores (easy / medium / hard) |
| `POST` | `/baseline` | Run optimal policy, return scores |
| `GET` | `/docs` | Interactive API docs (Swagger) |

---

## Setup

### Docker
```bash
docker build -t ecommerce-support-env .
docker run -p 7860:7860 ecommerce-support-env
```

### Python
```bash
pip install -r requirements.txt
uvicorn server.app:app --port 7860
```

### Baseline (no LLM)
```bash
python baseline.py
```

### Inference (LLM agent)
```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

**Expected baseline scores:**
```json
{ "easy": 1.0, "medium": 0.9, "hard": 1.0 }
```

---

## Reward Design

Rewards are shaped across the full trajectory — not just at episode end:
- **Partial rewards** for each beneficial action (acknowledge, investigate)
- **Multiplicative bonus** for combining investigate + offer_refund (more confident resolution)
- **Penalties** for bad practices (escalating unnecessarily, asking for info already given, repeating actions)
- **Terminal reward** = final customer sentiment (continuous signal, 0.0–1.0)

This makes the reward signal dense and informative for RL training.
