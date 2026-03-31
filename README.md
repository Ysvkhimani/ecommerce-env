---
title: Ecommerce Env
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
python_version: "3.11"
startup_duration_timeout: 30m
tags:
  - openenv
---

# Ecommerce OpenEnv

An **OpenEnv**-compliant e-commerce cart simulation where an AI agent learns to complete a shopping workflow through the standard `reset()` / `step()` / `state()` API.

## Environment Description

The agent operates a shopping cart with four discrete actions. It receives partial reward at each step and a terminal reward on successful checkout. Three graded tasks (easy → medium → hard) measure progressively stricter completion criteria.

## Action Space

| Action | Effect | Reward |
|---|---|---|
| `add_item` | Add a $100 item to cart | +0.2 |
| `apply_coupon` | Apply 10% discount (once per episode) | +0.3 (−0.1 if already used) |
| `checkout` | Proceed to checkout | +0.3 (−0.5 if cart empty) |
| `pay` | Complete payment | +1.0 (−1.0 if cart empty) |

## Observation Space

```json
{
  "cart": ["item"],
  "total": 90.0,
  "coupon_applied": true,
  "payment_done": false,
  "order_status": "incomplete",
  "reward": 0.3,
  "done": false
}
```

## Tasks

| ID | Description | Difficulty | Max Score |
|---|---|---|---|
| `easy` | Add item and pay | Easy | 1.0 |
| `medium` | Apply coupon then pay | Medium | 1.0 |
| `hard` | Exact sequence: add_item → apply_coupon → checkout → pay | Hard | 1.0 |

## HTTP API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | Task list + action schema |
| `POST` | `/reset` | Reset episode |
| `POST` | `/step` | Execute `{"action": "add_item"}` |
| `GET` | `/state` | Full environment state |
| `GET` | `/grader` | Grader scores (easy / medium / hard) |
| `POST` | `/baseline` | Run optimal policy, return scores |
| `GET` | `/web` | Interactive Gradio UI |
| `GET` | `/docs` | Auto-generated OpenAPI docs |

## Setup Instructions

### Docker (recommended)

```bash
docker build -t ecommerce-env .
docker run -p 7860:7860 ecommerce-env
# API: http://localhost:7860
# UI:  http://localhost:7860/web
```

### Python (local)

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py
```

Expected baseline scores (optimal policy):
```json
{
  "easy":   1.0,
  "medium": 1.0,
  "hard":   1.0
}
```

## Example API Calls

```bash
# Reset
curl -X POST http://localhost:7860/reset

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "add_item"}'

# Grader scores
curl http://localhost:7860/grader

# Run full baseline
curl -X POST http://localhost:7860/baseline
```
