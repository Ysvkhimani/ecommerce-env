---
title: Ecommerce Env
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
python_version: "3.11"
startup_duration_timeout: 30m
---

# Ecommerce OpenEnv

An **OpenEnv**-compliant e-commerce cart simulation environment where an AI agent learns to complete a shopping workflow through a standard `reset()` / `step()` / `state()` API.

## Environment Description

The agent operates a shopping cart with four discrete actions. It receives partial reward at each step and a terminal reward on successful checkout. Three graded tasks (easy → medium → hard) measure progressively stricter completion criteria.

## Action Space

| Action | Effect | Reward |
|---|---|---|
| `add_item` | Add a $100 item to cart | +0.2 |
| `apply_coupon` | Apply 10% discount (once) | +0.3 (or −0.1 if already applied) |
| `checkout` | Proceed to checkout | +0.3 (or −0.5 if cart empty) |
| `pay` | Complete payment | +1.0 (or −1.0 if cart empty) |

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

| ID | Description | Difficulty |
|---|---|---|
| `easy` | Add item and pay | Easy — score 1.0 if paid |
| `medium` | Apply coupon then pay | Medium — 1.0 paid+coupon, 0.5 paid only |
| `hard` | Exact sequence: add_item → apply_coupon → checkout → pay | Hard — 1.0 exact, 0.5 if paid |

## HTTP API

All endpoints available on the deployed Space:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/tasks` | Task list + action schema |
| `POST` | `/reset` | Reset episode, get initial observation |
| `POST` | `/step` | Execute action `{"action": "add_item"}` |
| `GET` | `/state` | Full environment state |
| `GET` | `/grader` | Grader scores (easy / medium / hard) |
| `POST` | `/baseline` | Run optimal policy, return scores |
| `GET` | `/ui` | Gradio interactive UI |
| `GET` | `/docs` | Auto-generated OpenAPI docs |

## Setup Instructions

### Local (Docker)

```bash
docker build -t ecommerce-env .
docker run -p 7860:7860 ecommerce-env
# Open http://localhost:7860
```

### Local (Python)

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860/ui
```

### Baseline Inference

```bash
pip install -r requirements.txt
python baseline.py
```

Expected output:
```
easy:   1.0
medium: 1.0
hard:   1.0
```

## Example API Usage

```bash
# Reset
curl -X POST http://localhost:7860/reset

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "add_item"}'

# Get grader scores
curl http://localhost:7860/grader

# Run full baseline
curl -X POST http://localhost:7860/baseline
```
