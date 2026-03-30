---
title: Ecommerce Env
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
startup_duration_timeout: 30m
---

# Ecommerce OpenEnv

This Space uses the **Gradio** SDK (no Docker), so it usually builds in **a few minutes** instead of waiting on a Docker image.

Use **Reset** → pick an action → **Step**. **Run grader** scores the current cart state (easy / medium / hard).

**Logs:** open **Logs → Container** (not only Build). Build logs show `pip install`; runtime lines start with `ecommerce-env: app.py`.

**Blank Space / no logs:** if `huggingface_hub` upgraded to 1.x, Gradio 4 fails on import (`HfFolder`). This repo pins `huggingface-hub==0.24.7` and `gradio-client==1.3.0` in `requirements.txt` — do not remove those pins.
