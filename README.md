---
title: Ecommerce Env
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.11"
startup_duration_timeout: 30m
---

# Ecommerce OpenEnv

This Space uses the **Gradio** SDK (no Docker), so it usually builds in **a few minutes** instead of waiting on a Docker image.

Use **Reset** → pick an action → **Step**. **Run grader** scores the current cart state (easy / medium / hard).

**Logs:** open **Logs → Container** (not only Build). Build logs show `pip install`; runtime lines start with `ecommerce-env: app.py`.

**Gradio / hub versions:** `sdk_version` in this file must match the `gradio==…` line in `requirements.txt` (Hugging Face installs `gradio[oauth]` from that version). This repo uses **Gradio 5.50** with `huggingface-hub==0.34.6` so the Space image stays compatible with the preinstalled `datasets` package (`huggingface-hub>=0.25`).

**HTTP 500:** after changing versions, push to `main` and wait for a full rebuild; then hard-refresh the Space. Check **Logs → Container** for Python tracebacks.
