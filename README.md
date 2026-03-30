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

**Logs:** open **Logs → Container** (not only Build) to see app startup lines such as `ecommerce-env: loading app.py`. Build logs mostly show `pip install`; runtime messages go to Container.
