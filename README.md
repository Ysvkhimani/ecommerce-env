---
title: Ecommerce Env
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
base_path: /web
startup_duration_timeout: 30m
---

# Ecommerce OpenEnv

Interactive playground: **Gradio UI** at `/web`. REST API: `/docs` for Swagger.

If the Space stays on “Building” for a long time, open **Logs → Build** on the Space page: the first build downloads Python packages (often a few minutes). After changing `Dockerfile` or `requirements.txt`, trigger a **Factory reboot** or **Rebuild** so the old image is not reused.