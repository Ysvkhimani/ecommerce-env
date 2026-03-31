#!/usr/bin/env bash
# Run the Gradio app locally using the pinned .venv-hf virtual environment.
# DO NOT use `uvicorn app:app` or `python app.py` directly — the global Python
# has huggingface_hub>=1.0 which removed HfFolder; Gradio 5.50 needs <=0.34.x.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -d "$REPO_ROOT/.venv-hf" ]]; then
    echo "ERROR: .venv-hf not found. Create it first:"
    echo "  python3.11 -m venv .venv-hf"
    echo "  .venv-hf/bin/pip install -r requirements.txt"
    exit 1
fi

echo "Starting app with .venv-hf (huggingface-hub 0.34.6, gradio 5.50.0)..."
cd "$REPO_ROOT"
exec .venv-hf/bin/python app.py
