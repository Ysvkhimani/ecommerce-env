#!/usr/bin/env bash
# Hugging Face Spaces build from branch "main" by default.
# Push your current branch to origin/main so the Space always gets the latest code.
set -euo pipefail
branch="$(git rev-parse --abbrev-ref HEAD)"
echo "Pushing ${branch} -> origin/main (Hugging Face Space)"
git push origin "${branch}:main"
