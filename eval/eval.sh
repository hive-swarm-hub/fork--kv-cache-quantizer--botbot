#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="$(dirname "$SCRIPT_DIR")"

cd "$TASK_DIR"

# Activate venv if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Install deps if needed
uv pip install -q -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt 2>/dev/null || true

python eval/run_eval.py
