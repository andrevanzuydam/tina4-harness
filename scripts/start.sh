#!/usr/bin/env bash
# Start the harness: ONE shared model server, then JupyterLab. Every notebook kernel inherits
# HARNESS_MODEL_SERVER and talks to the single model process (so the 2B lives in ~1.6 GB total,
# not once per notebook). Threads default to 8.
#
#   ./scripts/start.sh            # model server on :7777 + JupyterLab on :8890
#
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

PY="$ROOT/.venv-notebook/bin/python"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HARNESS_MODEL_PORT="${HARNESS_MODEL_PORT:-7777}"
export HARNESS_THREADS="${HARNESS_THREADS:-8}"
export HARNESS_MODEL_SERVER="http://127.0.0.1:${HARNESS_MODEL_PORT}"
JUPYTER_PORT="${JUPYTER_PORT:-8890}"

mkdir -p "$ROOT/logs"

# 1) Shared model server (skip if one is already answering).
if curl -s -o /dev/null --max-time 2 "$HARNESS_MODEL_SERVER/health"; then
  echo "✓ model server already running at $HARNESS_MODEL_SERVER"
else
  echo "▶ starting shared model server ($HARNESS_THREADS threads) on $HARNESS_MODEL_SERVER …"
  nohup "$PY" -m harness.server > "$ROOT/logs/model-server.log" 2>&1 &
  echo "  pid $! · log: logs/model-server.log"
  # Wait for the model to finish loading (first boot loads the weights).
  for i in $(seq 1 120); do
    if curl -s -o /dev/null --max-time 2 "$HARNESS_MODEL_SERVER/health"; then
      echo "✓ model server ready: $(curl -s "$HARNESS_MODEL_SERVER/memory")"
      break
    fi
    sleep 1
  done
fi

# 2) JupyterLab — rooted at notebooks/ so every notebook shares one cwd (the notebook space),
#    and kernels inherit HARNESS_MODEL_SERVER to share the one model.
NB_ROOT="${NB_ROOT:-$ROOT/notebooks}"
mkdir -p "$NB_ROOT"
echo "▶ starting JupyterLab on http://localhost:${JUPYTER_PORT}/lab  (root: $NB_ROOT) …"
exec "$PY" -m jupyterlab \
  --ServerApp.root_dir="$NB_ROOT" \
  --IdentityProvider.token= --ServerApp.token= --ServerApp.password= \
  --PasswordIdentityProvider.hashed_password= --PasswordIdentityProvider.password_required=False \
  --ServerApp.disable_check_xsrf=True --no-browser --port "$JUPYTER_PORT"
