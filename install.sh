#!/usr/bin/env bash
# Set up the notebook environment (.venv-notebook) on macOS or Linux.
# Safe to run again. Uses uv if present (fast), otherwise python venv + pip.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-notebook"
if command -v uv >/dev/null 2>&1; then
  echo "▶ creating $VENV with uv (Python 3.12)…"
  uv venv --python 3.12 "$VENV"
  PY="$VENV/bin/python"
  echo "▶ installing notebook requirements…"
  uv pip install -p "$PY" -r requirements-notebook.txt
else
  echo "▶ creating $VENV with python venv…"
  python3 -m venv "$VENV"
  PY="$VENV/bin/python"
  echo "▶ installing notebook requirements…"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements-notebook.txt
fi

echo "▶ registering the Jupyter kernel…"
"$PY" -m ipykernel install --user --name tina4-harness --display-name "Python 3 (tina4-harness)" >/dev/null

echo "✓ done. Start JupyterLab with:  ./run.sh"
