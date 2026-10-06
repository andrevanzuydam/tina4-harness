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

# Build and install the right-click JupyterLab extension (Check Code / Create Cell / Fix Code).
if command -v node >/dev/null 2>&1; then
  echo "▶ building the right-click extension (labext)…"
  ROOT="$(pwd)"
  (
    cd labext
    "$ROOT/$VENV/bin/jlpm" install
    "$ROOT/$VENV/bin/jlpm" run build:lib:prod
    PATH="$ROOT/$VENV/bin:$PATH" jupyter labextension build .
  )
  DST="$VENV/share/jupyter/labextensions/tina4-harness-lab"
  rm -rf "$DST"; mkdir -p "$DST"
  cp -R labext/tina4_harness_lab/labextension/* "$DST/"
  echo "✓ right-click menus installed"
else
  echo "⚠ Node.js not found — skipping the right-click menus. The notebook still works fully;"
  echo "  install Node 18+ and re-run ./install.sh to add them."
fi

echo "✓ done. Start JupyterLab with:  ./run.sh"
