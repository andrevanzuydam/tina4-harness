#!/usr/bin/env bash
# Set up the notebook environment (.venv-notebook) on macOS or Linux.
# Safe to run again. Uses uv if present (fast), otherwise python venv + pip.
#
# GPU vs CPU is automatic:
#   * Apple Silicon          -> MLX (Metal GPU) via mlx-vlm — handled by requirements markers.
#   * Linux with an NVIDIA GPU (nvidia-smi present) -> the CUDA build of torch + torchvision,
#                               so the model runs on the GPU in float16 (much faster).
#   * Everything else        -> the CPU build of torch (works everywhere, just slower).
# Override with an env var before running:
#   HARNESS_CUDA=cu126 ./install.sh   # force a specific CUDA build (match your driver)
#   HARNESS_CUDA=cpu   ./install.sh   # force the CPU build even though a GPU is present
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-notebook"
if command -v uv >/dev/null 2>&1; then
  echo "▶ creating $VENV with uv (Python 3.12)…"
  uv venv --python 3.12 "$VENV"
  PY="$VENV/bin/python"
  PIP_INSTALL=(uv pip install -p "$PY")
else
  echo "▶ creating $VENV with python venv…"
  python3 -m venv "$VENV"
  PY="$VENV/bin/python"
  "$PY" -m pip install --upgrade pip
  PIP_INSTALL=("$PY" -m pip install)
fi

# --- choose the torch build: GPU (CUDA) if an NVIDIA GPU is present, else CPU --------------
# (On Apple Silicon there is no nvidia-smi, so CUDA=cpu and torch is skipped entirely — the
#  requirements markers install mlx-vlm instead.)
CUDA="${HARNESS_CUDA:-}"
if [ -z "$CUDA" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then CUDA="cu128"; else CUDA="cpu"; fi
fi

if [ "$CUDA" != "cpu" ]; then
  echo "▶ NVIDIA GPU detected → installing the CUDA ($CUDA) build of torch + torchvision."
  echo "  (override with HARNESS_CUDA=cuXXX, or HARNESS_CUDA=cpu to force CPU)"
  "${PIP_INSTALL[@]}" --index-url "https://download.pytorch.org/whl/$CUDA" torch torchvision
else
  echo "▶ no NVIDIA GPU (or HARNESS_CUDA=cpu) → CPU torch on Intel, MLX on Apple Silicon."
fi

# --- the rest of the notebook requirements (torch already satisfied if CUDA was installed) -
echo "▶ installing notebook requirements…"
"${PIP_INSTALL[@]}" -r requirements-notebook.txt

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
    "$ROOT/$VENV/bin/jupyter-builder" build .
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
