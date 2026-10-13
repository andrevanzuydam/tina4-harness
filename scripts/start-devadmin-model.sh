#!/bin/sh
# Run the shared local model consumed by `tina4 ai serve --backend http://127.0.0.1:11438`.
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export HARNESS_MLX_REPO="${HARNESS_MLX_REPO:-mlx-community/gemma-4-e4b-it-4bit}"
export HARNESS_MODEL_PORT="${HARNESS_MODEL_PORT:-11438}"
exec "${HARNESS_PYTHON:-python3}" -m harness.server
