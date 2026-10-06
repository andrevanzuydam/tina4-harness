#!/usr/bin/env bash
# Start JupyterLab with the login lock OFF (no token, no password) — for local use only.
# Opens the harness notebook. Change the port with PORT=8899 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv-notebook/bin/python"
if [ ! -x "$PY" ]; then
  echo "No .venv-notebook yet — run ./install.sh first." >&2
  exit 1
fi

PORT="${PORT:-8888}"
echo "▶ JupyterLab on http://127.0.0.1:${PORT}  (login lock OFF — local use only)"
# The full set is needed if you have ever set a Jupyter password: the saved hash
# otherwise forces a login page even with an empty token.
exec "$PY" -m jupyterlab notebooks/harness.ipynb \
  --IdentityProvider.token='' \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --PasswordIdentityProvider.hashed_password='' \
  --PasswordIdentityProvider.password_required=False \
  --ServerApp.disable_check_xsrf=True \
  --no-browser --port "$PORT"
