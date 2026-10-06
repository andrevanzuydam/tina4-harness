# Start JupyterLab with the login lock OFF (no token, no password) — for local use only.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path ".venv-notebook" "Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "No .venv-notebook yet — run .\install.ps1 first."
    exit 1
}

$port = if ($env:PORT) { $env:PORT } else { "8888" }
Write-Host "> JupyterLab on http://127.0.0.1:$port  (login lock OFF — local use only)"
# The full set is needed if you have ever set a Jupyter password: the saved hash
# otherwise forces a login page even with an empty token.
& $py -m jupyterlab notebooks\harness.ipynb `
  --IdentityProvider.token='' `
  --ServerApp.token='' `
  --ServerApp.password='' `
  --PasswordIdentityProvider.hashed_password='' `
  --PasswordIdentityProvider.password_required=`$false `
  --ServerApp.disable_check_xsrf=True `
  --no-browser --port $port
