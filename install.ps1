# Set up the notebook environment (.venv-notebook) on Windows (PowerShell).
# Safe to run again. On Windows the model runs on PyTorch/transformers (CPU, or CUDA if present).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venv = ".venv-notebook"
Write-Host "> creating $venv (Python 3.12)..."
py -3.12 -m venv $venv
$py = Join-Path $venv "Scripts\python.exe"

Write-Host "> installing notebook requirements..."
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements-notebook.txt

Write-Host "> registering the Jupyter kernel..."
& $py -m ipykernel install --user --name tina4-harness --display-name "Python 3 (tina4-harness)" | Out-Null

Write-Host "done. Start JupyterLab with:  .\run.ps1"
