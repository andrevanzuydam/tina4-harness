# Set up the notebook environment (.venv-notebook) on Windows (PowerShell).
# Safe to run again. Uses uv if present (fast), otherwise py -3.12 venv + pip — the same
# choice install.sh makes on macOS/Linux.
#
# GPU vs CPU is automatic:
#   * An NVIDIA GPU (nvidia-smi present) -> the CUDA build of torch + torchvision, so the
#     model runs on the GPU in float16 (much faster).
#   * No NVIDIA GPU -> the CPU build (works everywhere, just slower).
# Override with an env var before running:
#   $env:HARNESS_CUDA = "cu126"   # force a specific CUDA build (see your driver's CUDA version)
#   $env:HARNESS_CUDA = "cpu"     # force the CPU build even though a GPU is present
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venv = ".venv-notebook"
$py = Join-Path $venv "Scripts\python.exe"
$useUv = [bool](Get-Command uv -ErrorAction SilentlyContinue)

if ($useUv) {
    Write-Host "> creating $venv with uv (Python 3.12)..."
    uv venv --python 3.12 $venv
} else {
    Write-Host "> creating $venv with py -3.12 venv..."
    py -3.12 -m venv $venv
    & $py -m pip install --upgrade pip
}

# --- choose the torch build: GPU (CUDA) if an NVIDIA GPU is present, else CPU --------------
$cuda = $env:HARNESS_CUDA
if (-not $cuda) {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $cuda = "cu128" } else { $cuda = "cpu" }
}

if ($cuda -ne "cpu") {
    $index = "https://download.pytorch.org/whl/$cuda"
    Write-Host "> NVIDIA GPU detected -> installing the CUDA ($cuda) build of torch + torchvision."
    Write-Host "  (override with `$env:HARNESS_CUDA='cuXXX', or `$env:HARNESS_CUDA='cpu' to force CPU)"
    if ($useUv) {
        uv pip install -p $py --index-url $index torch torchvision
    } else {
        & $py -m pip install --index-url $index torch torchvision
    }
} else {
    Write-Host "> no NVIDIA GPU (or HARNESS_CUDA=cpu) -> using the CPU build of torch."
}

# --- the rest of the notebook requirements (torch already satisfied if CUDA was installed) -
Write-Host "> installing notebook requirements..."
if ($useUv) {
    uv pip install -p $py -r requirements-notebook.txt
} else {
    & $py -m pip install -r requirements-notebook.txt
}

Write-Host "> registering the Jupyter kernel..."
& $py -m ipykernel install --user --name tina4-harness --display-name "Python 3 (tina4-harness)" | Out-Null

# Build and install the right-click JupyterLab extension (Check Code / Create Cell / Fix Code).
if (Get-Command node -ErrorAction SilentlyContinue) {
    Write-Host "> building the right-click extension (labext)..."
    $jlpm = Join-Path $PSScriptRoot "$venv\Scripts\jlpm.exe"
    $builder = Join-Path $PSScriptRoot "$venv\Scripts\jupyter-builder.exe"
    Push-Location labext
    try {
        & $jlpm install
        & $jlpm run build:lib:prod
        & $builder build .
    } finally {
        Pop-Location
    }
    $dst = Join-Path $venv "share\jupyter\labextensions\tina4-harness-lab"
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Copy-Item -Recurse -Force "labext\tina4_harness_lab\labextension\*" $dst
    Write-Host "  right-click menus installed"
} else {
    Write-Host "! Node.js not found - skipping the right-click menus. The notebook still works fully;"
    Write-Host "  install Node 18+ and re-run .\install.ps1 to add them."
}

Write-Host "done. Start JupyterLab with:  .\run.ps1"
