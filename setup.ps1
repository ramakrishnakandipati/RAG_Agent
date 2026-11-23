<#
  Windows-native setup for the rag-agent-gpu Conda environment.
  Run from the project root:
    powershell -ExecutionPolicy Bypass -File .\setup.ps1
  Non-interactive. Use -Recreate to delete and rebuild an existing env.
#>
[CmdletBinding()]
param(
    [switch]$Recreate,
    [string]$CondaRoot = "$env:USERPROFILE\anaconda3",
    [string]$EnvName   = "rag-agent-gpu",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param([string]$Exe, [string[]]$CmdArgs)
    Write-Host ">> $Exe $($CmdArgs -join ' ')" -ForegroundColor Cyan
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Exe $($CmdArgs -join ' ')" }
}

# --- 0. Locate conda (not assumed to be on PATH)
$conda = Join-Path $CondaRoot "Scripts\conda.exe"
if (-not (Test-Path $conda)) {
    $found = (Get-Command conda.exe -ErrorAction SilentlyContinue).Source
    if ($found) { $conda = $found } else { throw "conda.exe not found. Pass -CondaRoot <path>." }
}
Write-Host "Using conda: $conda"

# conda.exe called without activating base cannot find OpenSSL DLLs ("SSL module is not available").
# Put the Anaconda DLL/script dirs on PATH for this process only (same effect as the base activation hook).
$condaRootDir = Split-Path (Split-Path $conda -Parent) -Parent
$env:PATH = (@(
    $condaRootDir,
    "$condaRootDir\Library\mingw-w64\bin",
    "$condaRootDir\Library\usr\bin",
    "$condaRootDir\Library\bin",
    "$condaRootDir\Scripts"
) -join ";") + ";" + $env:PATH

# --- 1. GPU audit
Write-Host "`n=== NVIDIA GPU ===" -ForegroundColor Green
Invoke-Checked "nvidia-smi" @("--query-gpu=name,driver_version,memory.total", "--format=csv")

# --- 2. Short cache dirs (avoid MAX_PATH)
foreach ($d in "C:\hf_cache", "C:\pip_cache") { New-Item -ItemType Directory -Force -Path $d | Out-Null }
$env:HF_HOME = "C:\hf_cache"
$env:PIP_CACHE_DIR = "C:\pip_cache"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_NO_INPUT = "1"

# --- 3. Create env (conda-forge only; python + pip only)
$envPath = Join-Path $CondaRoot "envs\$EnvName"
if ((Test-Path $envPath) -and $Recreate) {
    Invoke-Checked $conda @("env", "remove", "-n", $EnvName, "-y")
}
if (-not (Test-Path (Join-Path $envPath "python.exe"))) {
    Invoke-Checked $conda @("create", "-n", $EnvName, "-c", "conda-forge", "--override-channels", "python=3.11", "pip", "-y")
}
$py = Join-Path $envPath "python.exe"   # equivalent to `conda activate`, but robust in non-interactive shells
Invoke-Checked $py @("--version")

# --- 4. GPU PyTorch first, from the official CUDA wheel index
Invoke-Checked $py @("-m", "pip", "install", "--no-input", "numpy<2.0", "torch", "torchvision", "torchaudio", "--index-url", $TorchIndex)

# --- 5. Remaining stack
Invoke-Checked $py @("-m", "pip", "install", "--no-input", "-r", "requirements.txt")
& $py -m pip check
if ($LASTEXITCODE -ne 0) { Write-Warning "pip check reported conflicts (see above)." }

# --- 6. CUDA sanity check
Invoke-Checked $py @("-c", "import torch,sys; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)")

# --- 7. Jupyter kernel
Invoke-Checked $py @("-m", "ipykernel", "install", "--user", "--name", $EnvName, "--display-name", "Python ($EnvName)")

# --- 8. .env
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env"; Write-Host "Created .env from .env.example" }

Write-Host "`nSetup complete. Interpreter: $py" -ForegroundColor Green
Write-Host "Verify with:  & '$py' verify_env.py"
