# feast AI launcher (Windows PowerShell)
# Installs dependencies on first run, then opens the TUI.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import textual, psutil, httpx" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    python -m pip install -r requirements.txt
}

python -m feast_ai @args
