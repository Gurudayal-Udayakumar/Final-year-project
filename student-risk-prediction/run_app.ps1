<#
Run Streamlit using the project's virtualenv Python so the correct packages (like plotly) are used.
Usage:
  # Run the app (uses .venv/Scripts/python.exe)
  .\run_app.ps1

  # Install requirements into the venv first, then run
  .\run_app.ps1 -InstallRequirements
#>
param(
    [switch]$InstallRequirements
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPython = Join-Path $scriptDir '.venv\Scripts\python.exe'
$requirements = Join-Path $scriptDir 'requirements.txt'
$appMain = Join-Path $scriptDir 'app\main.py'

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtualenv python not found at $venvPython. Create or point to your venv first."
    exit 1
}

if ($InstallRequirements) {
    Write-Host "Installing requirements into virtualenv..."
    & $venvPython -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { Write-Error "pip install failed"; exit $LASTEXITCODE }
}

Write-Host "Starting Streamlit with virtualenv python: $venvPython"
# Ensure current directory is the project root so imports like `from utils...` work
Push-Location $scriptDir
try {
    # Prepend project root to PYTHONPATH for child processes so pages can import sibling packages
    $oldPYTHONPATH = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
    if ([string]::IsNullOrEmpty($oldPYTHONPATH)) {
        [Environment]::SetEnvironmentVariable('PYTHONPATH', $scriptDir, 'Process')
    } else {
        [Environment]::SetEnvironmentVariable('PYTHONPATH', "$scriptDir;$oldPYTHONPATH", 'Process')
    }

    & $venvPython -m streamlit run $appMain
} finally {
    Pop-Location
}
