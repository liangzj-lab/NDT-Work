$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
# Other workstation:
# $python = "C:\Users\Administrator\.conda\envs\pytorch_env\python.exe"

# Current workstation:
$python = "D:\Anaconda\envs\PyTorch\python.exe"
$server = Join-Path $repoRoot "ZJDKY\DL\codex\region_detector_web_server.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing Python interpreter: $python"
}
if (-not (Test-Path -LiteralPath $server)) {
    throw "Missing server script: $server"
}

Set-Location $repoRoot
Write-Host "Starting region detector test UI..."
Write-Host "Open: http://127.0.0.1:8765/"
Write-Host "Press Ctrl+C in this window to stop the service."

& $python $server --host 127.0.0.1 --port 8765
