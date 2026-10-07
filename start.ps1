param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $localPython)) {
    throw 'Local environment missing. Run setup.ps1 first.'
}
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be 1024..65535.' }
$address = "http://127.0.0.1:$Port"
try {
    $existing = Invoke-RestMethod -Uri "$address/api/health" -TimeoutSec 2
    if ($existing.privacy -eq 'local-only' -and $existing.model_name -eq 'UVR MDX-Net Main (ONNX / CPU)') {
        Write-Host "Singing Lab is already running: $address"
        if (-not $NoBrowser) { Start-Process $address }
        exit 0
    }
} catch { }
Write-Host "Singing Lab: $address"
Write-Host 'Local audio processing only. Press Ctrl+C in this window to stop.'
$browserJob = $null
if (-not $NoBrowser) {
    try {
        $browserJob = Start-Job -ScriptBlock {
            param($url)
            for ($attempt = 0; $attempt -lt 40; $attempt++) {
                try {
                    $status = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 2
                    if ($status.privacy -eq 'local-only') { Start-Process $url; break }
                } catch { }
                Start-Sleep -Milliseconds 250
            }
        } -ArgumentList $address
    } catch {
        Write-Warning "Browser could not open automatically. Open $address manually."
    }
}
try {
    & $localPython -m uvicorn backend.app:app --host 127.0.0.1 --port $Port --no-access-log
    exit $LASTEXITCODE
} finally {
    if ($browserJob) { Stop-Job $browserJob -ErrorAction SilentlyContinue; Remove-Job $browserJob -Force -ErrorAction SilentlyContinue }
}
