param([switch]$AcceptDownloads, [switch]$WithTests)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
Write-Host 'Project-local installation only. Python 3.13 x64 required.'
Write-Host 'Sources: PyPI (runtime wheels roughly 105 MB; tests add roughly 40 MB),'
Write-Host 'GitHub/TRvlvr/model_repo UVR_MDXNET_Main.onnx: 66,759,214 bytes.'
Write-Host 'Installed size is larger. No GPU/CUDA, browser, global software, or paid service is installed.'
Write-Host 'Licenses: MIT/BSD/Apache dependencies; bundled FFmpeg is GPLv3. See THIRD_PARTY.md.'
if (-not $AcceptDownloads) {
    $answer = Read-Host 'Download/install these dependencies into this project? Type YES'
    if ($answer -cne 'YES') { Write-Host 'Cancelled. No installation performed.'; exit 1 }
}
$localPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $localPython)) {
    & py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python venv creation failed.' }
}
$requirementsFile = if ($WithTests) { 'requirements-dev.txt' } else { 'requirements.txt' }
& $localPython -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed; retry after fixing the network.' }
$modelDir = Join-Path $PSScriptRoot 'models'
New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
$modelFile = Join-Path $modelDir 'UVR_MDXNET_Main.onnx'
$expectedHash = '8289784CDA38543FF431ADD4070662813311A8CCCFC0112CA82F76D9DBA2B4CA'
if (-not (Test-Path -LiteralPath $modelFile) -or (Get-FileHash -LiteralPath $modelFile -Algorithm SHA256).Hash -ne $expectedHash) {
    $partFile = Join-Path $modelDir 'UVR_MDXNET_Main.onnx.part'
    Invoke-WebRequest -Uri 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR_MDXNET_Main.onnx' -OutFile $partFile -TimeoutSec 300
    if ((Get-FileHash -LiteralPath $partFile -Algorithm SHA256).Hash -ne $expectedHash) { throw 'Model checksum mismatch; partial file retained for inspection.' }
    Move-Item -LiteralPath $partFile -Destination $modelFile -Force
}
& $localPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
Write-Host 'Ready. Run .\start.ps1 and open http://127.0.0.1:8765'
