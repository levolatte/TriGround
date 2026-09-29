# Keep the review server independent of a Codex command session.
# Preserve port and document root so existing browser drafts remain accessible.
$ErrorActionPreference = 'Stop'
$reviewRoot = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'results/triground_abv_20260927/review'
$logRoot = Join-Path (Split-Path $reviewRoot -Parent) 'verification'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
if (-not (Get-NetTCPConnection -LocalPort 8766 -State Listen -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath (Get-Command python).Source -WindowStyle Hidden `
        -ArgumentList @('-m', 'http.server', '8766', '--bind', '127.0.0.1') `
        -WorkingDirectory $reviewRoot `
        -RedirectStandardOutput (Join-Path $logRoot 'review-server.stdout.log') `
        -RedirectStandardError (Join-Path $logRoot 'review-server.stderr.log')
}
Write-Output 'http://127.0.0.1:8766/depth-v2/index.html'
