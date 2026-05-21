$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:TERM = "xterm-256color"

$Repo = "D:\STDR"
$PythonExe = "C:\Users\WMNL1691\Anaconda3\envs\stable_diffusion\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Repo "runs\rcog_v3_focused_fullscale_$Stamp"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$Log = Join-Path $RunDir "terminal.log"
$Summary = Join-Path $RunDir "rcog_v3_focused_fullscale_summary.json"

Write-Host "========================================================================================================" -ForegroundColor Cyan
Write-Host "RCOG v3 focused full-scale started" -ForegroundColor Cyan
Write-Host "Variants: larger_gate_train10000 + dynamic_bucket_threshold" -ForegroundColor Cyan
Write-Host "RunDir:  $RunDir" -ForegroundColor Yellow
Write-Host "Log:     $Log" -ForegroundColor Yellow
Write-Host "Summary: $Summary" -ForegroundColor Yellow
Write-Host "========================================================================================================" -ForegroundColor Cyan

Set-Location $Repo

$PythonArgs = @(
  "D:\STDR\rcog_v3_focused_fullscale.py",
  "--checkpoint", "D:\STDR\runs\candidate_route_reranker_stgat_opportunity_gpu_pool300_50k\ddqn_reranker_final.pt",
  "--train-config", "D:\STDR\runs\candidate_route_reranker_stgat_opportunity_gpu_pool300_50k\config.json",
  "--run-dir", $RunDir,
  "--train-pool-size", "10000",
  "--eval-pool-size", "3000",
  "--candidate-build-log-interval", "250",
  "--candidate-pool-attempt-multiplier", "1800",
  "--model-iter", "160",
  "--device", "cuda",
  "--no-require-candidate-oracle-diff"
)

& $PythonExe @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
  Write-Host "========================================================================================================" -ForegroundColor Green
  Write-Host "Focused full-scale finished successfully." -ForegroundColor Green
  Write-Host "Summary: $Summary" -ForegroundColor Green
  Write-Host "========================================================================================================" -ForegroundColor Green
} else {
  Write-Host "========================================================================================================" -ForegroundColor Red
  Write-Host "Focused full-scale failed with exit code $ExitCode." -ForegroundColor Red
  Write-Host "Check log: $Log" -ForegroundColor Red
  Write-Host "========================================================================================================" -ForegroundColor Red
}
