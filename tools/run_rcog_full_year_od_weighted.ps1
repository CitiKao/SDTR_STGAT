$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:TERM = "xterm-256color"

$Repo = "D:\STDR"
$PythonExe = "C:\Users\WMNL1691\Anaconda3\envs\stable_diffusion\python.exe"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $Repo "runs\rcog_full_year_od_weighted_$Stamp"
$Log = Join-Path $RunDir "terminal.log"
$Summary = Join-Path $RunDir "rcog_full_year_od_weighted_summary.json"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Set-Content -Path (Join-Path $Repo "runs\latest_rcog_full_year_od_weighted_run.txt") -Value $RunDir

Write-Host "======================================================================================================================" -ForegroundColor Cyan
Write-Host "RCOG full-year OD-time weighted evaluation started" -ForegroundColor Cyan
Write-Host "RunDir:  $RunDir" -ForegroundColor Yellow
Write-Host "Log:     $Log" -ForegroundColor Yellow
Write-Host "Summary: $Summary" -ForegroundColor Yellow
Write-Host "======================================================================================================================" -ForegroundColor Cyan

Set-Location $Repo

$PythonArgs = @(
  "D:\STDR\rcog_full_year_od_weighted_eval.py",
  "--checkpoint", "D:\STDR\runs\candidate_route_reranker_stgat_opportunity_gpu_pool300_50k\ddqn_reranker_final.pt",
  "--train-config", "D:\STDR\runs\candidate_route_reranker_stgat_opportunity_gpu_pool300_50k\config.json",
  "--run-dir", $RunDir,
  "--seeds", "121,122,123,124,125",
  "--train-resample-mode", "bootstrap",
  "--train-weight-mode", "none",
  "--bad-positive-weight-mult", "2.0",
  "--max-bad-gt-5pct", "0.30",
  "--target-bad-gt-5pct", "0.50",
  "--max-activation", "11",
  "--min-benefit-precision", "64",
  "--selection-objective", "balanced",
  "--threshold-search-grid", "dense",
  "--bootstrap-iters", "1000",
  "--profile-batch-size", "24",
  "--record-log-interval", "5000",
  "--sweep-log-interval", "20000",
  "--device", "cuda"
)

& $PythonExe @PythonArgs 2>&1 | Tee-Object -FilePath $Log
$ExitCode = $LASTEXITCODE

if ($ExitCode -eq 0) {
  Write-Host "======================================================================================================================" -ForegroundColor Green
  Write-Host "RCOG full-year OD-time weighted evaluation finished successfully." -ForegroundColor Green
  Write-Host "Summary: $Summary" -ForegroundColor Green
  Write-Host "======================================================================================================================" -ForegroundColor Green
} else {
  Write-Host "======================================================================================================================" -ForegroundColor Red
  Write-Host "RCOG full-year OD-time weighted evaluation failed with exit code $ExitCode." -ForegroundColor Red
  Write-Host "Check log: $Log" -ForegroundColor Red
  Write-Host "======================================================================================================================" -ForegroundColor Red
}
